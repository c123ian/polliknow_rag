from fasthtml.common import *
import modal
import fastapi
import logging
import uuid
import asyncio
import aiohttp
import os
import io
import sqlite3
import torch
import base64
from modal import Secret
from fastlite import Database
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import json

# Constants
QWEN_MODELS_DIR = "/Qwen"  # existing volume
COLQWEN_MODELS_DIR = "/ColQwen"  # new volume for ColQwen2 models
DATA_DIR = "/bee_pdf"
TEMP_UPLOAD_DIR = "/bee_pdf/temp_uploads"
DEFAULT_QWEN_NAME = "Qwen/Qwen2.5-7B-Instruct-1M"
DEFAULT_COLQWEN_NAME = "vidore/colqwen2-v1.0"
USERNAME = "c123ian"
APP_NAME = "polliknow-rag"
DATABASE_DIR = "/db_rag_advan"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Setup database
db_path = os.path.join(DATABASE_DIR, 'image_analysis.db')
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_analyses (
        analysis_id TEXT PRIMARY KEY,
        image_path TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        query TEXT NOT NULL,
        response TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()

# Set up volumes
try:
    qwen_volume = modal.Volume.lookup("Qwen", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download Qwen models first with the appropriate script")

try:
    colqwen_volume = modal.Volume.lookup("ColQwen", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download ColQwen models first with the appropriate script")

try:
    bee_volume = modal.Volume.from_name("bee_pdf", create_if_missing=True)
except modal.exception.NotFoundError:
    bee_volume = modal.Volume.persisted("bee_pdf")

try:
    db_volume = modal.Volume.lookup("db_data", create_if_missing=True)
except modal.exception.NotFoundError:
    db_volume = modal.Volume.persisted("db_data")

# Setup image with required dependencies
image = modal.Image.debian_slim(python_version="3.10") \
    .pip_install(
        "vllm==0.7.2",
        "python-fasthtml==0.4.3",
        "aiohttp",
        "faiss-cpu",
        "sentence-transformers",
        "pandas",
        "numpy",
        "huggingface_hub",
        "transformers==4.48.3",
        "rerankers",
        "sqlite-minutils",
        "rank-bm25",
        "nltk",
        "sqlalchemy",
        "pdf2image",
        "colpali-engine[interpretability]>=0.3.2",  # Add interpretability extras
        "torch",
        "matplotlib"
    )

app = modal.App(APP_NAME)

# Create app function for Qwen LLM serving
@app.function(
    image=image,
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={
        QWEN_MODELS_DIR: qwen_volume,
        COLQWEN_MODELS_DIR: colqwen_volume,
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume
    },
)
@modal.asgi_app()
def serve_vllm():
    import os
    import asyncio
    import fastapi
    import uuid
    from fastapi.responses import StreamingResponse, JSONResponse
    from typing import Optional

    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels
    from vllm.entrypoints.logger import RequestLogger
    from vllm.sampling_params import SamplingParams

    MODELS_DIR = "/Qwen"

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {DEFAULT_QWEN_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM",
        version="0.0.1",
        docs_url="/docs",
    )

    def find_model_path(base_dir):
        for root, _, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    def find_tokenizer_path(base_dir):
        for root, _, files in os.walk(base_dir):
            if "tokenizer_config.json" in files:
                return root
        return None

    model_path = find_model_path(MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MODELS_DIR}")

    tokenizer_path = find_tokenizer_path(MODELS_DIR)
    if not tokenizer_path:
        raise Exception(f"Could not find tokenizer files in {MODELS_DIR}")

    logging.info(f"Initializing AsyncLLMEngine with model path: {model_path} and tokenizer path: {tokenizer_path}")

    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=94192
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    event_loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if event_loop and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())

    models = OpenAIServingModels(engine_client=engine, model_config=model_config, base_model_paths={DEFAULT_QWEN_NAME: model_path})
    request_logger = RequestLogger(max_log_len=256)

    openai_serving_chat = OpenAIServingChat(
        engine_client=engine,
        model_config=model_config,
        models=models,
        response_role="assistant",
        request_logger=request_logger,
        chat_template=None,  
        chat_template_content_format="string",  
    )

    @web_app.post("/v1/completions")
    async def completion_generator(request: fastapi.Request) -> StreamingResponse:
        try:
            body = await request.json()
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 100)
            stream = body.get("stream", True)
            request_id = str(uuid.uuid4())

            logging.info(f"Received completion request: max_tokens={max_tokens}, stream={stream}")

            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens,
                stop=["User:", "Assistant:", "\n\n"],
            )

            if stream:
                async def generate_text():
                    full_response = ""
                    last_yielded_position = 0
                    assistant_prefix_removed = False
                    buffer = ""

                    async for result in engine.generate(prompt, sampling_params, request_id):
                        if len(result.outputs) > 0:
                            new_text = result.outputs[0].text

                            if not assistant_prefix_removed:
                                new_text = new_text.split("Assistant:")[-1].lstrip()
                                assistant_prefix_removed = True

                            if len(new_text) > last_yielded_position:
                                new_part = new_text[last_yielded_position:]
                                buffer += new_part

                                words = buffer.split()
                                if len(words) > 1:
                                    to_yield = ' '.join(words[:-1]) + ' '
                                    for punct in ['.', '!', '?']:
                                        to_yield = to_yield.replace(f"{punct}", f"{punct} ")
                                    to_yield = ' '.join(to_yield.split())
                                    buffer = words[-1]
                                    yield to_yield + ' '

                                last_yielded_position = len(new_text)

                            full_response = new_text

                    if buffer:
                        for punct in ['.', '!', '?']:
                            buffer = buffer.replace(f"{punct}", f"{punct} ")
                        buffer = ' '.join(buffer.split())
                        yield buffer

                return StreamingResponse(generate_text(), media_type="text/plain")
            else:
                # Non-streaming response
                outputs = await engine.generate(prompt, sampling_params, request_id)
                if len(outputs.outputs) > 0:
                    response_text = outputs.outputs[0].text
                    # Remove Assistant: prefix if present
                    response_text = response_text.split("Assistant:")[-1].lstrip()
                    
                    return JSONResponse(content={
                        "id": request_id,
                        "choices": [
                            {
                                "text": response_text,
                                "finish_reason": "stop"
                            }
                        ]
                    })
                else:
                    return JSONResponse(content={"error": "No output generated"}, status_code=500)
                
        except Exception as e:
            logging.error(f"Error in completion_generator: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app

# Create app function for FastHTML UI and ColQwen processing
@app.function(
    image=image,
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    volumes={
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume,
        COLQWEN_MODELS_DIR: colqwen_volume
    },
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    import os
    import pickle
    import pandas as pd
    import logging
    from starlette.middleware import Middleware
    from starlette.websockets import WebSocket
    import uuid
    import datetime
    from sqlalchemy import create_engine, Column, String, DateTime, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from fastapi.responses import FileResponse, Response, HTMLResponse
    from io import BytesIO
    import base64
    from PIL import Image
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
    
    # Helper function to ensure ColQwen model is loaded
    def ensure_colqwen_model_loaded():
        """Ensure the ColQwen2 model is properly loaded"""
        global colpali_model, colpali_processor
        
        # Verify CUDA is available
        if torch.cuda.is_available():
            logging.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("WARNING: CUDA is NOT available, ColQwen will run on CPU which is slower")
        
        # Check if we need to load the model
        if 'colpali_model' not in globals() or colpali_model is None:
            logging.info(f"Loading ColQwen2 model...")
            try:
                # First try to find model in volume
                model_path = os.path.join(COLQWEN_MODELS_DIR, os.path.basename(DEFAULT_COLQWEN_NAME))
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    logging.info(f"Using local model from volume: {model_path}")
                    colpali_model = ColQwen2.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda" if torch.cuda.is_available() else "cpu"
                    ).eval()
                    colpali_processor = ColQwen2Processor.from_pretrained(model_path)
                else:
                    # Fall back to HuggingFace download
                    logging.info(f"Model not found in volume, downloading from HuggingFace: {DEFAULT_COLQWEN_NAME}")
                    colpali_model = ColQwen2.from_pretrained(
                        DEFAULT_COLQWEN_NAME,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda" if torch.cuda.is_available() else "cpu"
                    ).eval()
                    colpali_processor = ColQwen2Processor.from_pretrained(DEFAULT_COLQWEN_NAME)
                
                logging.info(f"ColQwen2 model loaded successfully on device: {colpali_model.device}")
                return True
            except Exception as e:
                logging.error(f"Error loading ColQwen2 model: {e}")
                import traceback
                traceback.print_exc()
                return False
        return True

    # Make temporary directories
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

    # Setup database connection
    db_engine = create_engine(f'sqlite:///{db_path}')
    Base = declarative_base()
    Session = sessionmaker(bind=db_engine)
    sqlalchemy_session = Session()

    class ImageAnalysis(Base):
        __tablename__ = 'image_analyses'
        analysis_id = Column(String, primary_key=True)
        image_path = Column(String, nullable=False)
        analysis_type = Column(String, nullable=False)
        query = Column(String, nullable=False)
        response = Column(String)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Initialize the FastHTML app
    fasthtml_app, rt = fast_app(
        hdrs=(
            # Explicit HTMX loading and other scripts
            Script(src="https://unpkg.com/htmx.org@1.9.6"),
            Script(src="https://cdn.tailwindcss.com"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"),
            # Add custom CSS for better interactive elements
            Style("""
                button, .btn, a[href] {
                    cursor: pointer !important;
                }
                
                button:hover, .btn:hover, a[href]:hover {
                    opacity: 0.9;
                }
                
                .file-input {
                    cursor: pointer !important;
                }
                
                /* Ensure nothing blocks interaction */
                #main-content, #analysis-results {
                    position: relative;
                    z-index: 1;
                }
                
                /* Only show loading indicator when active */
                #loading-indicator:not(.htmx-request) {
                    display: none !important;
                }
                
                /* Custom animations */
                .fade-in {
                    animation: fadeIn 0.5s;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
            """),
        ),
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=os.environ.get('YOUR_KEY', 'default-secret-key'),
                session_cookie="secure_session",
                max_age=86400,
                same_site="strict",
                https_only=True
            )
        ]
    )

    # Define query templates by analysis type
    def get_query_for_analysis(analysis_type):
        """Return a predefined query based on the analysis type"""
        queries = {
            "classify_insect": "Classify the insect in this image based on taxonomic hierarchy. Identify its order, family, and genus if possible.",
            "identify_species": "Identify the species shown in this image. Describe its key visual characteristics and habitat.",
            "analyze_features": "Analyze the visible features of this organism. Describe its morphology, coloration, and any distinctive anatomical structures."
        }
        return queries.get(analysis_type, queries["classify_insect"])

    # Define UI components
    def upload_form():
        """Render upload form for images with proper interactive elements"""
        # File input with visual styling
        upload_input = Input(
            type="file",
            name="image",
            accept=".jpg,.jpeg,.png",
            required=True,
            cls="file-input file-input-bordered file-input-secondary w-full"
        )
        
        # Analysis type selection
        analysis_selector = Div(
            Label("Select Analysis Type:", cls="text-white text-sm font-medium mb-2"),
            Select(
                Option("Classify Insect", value="classify_insect", selected=True),
                Option("Identify Species", value="identify_species"),
                Option("Analyze Features", value="analyze_features"),
                name="analysis_type",
                cls="select select-bordered w-full mb-4"
            ),
            cls="mb-4"
        )
        
        # Process button with loading state - using direct form action to ensure it works
        process_button = Button(
            "Analyze Image",
            type="submit",
            id="analyze-button",
            cls="btn btn-primary w-full hover:bg-blue-600 active:bg-blue-700"
        )
        
        # Instructions
        instructions = Div(
            P("Upload an image of an insect or other species for AI analysis.", 
              cls="text-zinc-300 text-center mb-2"),
            P("The system will analyze the image and provide visual insights into how the AI interprets it.",
              cls="text-zinc-300 text-center text-sm"),
            cls="mt-4 mb-6"
        )
        
        # Create form with action and method (for non-HTMX fallback) and HTMX attributes
        return Form(
            Div(
                H2("Upload Image for Analysis", cls="text-xl font-semibold text-white mb-4"),
                instructions,
                
                # Image upload card
                Div(
                    Div(
                        Label("Upload Image:", cls="text-white font-medium mb-2"),
                        upload_input,
                        cls="grid place-items-center p-4"
                    ),
                    cls="card bg-zinc-800 border border-zinc-700 rounded-box w-full mb-4"
                ),
                
                # Analysis selector & Process button
                analysis_selector,
                process_button,
                
                cls="bg-zinc-900 rounded-md p-6 w-full max-w-lg border border-zinc-700"
            ),
            action="/process-image",  # Direct action for fallback
            method="post",            # Direct method for fallback
            enctype="multipart/form-data",
            id="upload-form",
            hx_post="/process-image",
            hx_target="#analysis-results",
            hx_indicator="#loading-indicator",
        )

    def analysis_results_ui(image_path, qwen_response, token_maps, analysis_id):
        """Create UI to display analysis results"""
        
        # Create token map tabs
        token_tabs = None
        if token_maps:
            # Create tab-style navigation for token maps
            tab_buttons = [
                Button(
                    map_data['token'],
                    hx_get=f"/get-token-map/{os.path.basename(map_data['path'])}",
                    hx_target="#token-map-display",
                    hx_trigger="click",
                    cls=f"btn {'btn-primary' if i == 0 else 'btn-outline btn-secondary'} btn-sm m-1",
                    id=f"token-tab-{i}"
                )
                for i, map_data in enumerate(token_maps)
            ]
            
            # Main token exploration UI
            token_tabs = Div(
                H3("Token Similarity Maps", cls="text-xl font-semibold text-white mb-2"),
                P("See how the model focuses on different parts of the image for each word in the query:", 
                  cls="text-zinc-300 mb-3"),
                
                # Token tabs navigation
                Div(
                    *tab_buttons,
                    cls="flex flex-wrap gap-2 mb-4"
                ),
                
                # Token map display area
                Div(
                    Div("Select a token above to see its heatmap", cls="text-zinc-400 text-center py-8"),
                    id="token-map-display",
                    cls="w-full bg-zinc-700 rounded-md overflow-hidden"
                ),
                
                Script("""
                // Set up token tab clicks
                document.addEventListener('htmx:afterSwap', function() {
                    if (document.getElementById('token-tab-0')) {
                        setTimeout(function() {
                            document.getElementById('token-tab-0').click();
                        }, 300);
                    }
                });
                
                document.addEventListener('htmx:afterRequest', function(evt) {
                    if (evt.detail.target && evt.detail.target.id === 'token-map-display') {
                        // Remove active class from all tabs
                        document.querySelectorAll('[id^="token-tab-"]').forEach(function(tab) {
                            tab.classList.remove('btn-primary');
                            tab.classList.add('btn-outline', 'btn-secondary');
                        });
                        
                        // Add active class to clicked tab
                        if (evt.detail.requestConfig.triggeringElement) {
                            evt.detail.requestConfig.triggeringElement.classList.remove('btn-outline', 'btn-secondary');
                            evt.detail.requestConfig.triggeringElement.classList.add('btn-primary');
                        }
                    }
                });
                """),
                
                cls="mt-8"
            )
        
        return Div(
            H2("Analysis Results", cls="text-2xl font-bold text-white mb-4"),
            
            # Results layout - side by side for larger screens
            Div(
                # Left side - image & analysis text
                Div(
                    # Display the original image
                    Div(
                        H3("Uploaded Image", cls="text-xl font-semibold text-white mb-2"),
                        Img(src=f"/temp-image/{os.path.basename(image_path)}", 
                            cls="mx-auto max-h-64 rounded-lg border border-zinc-700"),
                        cls="mb-6"
                    ),
                    
                    # Display Qwen's response
                    Div(
                        H3("AI Analysis", cls="text-xl font-semibold text-white mb-2"),
                        Div(
                            qwen_response,
                            cls="bg-zinc-700 rounded-md p-4 text-white"
                        ),
                        cls="mb-6"
                    ),
                    
                    cls="lg:w-1/2 lg:pr-4 mb-6 lg:mb-0"
                ),
                
                # Right side - token maps (if available)
                Div(
                    token_tabs if token_tabs else "",
                    cls="lg:w-1/2 lg:pl-4"
                ),
                
                cls="flex flex-col lg:flex-row w-full"
            ),
            
            # Button to start over
            Div(
                Button(
                    "Analyze Another Image",
                    hx_get="/",
                    hx_target="#main-content",
                    cls="btn btn-secondary w-full max-w-xs"
                ),
                cls="mt-8 text-center w-full"
            ),
            
            id="analysis-results",
            cls="w-full max-w-2xl bg-zinc-800 rounded-md p-6 fade-in"
        )

    # Define image processing functions
    async def process_with_qwen(image, query, analysis_id):
        """Process the image with Qwen LLM"""
        logging.info(f"Processing image {analysis_id} with Qwen using query: {query}")
        
        try:
            # Build the prompt for Qwen
            prompt = f"""You are an expert biologist analyzing visual specimens.

Query: {query}

Provide a detailed, scientifically accurate response with proper taxonomic terminology.
Keep your answer under 5 sentences but be thorough and precise.
"""
            
            # Use vLLM to get Qwen's response
            vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
            payload = {
                "prompt": prompt,
                "max_tokens": 500,
                "stream": False  # Non-streaming for easier processing
            }
            
            logging.info(f"Sending request to Qwen LLM: {vllm_url}")
            
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(vllm_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        qwen_response = result.get("choices", [{}])[0].get("text", "")
                        logging.info(f"Qwen response received for {analysis_id}")
                        return qwen_response
                    else:
                        error_text = await response.text()
                        logging.error(f"Error from Qwen: {error_text}")
                        return f"Error processing image: {error_text}"
        
        except Exception as e:
            logging.error(f"Exception in process_with_qwen: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    async def generate_similarity_maps(image, query, analysis_id):
        """Generate token similarity maps using ColQwen"""
        logging.info(f"Generating similarity maps for {analysis_id} with query: {query}")
        
        try:
            # Ensure ColQwen model is loaded
            if not ensure_colqwen_model_loaded():
                logging.error("Failed to load ColQwen model")
                return None
                
            # Process the query and image with ColQwen
            processed_image = colpali_processor.process_images([image]).to(colpali_model.device)
            processed_query = colpali_processor.process_queries([query]).to(colpali_model.device)
            
            logging.info("Processing image and query with ColQwen")
            
            # Forward pass to get embeddings
            with torch.no_grad():
                image_embedding = colpali_model(**processed_image)
                query_embedding = colpali_model(**processed_query)
            
            # Get the number of image patches
            n_patches = colpali_processor.get_n_patches(
                image_size=image.size,
                patch_size=colpali_model.patch_size,
                spatial_merge_size=getattr(colpali_model, 'spatial_merge_size', None)
            )
            
            logging.info(f"Image patches: {n_patches}")
            
            # Get the image mask
            image_mask = colpali_processor.get_image_mask(processed_image)
            
            # Generate the similarity maps
            similarity_maps = get_similarity_maps_from_embeddings(
                image_embeddings=image_embedding,
                query_embeddings=query_embedding,
                n_patches=n_patches,
                image_mask=image_mask
            )[0]
            
            # Get tokens for reference
            query_content = colpali_processor.decode(processed_query.input_ids[0]).replace(
                colpali_processor.tokenizer.pad_token, "")
                
            # Handle query augmentation token if it exists
            query_augmentation_token = getattr(colpali_processor, 'query_augmentation_token', None)
            if query_augmentation_token is not None:
                query_content = query_content.replace(query_augmentation_token, "").strip()
                
            # Tokenize the query
            query_tokens = colpali_processor.tokenizer.tokenize(query_content)
            
            logging.info(f"Generated {len(query_tokens)} tokens: {query_tokens[:10]}")
            
            # Filter and save token maps
            token_maps = []
            for idx, token in enumerate(query_tokens[:10]):  # Process up to 10 tokens
                if idx >= similarity_maps.shape[0]:
                    break
                    
                # Get meaningful tokens - skip very short or empty ones
                token_text = str(token).strip()
                if len(token_text) <= 1 or token_text in ["", ".", ",", "?", "!"]:
                    continue
                    
                # Generate the visualization
                try:
                    fig, ax = plot_similarity_map(
                        image=image,
                        similarity_map=similarity_maps[idx],
                        figsize=(10, 10),
                        show_colorbar=True
                    )
                    
                    # Add a title with the token
                    max_sim = similarity_maps[idx].max().item()
                    ax.set_title(f"Token: '{token}' (MaxSim: {max_sim:.2f})")
                    
                    # Save to a temporary file
                    map_path = os.path.join(TEMP_UPLOAD_DIR, f"{analysis_id}_token_{idx}.png")
                    fig.savefig(map_path, dpi=120)
                    plt.close(fig)
                    
                    # Only use first several meaningful tokens to avoid cluttering the UI
                    if len(token_maps) < 5:
                        token_maps.append({"token": token, "path": map_path, "index": idx})
                except Exception as plot_error:
                    logging.error(f"Error plotting similarity map for token {token}: {plot_error}")
            
            logging.info(f"Generated {len(token_maps)} token maps for {analysis_id}")
            return token_maps
            
        except Exception as e:
            logging.error(f"Exception in generate_similarity_maps: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    # Main route for homepage
    @rt("/")
    def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        logging.info(f"New session started: {session['session_id']}")
        
        return (
            Title("Image Analysis with AI"),
            Main(
                # Loading indicator with better visibility
                Div(
                    Div(cls="loading loading-spinner loading-lg text-primary"),
                    Div("Processing your image...", cls="text-white mt-4 text-lg"),
                    id="loading-indicator",
                    cls="htmx-indicator fixed top-0 left-0 w-full h-full bg-black bg-opacity-80 flex flex-col items-center justify-center z-50"
                ),
                
                # Page header 
                H1("Image Analysis with AI", cls="text-3xl font-bold mb-4 text-white"),
                
                # Header info
                Div(
                    P("Upload an insect or species image to analyze it with AI", 
                      cls="text-white text-center mb-8"),
                    cls="w-full max-w-2xl"
                ),
                
                # Main content area 
                Div(
                    upload_form(),
                    id="main-content",
                    cls="w-full max-w-2xl"
                ),
                
                # Results area - will be populated by process-image
                Div(id="analysis-results", cls="w-full max-w-2xl mt-8"),
                
                # Global script to initialize UI behaviors
                Script("""
                // Set up click tracking for debugging
                document.addEventListener('click', function(e) {
                    console.log('Click detected on:', e.target);
                });
                
                // Make sure form submits properly
                document.addEventListener('DOMContentLoaded', function() {
                    const form = document.getElementById('upload-form');
                    if (form) {
                        // Add visual feedback when button is clicked
                        const button = document.getElementById('analyze-button');
                        if (button) {
                            button.addEventListener('click', function() {
                                this.classList.add('opacity-70');
                                setTimeout(() => {
                                    this.classList.remove('opacity-70');
                                }, 100);
                            });
                        }
                        
                        // Show loading state when form is submitted
                        form.addEventListener('submit', function() {
                            const btn = document.getElementById('analyze-button');
                            if (btn) {
                                btn.innerHTML = '<span class="loading loading-spinner loading-sm"></span> Processing...';
                                btn.disabled = true;
                            }
                            
                            // Show loading indicator
                            document.getElementById('loading-indicator').style.display = 'flex';
                        });
                    }
                });
                
                // Handle tabs for token maps
                document.addEventListener('htmx:afterSwap', function(evt) {
                    if (evt.detail.target.id === 'analysis-results') {
                        // Auto-select first token tab if available
                        setTimeout(function() {
                            const firstTab = document.getElementById('token-tab-0');
                            if (firstTab) firstTab.click();
                        }, 300);
                    }
                });
                """),
                
                cls="flex flex-col items-center min-h-screen bg-black p-4",
            )
        )

    # Process uploaded image
    @rt("/process-image", methods=["POST"])
    async def process_image(request: Request):
        form = await request.form()
        
        # Directly access the form data for more reliable handling
        image_file = form.get("image")
        analysis_type = form.get("analysis_type", "classify_insect")
        
        # Generate unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        if not image_file:
            return Div("No image file uploaded", cls="text-red-500")
        
        logging.info(f"Image uploaded: {getattr(image_file, 'filename', 'unknown')}, Analysis type: {analysis_type}, ID: {analysis_id}")
        
        # Create a temporary path to save the image
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        filename = getattr(image_file, 'filename', f"image_{analysis_id}.jpg")
        image_path = os.path.join(TEMP_UPLOAD_DIR, f"{analysis_id}_{filename}")
        
        # Save the uploaded image
        logging.info(f"Saving image to {image_path}")
        try:
            # Read the content of the uploaded file
            content = await image_file.read()
            with open(image_path, "wb") as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            return Div(f"Error saving image: {str(e)}", cls="text-red-500")
        
        # Open the image for processing
        try:
            image = Image.open(image_path)
            logging.info(f"Image opened: {image.size}, {image.mode}")
        except Exception as e:
            logging.error(f"Error opening image: {e}")
            return Div(f"Error opening image: {str(e)}", cls="text-red-500")
        
        # Get the appropriate query based on analysis type
        query = get_query_for_analysis(analysis_type)
        logging.info(f"Using query: {query}")
        
        # Save analysis to database
        try:
            analysis = ImageAnalysis(
                analysis_id=analysis_id,
                image_path=image_path,
                analysis_type=analysis_type,
                query=query
            )
            sqlalchemy_session.add(analysis)
            sqlalchemy_session.commit()
            logging.info(f"Analysis record created in database: {analysis_id}")
        except Exception as db_error:
            logging.error(f"Database error: {db_error}")
            sqlalchemy_session.rollback()
        
        # Process with Qwen
        qwen_response = await process_with_qwen(image, query, analysis_id)
        
        # Update database with response
        try:
            analysis = sqlalchemy_session.query(ImageAnalysis).filter_by(analysis_id=analysis_id).first()
            if analysis:
                analysis.response = qwen_response
                sqlalchemy_session.commit()
                logging.info(f"Updated analysis record with response: {analysis_id}")
        except Exception as db_error:
            logging.error(f"Database error updating response: {db_error}")
            sqlalchemy_session.rollback()
        
        # Generate similarity maps with ColQwen
        token_maps = await generate_similarity_maps(image, query, analysis_id)
        
        # Return the analysis results UI
        logging.info(f"Returning analysis results UI for {analysis_id}")
        return analysis_results_ui(image_path, qwen_response, token_maps, analysis_id)

    @rt("/get-token-map/{filename}")
    async def get_token_map(filename: str):
        """Return the token map image HTML for display in the UI"""
        # Get the file path
        image_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        
        if not os.path.exists(image_path):
            return Div("Token map not found", cls="text-red-500 p-4")
            
        # Return the HTML for the token map display
        return Div(
            Img(
                src=f"/temp-image/{filename}",
                cls="mx-auto max-h-96 w-full object-contain"
            ),
            cls="p-2 flex items-center justify-center min-h-[300px]"
        )

    @rt("/temp-image/{filename}")
    async def serve_temp_image(filename: str):
        """Serve temporary images (uploads and similarity maps)"""
        image_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return Response(content=f"Image not found: {filename}", 
                        media_type="text/plain", status_code=404)
        
        return FileResponse(image_path, media_type=f"image/{os.path.splitext(filename)[1][1:]}")

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
