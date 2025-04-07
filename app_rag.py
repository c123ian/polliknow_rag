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
import datetime
from modal import Secret
from fastlite import Database
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, Response
from fastapi import FastAPI, Request, Response as FastAPIResponse
from fastapi.responses import FileResponse as FastAPIFileResponse
from starlette.responses import Response as StarletteResponse
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import json
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from rerankers import Reranker
from io import BytesIO

# Constants
MISTRAL_MODELS_DIR = "/Mistral"  
COLQWEN_MODELS_DIR = "/ColQwen"  # volume for ColQwen2 models
DATA_DIR = "/bee_pdf"
TEMP_UPLOAD_DIR = "/bee_pdf/temp_uploads"
HEATMAP_DIR = "/bee_pdf/heatmaps"  # Directory for storing heatmaps
DEFAULT_MISTRAL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_COLQWEN_NAME = "vidore/colqwen2-v1.0"
USERNAME = "c123ian"
APP_NAME = "polliknow-rag"
DATABASE_DIR = "/db_rag_advan"
PDF_IMAGES_DIR = "/bee_pdf/pdf_images"

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
os.makedirs(HEATMAP_DIR, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
# Check if context_source column exists, add it if not
cursor.execute("PRAGMA table_info(image_analyses)")
columns = [column[1] for column in cursor.fetchall()]
if 'image_analyses' not in columns:
    # Create the table if it doesn't exist
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

# Don't try to add context_source if it's already there
if 'context_source' not in columns:
    try:
        cursor.execute('ALTER TABLE image_analyses ADD COLUMN context_source TEXT')
        conn.commit()
        logging.info("Added context_source column to image_analyses table")
    except Exception as e:
        logging.error(f"Error adding context_source column: {e}")
conn.close()

# Set up volumes
try:
    mistral_volume = modal.Volume.lookup("Mistral", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download Mistral models first with the appropriate script")

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
        "vllm==0.8.1",              # Updated to minimum required version
        "mistral_common>=1.5.4",    # Required for Mistral 3.1
        "python-fasthtml==0.4.3",
        "aiohttp",
        "faiss-cpu",
        "sentence-transformers",
        "pandas",
        "numpy",
        "huggingface_hub",
        "transformers>=4.48.3",
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
# Define global engine variable
engine = None

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),  # Changed to A100 to match actual needs
    container_idle_timeout=10 * 60,  # Increased idle timeout
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=20,  # Reduced from 100 for better stability
    volumes={
        MISTRAL_MODELS_DIR: mistral_volume,
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
    import datetime
    from fastapi.responses import StreamingResponse, JSONResponse
    from typing import Optional

    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels
    from vllm.entrypoints.logger import RequestLogger
    from vllm.sampling_params import SamplingParams

    global engine
    MODELS_DIR = "/Mistral"

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {DEFAULT_MISTRAL_NAME} server",
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

    # Add these parameters to your AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        dtype="float16",
        # ADD THESE NEW PARAMETERS:
        limit_mm_per_prompt={"image": 4},  # Allow up to 4 images per prompt
        disable_mm_preprocessor_cache=False  # Enable caching for better performance
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

    models = OpenAIServingModels(engine_client=engine, model_config=model_config, base_model_paths={DEFAULT_MISTRAL_NAME: model_path})
    request_logger = RequestLogger(max_log_len=256)

    openai_serving_chat = OpenAIServingChat(
        engine_client=engine,
        model_config=model_config,
        models=models,
        response_role="assistant",
        request_logger=request_logger,
        chat_template=None,  
        #chat_template="mistral",  # Use Mistral's template instead of None
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

            sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

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
                # Non-streaming response - FIXED to properly collect from generator
                final_output = None
                try:
                    # Collect the final result from the generator
                    async for result in engine.generate(prompt, sampling_params, request_id):
                        final_output = result
                    
                    # Now check if we got a valid result
                    if final_output and len(final_output.outputs) > 0:
                        response_text = final_output.outputs[0].text
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
                except Exception as gen_error:
                    logging.error(f"Error in non-streaming generation: {str(gen_error)}")
                    import traceback
                    traceback.print_exc()
                    return JSONResponse(content={"error": f"Generation error: {str(gen_error)}"}, status_code=500)
                
        except Exception as e:
            logging.error(f"Error in completion_generator: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.post("/v1/chat/completions")
    async def chat_completions(request: fastapi.Request) -> JSONResponse:
        """Handle multimodal chat completions"""
        try:
            body = await request.json()
            messages = body.get("messages", [])
            model = body.get("model", DEFAULT_MISTRAL_NAME)
            max_tokens = body.get("max_tokens", 500)
            temperature = body.get("temperature", 0.7)
            request_id = str(uuid.uuid4())
            
            # Format messages into a prompt manually for Mistral
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle different message formats
                if isinstance(content, list):
                    # Multimodal message
                    text_parts = []
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    formatted_text = " ".join(text_parts)
                    formatted_messages.append(f"{role.capitalize()}: {formatted_text}")
                else:
                    # Text-only message - FIXED INDENTATION HERE
                    formatted_messages.append(f"{role.capitalize()}: {content}")
            
            # Join messages with newlines
            prompt = "\n".join(formatted_messages) + "\nAssistant:"
            
            # Create sampling params
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                #stop=["User:", "System:", "\n\n"],
            )
            
            # Generate response - collect all outputs from the generator
            final_output = None
            async for output in engine.generate(prompt, sampling_params, request_id):
                final_output = output
            
            if final_output and len(final_output.outputs) > 0:
                response_text = final_output.outputs[0].text
                
                # Create a properly formatted response
                return JSONResponse(content={
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(datetime.datetime.now().timestamp()),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }
                    ]
                })
            else:
                return JSONResponse(status_code=500, content={"error": "No output generated"})
        except Exception as e:
            logging.error(f"Error in chat_completions: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})
                

    # Add a simple health check endpoint
    @web_app.get("/health")
    async def health_check():
        """Check if the server is running and the model is loaded"""
        return JSONResponse(content={
            "status": "healthy", 
            "model": DEFAULT_MISTRAL_NAME,
            "supports_multimodal": True,
            "max_tokens": 32768
        })

    return web_app

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
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
    from PIL import Image
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
    import nltk
    from nltk.tokenize import word_tokenize
    import numpy as np
    from rank_bm25 import BM25Okapi
    from pdf2image import convert_from_path
    from vllm.sampling_params import SamplingParams
    
    # Setup NLTK
    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    
    # Global variables
    colpali_model = None
    colpali_processor = None
    colpali_embeddings = None
    df = None
    page_images = {}
    bm25_index = None
    tokenized_docs = None
    
    # Load ColPali model and data functions
    def ensure_colpali_model_loaded():
        """Ensure the ColQwen2 model is properly loaded from volume if possible"""
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
                # First check if files exist directly in the volume root
                model_files_in_root = False
                volume_root = COLQWEN_MODELS_DIR
                required_files = ["tokenizer.json", "adapter_model.safetensors", "special_tokens_map.json"]
                
                if os.path.exists(volume_root):
                    root_files = os.listdir(volume_root)
                    if all(file in root_files for file in required_files):
                        model_files_in_root = True
                        logging.info(f"Found model files directly in volume root: {volume_root}")
                
                if model_files_in_root:
                    # Use volume root directly as model path
                    logging.info(f"Using local model from volume root: {volume_root}")
                    colpali_model = ColQwen2.from_pretrained(
                        volume_root,
                        torch_dtype=torch.bfloat16,
                        device_map="cuda" if torch.cuda.is_available() else "cpu"
                    ).eval()
                    colpali_processor = ColQwen2Processor.from_pretrained(volume_root)
                else:
                    # Check for subdirectory structure
                    model_path = os.path.join(COLQWEN_MODELS_DIR, os.path.basename(DEFAULT_COLQWEN_NAME))
                    if os.path.exists(model_path) and os.path.isdir(model_path):
                        logging.info(f"Using local model from volume subdirectory: {model_path}")
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

    def load_data():
        """Load all data needed for document retrieval"""
        global colpali_embeddings, df, page_images, bm25_index, tokenized_docs
        
        # Path definitions
        COLPALI_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "colpali_embeddings.pkl")
        DATA_PICKLE_PATH = os.path.join(DATA_DIR, "data.pkl")
        PDF_PAGE_IMAGES_PATH = os.path.join(DATA_DIR, "pdf_page_image_paths.pkl")
        BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
        TOKENIZED_PARAGRAPHS_PATH = os.path.join(DATA_DIR, "tokenized_paragraphs.pkl")
        
        # Load data frame with metadata
        if os.path.exists(DATA_PICKLE_PATH):
            try:
                df = pd.read_pickle(DATA_PICKLE_PATH)
                logging.info(f"Loaded DataFrame with {len(df)} documents")
            except Exception as e:
                logging.error(f"Error loading DataFrame: {e}")
                df = pd.DataFrame(columns=["filename", "page", "paragraph_size", "text", "image_key", "full_path"])
        else:
            logging.error(f"DataFrame not found at {DATA_PICKLE_PATH}")
            df = pd.DataFrame(columns=["filename", "page", "paragraph_size", "text", "image_key", "full_path"])
        
        # Load image paths
        if os.path.exists(PDF_PAGE_IMAGES_PATH):
            try:
                with open(PDF_PAGE_IMAGES_PATH, "rb") as f:
                    page_images = pickle.load(f)
                logging.info(f"Loaded {len(page_images)} image paths")
            except Exception as e:
                logging.error(f"Error loading image paths: {e}")
                page_images = {}
        else:
            logging.error(f"Image paths file not found at {PDF_PAGE_IMAGES_PATH}")
            page_images = {}
        
        # Load ColPali embeddings
        if os.path.exists(COLPALI_EMBEDDINGS_PATH):
            try:
                with open(COLPALI_EMBEDDINGS_PATH, "rb") as f:
                    colpali_embeddings = pickle.load(f)
                logging.info(f"Loaded {len(colpali_embeddings)} ColPali embeddings")
            except Exception as e:
                logging.error(f"Error loading ColPali embeddings: {e}")
                colpali_embeddings = None
        else:
            logging.error(f"ColPali embeddings not found at {COLPALI_EMBEDDINGS_PATH}")
            colpali_embeddings = None
        
        # Load BM25 index
        try:
            if os.path.exists(BM25_INDEX_PATH) and os.path.exists(TOKENIZED_PARAGRAPHS_PATH):
                with open(BM25_INDEX_PATH, "rb") as f:
                    bm25_index = pickle.load(f)
                with open(TOKENIZED_PARAGRAPHS_PATH, "rb") as f:
                    tokenized_docs = pickle.load(f)
                logging.info("Loaded BM25 index successfully")
            else:
                logging.warning("BM25 index not found, will create if needed")
        except Exception as e:
            logging.error(f"Error loading BM25 index: {e}")
            bm25_index = None
            tokenized_docs = None

    # Load models and data at startup
    ensure_colpali_model_loaded()
    load_data()

    # Make temporary directories
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    # Setup database connection
    db_engine = create_engine(f'sqlite:///{db_path}')
    Base = declarative_base()
    Session = sessionmaker(bind=db_engine)
    sqlalchemy_session = Session()

    # Define the database model - matching the actual table schema
    class ImageAnalysis(Base):
        __tablename__ = 'image_analyses'
        analysis_id = Column(String, primary_key=True)
        image_path = Column(String, nullable=False)
        analysis_type = Column(String, nullable=False)
        query = Column(String, nullable=False)
        response = Column(String)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)
        # Only include context_source if we verified it exists
        context_source = Column(String, nullable=True)

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
                
                /* Custom styles for token maps */
                .token-tab.active {
                    background-color: #4CAF50;
                    color: white;
                }
                
                /* Custom animations */
                .fade-in {
                    animation: fadeIn 0.5s;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                /* Custom styling for token map display */
                .token-map {
                    max-height: 350px;
                    object-fit: contain;
                }
                
                .context-container {
                    background-color: #2a2a2a;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 12px;
                }
                
                .context-header {
                    color: #aaa;
                    font-size: 14px;
                    margin-bottom: 8px;
                }
                
                /* Carousel styles */
                .carousel-item {
                    scroll-snap-align: start;
                }
                
                .carousel {
                    scroll-behavior: smooth;
                    scroll-snap-type: x mandatory;
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

    # Helper function to extract the one-word classification from LLM response
    def extract_classification(response_text):
        """Extract the one-word classification from the LLM response"""
        # Try to extract the classification after "Answer:" if present
        if "Answer:" in response_text:
            parts = response_text.split("Answer:")
            if len(parts) > 1:
                answer = parts[1].strip().lower()
                # Take only the first word
                classification = answer.split()[0] if answer.split() else "unknown"
                return classification
        
        # If no "Answer:" pattern, look for specific categories in the response
        categories = ["bumblebee", "honeybee", "wasp", "solitary bee", "hoverfly", "flies", "butterfly", "moth", "other"]
        
        # Check for each category in the text
        response_lower = response_text.lower()
        for category in categories:
            if category in response_lower:
                if category == "flies":
                    return "other flies"
                elif category in ["butterfly", "moth"]:
                    return "butterfly & moths"
                return category
        
        # Default if no category found
        return "other insect"

    # Helper function to get badge color based on classification
    def get_badge_color(classification):
        """Return the appropriate badge color for a classification"""
        colors = {
            "bumblebee": "badge-primary",      # Blue
            "honeybee": "badge-warning",       # Yellow
            "wasp": "badge-accent",            # Dark yellow/orange
            "hoverfly": "badge-success",       # Green
            "other flies": "badge-info",       # Light blue
            "butterfly & moths": "badge-secondary", # Purple
            "other insect": "badge-neutral",   # Gray
            "error": "badge-error"             # Red
        }
        
        # Normalize the classification
        classification = classification.lower()
        
        # Check for partial matches - simplified for accurate matching
        for key, color in colors.items():
            if key == classification:
                return color
        
        # Default to neutral if no match
        return "badge-neutral"
        
    # Helper functions for saving to database
    async def save_to_database(analysis_id, image_path, analysis_type, query, response, top_sources):
        """Save analysis results to the database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert basic info
            cursor.execute(
                "INSERT INTO image_analyses (analysis_id, image_path, analysis_type, query, response) VALUES (?, ?, ?, ?, ?)",
                (analysis_id, image_path, analysis_type, query, response)
            )
            
            # Add context source if available
            if top_sources:
                top_source = top_sources[0]
                context_source = f"{top_source['filename']} (page {top_source['page']})"
                
                # Check if the column exists
                cursor.execute("PRAGMA table_info(image_analyses)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'context_source' in columns:
                    cursor.execute(
                        "UPDATE image_analyses SET context_source = ? WHERE analysis_id = ?",
                        (context_source, analysis_id)
                    )
            
            conn.commit()
            conn.close()
            logging.info(f"Saved analysis {analysis_id} to database")
            return True
        except Exception as db_error:
            logging.error(f"Database error: {db_error}")
            import traceback
            traceback.print_exc()
            return False

    # Retrieve relevant documents
    async def retrieve_relevant_documents(query, top_k=5):
        """Retrieve most relevant documents using ColPali embeddings and BM25"""
        global colpali_model, colpali_processor, colpali_embeddings, df, bm25_index, tokenized_docs
        
        # Ensure models and data are loaded
        ensure_colpali_model_loaded()
        
        if colpali_embeddings is None or df is None or len(df) == 0:
            logging.error("No documents or embeddings available for retrieval")
            return [], []
            
        retrieved_paragraphs = []
        top_sources_data = []
        
        # ColPali retrieval (vector search)
        try:
            # Process query with ColPali
            processed_query = colpali_processor.process_queries([query]).to(colpali_model.device)
            with torch.no_grad():
                query_embeddings = colpali_model(**processed_query)
            
            # Calculate similarities with all pages
            similarities = []
            for idx, page_emb in enumerate(colpali_embeddings):
                # Convert page embedding to tensor with matching dtype
                page_tensor = torch.tensor(page_emb, device=colpali_model.device, dtype=query_embeddings.dtype)
                
                # Score using ColPali's scoring method
                score = float(colpali_processor.score_multi_vector(
                    query_embeddings,
                    page_tensor.unsqueeze(0)  # Add batch dimension
                )[0])
                
                similarities.append((idx, score))
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)
            vector_top_indices = [idx for idx, _ in similarities[:top_k]]
            
            # Try BM25 keyword search if available
            keyword_top_indices = []
            bm25_scores = None
            if bm25_index is not None and tokenized_docs is not None:
                try:
                    # Tokenize query and get BM25 scores
                    tokenized_query = word_tokenize(query.lower())
                    bm25_scores = bm25_index.get_scores(tokenized_query)
                    keyword_top_indices = np.argsort(bm25_scores)[-top_k:][::-1].tolist()
                except Exception as e:
                    logging.error(f"Error in BM25 scoring: {e}")
            
            # Combine results (hybrid retrieval)
            all_indices = list(set(vector_top_indices + keyword_top_indices))
            
            # Get data for reranking
            docs_for_reranking = []
            doc_indices = []
            
            for idx in all_indices:
                if idx < len(df):
                    # Get document info
                    filename = df.iloc[idx]['filename']
                    page_num = df.iloc[idx]['page']
                    image_key = df.iloc[idx]['image_key']
                    text = df.iloc[idx]['text']
                    
                    # Get vector score
                    vector_score = 0.0
                    for v_idx, score in similarities:
                        if v_idx == idx:
                            vector_score = score
                            break
                    
                    # Get keyword score (if available)
                    keyword_score = 0.0
                    if bm25_scores is not None and len(bm25_scores) > idx:
                        keyword_score = float(bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0)
                    
                    # Combine scores (weighted)
                    alpha = 0.7  # Weight for vector search
                    combined_score = alpha * vector_score + (1 - alpha) * keyword_score
                    
                    # Store for reranking
                    docs_for_reranking.append(text)
                    doc_indices.append(idx)
                    
                    # Add to results
                    retrieved_paragraphs.append(text)
                    top_sources_data.append({
                        'filename': filename,
                        'page': page_num,
                        'score': combined_score,
                        'vector_score': vector_score,
                        'keyword_score': keyword_score,
                        'image_key': image_key,
                        'idx': idx
                    })
            
            # Rerank results if we have documents
            if docs_for_reranking:
                try:
                    # Use a cross-encoder reranker
                    from rerankers import Reranker
                    ranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2', model_type="cross-encoder", verbose=0)
                    ranked_results = ranker.rank(query=query, docs=docs_for_reranking)
                    top_ranked = ranked_results.top_k(min(3, len(docs_for_reranking)))
                    
                    # Get the final top documents after reranking
                    final_retrieved_paragraphs = []
                    final_top_sources = []
                    
                    for ranked_doc in top_ranked:
                        ranked_idx = docs_for_reranking.index(ranked_doc.text)
                        doc_idx = doc_indices[ranked_idx]
                        source_info = next((s for s in top_sources_data if s['idx'] == doc_idx), None)
                        if source_info:
                            source_info['reranker_score'] = ranked_doc.score
                            final_top_sources.append(source_info)
                            final_retrieved_paragraphs.append(ranked_doc.text)
                    
                    return final_retrieved_paragraphs, final_top_sources
                    
                except Exception as e:
                    logging.error(f"Error in reranking: {e}")
            
            # If reranking fails, sort by combined score
            sorted_indices = sorted(range(len(top_sources_data)), 
                                   key=lambda i: top_sources_data[i]['score'], 
                                   reverse=True)
            sorted_paragraphs = [retrieved_paragraphs[i] for i in sorted_indices[:3]]
            sorted_sources = [top_sources_data[i] for i in sorted_indices[:3]]
            
            return sorted_paragraphs, sorted_sources
            
        except Exception as e:
            logging.error(f"Error in document retrieval: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    # Modified generate_similarity_maps function to use similarity score selection
    async def generate_similarity_maps(query, image_key):
        """Generate token similarity maps for a retrieved document based on similarity scores"""
        global colpali_model, colpali_processor, page_images
        
        if not image_key or image_key not in page_images:
            logging.error(f"Invalid image key: {image_key}")
            return []
            
        try:
            # Get image path and load image
            image_path = page_images[image_key]
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                
                # Try to find the image by reconstructing path patterns
                parts = image_key.split('_')
                if len(parts) >= 2:
                    filename = '_'.join(parts[:-1])
                    page_num = parts[-1]
                    
                    # Check different potential locations
                    potential_paths = [
                        os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                        os.path.join(PDF_IMAGES_DIR, filename, f"page_{page_num}.png"),
                        os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
                    ]
                    
                    for potential_path in potential_paths:
                        if os.path.exists(potential_path):
                            logging.info(f"Found image at: {potential_path}")
                            image_path = potential_path
                            break
                
                if not os.path.exists(image_path):
                    return []
                    
            # Load the image
            image = Image.open(image_path)
            
            # Process query and image with ColPali
            processed_query = colpali_processor.process_queries([query]).to(colpali_model.device)
            batch_images = colpali_processor.process_images([image]).to(colpali_model.device)
            
            # Forward passes to get embeddings
            with torch.no_grad():
                query_embeddings = colpali_model(**processed_query)
                image_embeddings = colpali_model(**batch_images)
            
            # Get the number of image patches
            n_patches = colpali_processor.get_n_patches(
                image_size=image.size,
                patch_size=colpali_model.patch_size,
                spatial_merge_size=getattr(colpali_model, 'spatial_merge_size', None)
            )
            
            # Get image mask
            image_mask = colpali_processor.get_image_mask(batch_images)
            
            # Generate similarity maps
            batched_similarity_maps = get_similarity_maps_from_embeddings(
                image_embeddings=image_embeddings,
                query_embeddings=query_embeddings,
                n_patches=n_patches,
                image_mask=image_mask
            )
            
            # Get the similarity map for this image
            similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)
            
            # Get tokens for the query
            query_tokens = colpali_processor.tokenizer.tokenize(query)
            
            # Filter to meaningful tokens - approach from older version
            token_sims = []
            stopwords = set(["<bos>", "<eos>", "<pad>", "a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "to", "from"])
            
            for token_idx, token in enumerate(query_tokens):
                if token_idx >= similarity_maps.shape[0]:
                    continue
                    
                # Skip stopwords and short tokens
                if token in stopwords or len(token) <= 1:
                    continue
                    
                token_clean = token.replace("Ġ", "").replace("▁", "")
                if token_clean and len(token_clean) > 1:
                    max_sim = similarity_maps[token_idx].max().item()
                    token_sims.append((token_idx, token, max_sim))
            
            # Sort by similarity score and take top tokens
            token_sims.sort(key=lambda x: x[2], reverse=True)
            top_tokens = token_sims[:6]  # Get top 6 tokens
            
            # Create directory if it doesn't exist
            os.makedirs(HEATMAP_DIR, exist_ok=True)
            
            # Generate and save heatmaps
            image_heatmaps = []
            for token_idx, token, score in top_tokens:
                # Skip if score is very low
                if score < 0.1:
                    continue
                    
                # Generate heatmap
                fig, ax = plot_similarity_map(
                    image=image,
                    similarity_map=similarity_maps[token_idx],
                    figsize=(8, 8),
                    show_colorbar=False,
                )
                
                # Clean token for display
                token_display = token.replace("Ġ", "").replace("▁", "")
                ax.set_title(f"Token: '{token_display}', Score: {score:.2f}", fontsize=12)
                
                # Save heatmap
                heatmap_filename = f"{image_key}_token_{token_idx}.png"
                heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
                fig.savefig(heatmap_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                
                # Add to list
                image_heatmaps.append({
                    "token": token_display,
                    "score": score,
                    "path": heatmap_filename,
                    "token_idx": token_idx
                })
            
            logging.info(f"Generated {len(image_heatmaps)} token heatmaps")
            return image_heatmaps
            
        except Exception as e:
            logging.error(f"Error generating similarity maps: {e}")
            import traceback
            traceback.print_exc()
            return []

    # Updated process_with_mistral function with insect classification system prompt
    async def process_with_mistral(image, query, context_text="", analysis_id=""):
        """Process the image with Mistral LLM including context from retrieved documents"""
        logging.info(f"Processing image {analysis_id} with Mistral using query: {query}")
        
        try:
            # Convert image to base64
            img_buffer = BytesIO()
            image.save(img_buffer, format=image.format or "JPEG")
            img_data = img_buffer.getvalue()
            base64_img = base64.b64encode(img_data).decode("utf-8")
            
            # Create system prompt that incorporates the query but maintains classification focus
            system_prompt = (
                "You are an expert biologist. Classify the insect in the provided image into "
                "ONE of these categories (bumblebee, honeybee, wasp, solitary bee, hoverfly, other flies, "
                "butterfly & moths, other insect). Focus primarily on the visual characteristics "
                f"in the image itself. {query} "
                "Use any provided context as secondary information only. After analysis, provide the answer "
                "as a single word, for example 'Answer: honeybee'."
            )
            
            # Create multimodal message format with updated system prompt
            payload = {
            "model": DEFAULT_MISTRAL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Classify this insect.{' Additional reference information:' + context_text if context_text else ''}"},
                    {"type": "image", "image": {"data": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
                "max_tokens": 2000,
                "temperature": 0.7,
                "stream": False
            }
            
            # Send to vLLM endpoint
            multimodal_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/chat/completions"
            
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(multimodal_url, json=payload, timeout=320) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Add this line to log the full JSON response
                        logging.info(f"Raw API response for analysis {analysis_id}:\n{json.dumps(result, indent=2)}")
                        
                        if "choices" in result and len(result["choices"]) > 0:
                            model_response = result["choices"][0]["message"]["content"]
                            
                            # Log the full response content
                            logging.info(f"Full LLM response for analysis {analysis_id}:\n{'='*50}\n{model_response}\n{'='*50}")
                            
                            return model_response.strip()
                        else:
                            return "Error: No response choices returned from model"
                    else:
                        error_text = await response.text()
                        return f"Error processing image: {error_text}"
        
        except Exception as e:
            logging.error(f"Exception in process_with_mistral: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    # Define UI components
    def batch_upload_form():
        """Render form for uploading multiple images for batch insect classification"""
        
        return Form(
            Div(
                H2("Batch Insect Classification", cls="text-xl font-semibold text-white mb-4"),
                
                P("Upload multiple insect images (up to 10) for AI classification.", 
                  cls="text-zinc-300 text-center mb-6"),
                
                # Image upload card
                Div(
                    Div(
                        Label("Upload Insect Images:", cls="text-white font-medium mb-2"),
                        Input(
                            type="file",
                            name="image_files",
                            accept=".jpg,.jpeg,.png",
                            required=True,
                            multiple=True,
                            cls="file-input file-input-bordered file-input-warning w-full",
                            onchange="handleMultipleFiles(this)"
                        ),
                        P(id="file-count", cls="text-sm text-zinc-400 mt-2"),
                        # Hidden container for individual file inputs
                        Div(id="file-inputs-container", cls="hidden"),
                        cls="grid place-items-center p-4"
                    ),
                    cls="card bg-zinc-800 border border-zinc-700 rounded-box w-full mb-4"
                ),
                
                # Context sharing option
                Div(
                    Label(
                        Input(type="checkbox", name="share_context", value="true", cls="checkbox checkbox-warning mr-2"),
                        "Share context across all images (faster)",
                        cls="flex items-center cursor-pointer text-white"
                    ),
                    P("Uses the same reference document for all insects instead of finding unique matches.",
                      cls="text-zinc-400 text-sm mt-1 ml-6"),
                    cls="mb-6"
                ),
                
                # Process button
                Button(
                    Div(
                        "Classify Insects",
                        cls="flex items-center justify-center"
                    ),
                    id="batch-button",
                    type="submit",
                    cls="btn btn-warning w-full"
                ),
                
                # JavaScript for handling multiple files
                Script("""
                function handleMultipleFiles(input) {
                    const maxFiles = 10;
                    if (input.files.length > maxFiles) {
                        alert(`Please select a maximum of ${maxFiles} files.`);
                        input.value = '';
                        document.getElementById('file-count').textContent = '';
                        return;
                    }
                    
                    // Clear previous file inputs
                    const container = document.getElementById('file-inputs-container');
                    container.innerHTML = '';
                    
                    // Create hidden inputs for each file
                    for (let i = 0; i < input.files.length; i++) {
                        const fileInput = document.createElement('input');
                        fileInput.type = 'file';
                        fileInput.name = `image_${i}`;
                        fileInput.style.display = 'none';
                        container.appendChild(fileInput);
                        
                        // Use FileList API to set the file
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(input.files[i]);
                        fileInput.files = dataTransfer.files;
                    }
                    
                    // Update count display
                    document.getElementById('file-count').textContent = 
                        `${input.files.length} file${input.files.length !== 1 ? 's' : ''} selected`;
                }
                """),
                
                cls="bg-zinc-900 rounded-md p-6 w-full max-w-lg border border-zinc-700"
            ),
            action="/process-batch",
            method="post",
            enctype="multipart/form-data",
            id="batch-upload-form",
            hx_post="/process-batch",
            hx_target="#main-content",
            hx_indicator="#loading-indicator",
        )

    # Updated carousel UI with the new design for insect classification
    def carousel_ui(batch_results):
        """Create a carousel UI to display multiple image analysis results with the new design"""
        
        # Create carousel items
        carousel_items = []
        carousel_indicators = []
        
        for i, result in enumerate(batch_results):
            # Extract data
            analysis_id = result.get("analysis_id", f"result_{i}")
            image_path = result.get("image_path", "")
            response = result.get("response", "No response generated")
            context_paragraphs = result.get("context_paragraphs", [])
            top_sources = result.get("top_sources", [])
            token_maps = result.get("token_maps", {})
            has_error = result.get("error", False)
            
            # Extract the one-word classification
            classification = "error" if has_error else extract_classification(response)
            badge_color = get_badge_color(classification)
            
            # Create carousel item for the image
            carousel_items.append(
                Div(
                    # Image display
                    try_read_image(image_path),
                    id=f"item{i+1}",
                    cls="carousel-item w-full"
                )
            )
            
            # Create indicator buttons
            carousel_indicators.append(
                A(
                    str(i+1),
                    href=f"#item{i+1}",
                    cls=f"btn btn-xs {'' if i > 0 else 'btn-active'}"
                )
            )
        
        # Create the classification badges and context containers below the carousel
        classification_sections = []
        
        for i, result in enumerate(batch_results):
            # Extract data again
            analysis_id = result.get("analysis_id", f"result_{i}")
            response = result.get("response", "No response generated")
            context_paragraphs = result.get("context_paragraphs", [])
            top_sources = result.get("top_sources", [])
            token_maps = result.get("token_maps", {})
            has_error = result.get("error", False)
            
            # Extract the one-word classification
            classification = "error" if has_error else extract_classification(response)
            badge_color = get_badge_color(classification)
            
            # Create collapsible context section
            context_section = Div()
            
            if top_sources and len(top_sources) > 0:
                top_source = top_sources[0]
                image_key = top_source.get('image_key', '')
                
                # Get token maps for this image if available
                current_token_maps = token_maps.get(image_key, [])
                
                # Create a unique modal ID for this result
                modal_id = f"modal_{analysis_id}"
                
                context_section = Div(
                    # Badge and controls section
                    Div(
                        # Classification badge
                        Div(
                            classification,
                            cls=f"badge {badge_color} text-lg p-3 mr-4"
                        ),
                        
                        # Replace toggle with swap component
                        Label(
                            # Hidden checkbox still controls the state
                            Input(type="checkbox"),
                            # Swap-on shows when checked (thumbs down)
                            Div("👎", cls="swap-on"),
                            # Swap-off shows when unchecked (thumbs up)
                            Div("👍", cls="swap-off"),
                            cls="swap swap-flip text-3xl mx-2"
                        ),
                        
                        # Add dialog modal button
                        Button(
                            "view raw output",
                            onclick=f"{modal_id}.showModal()",
                            cls="btn btn-sm btn-outline ml-2"
                        ),
                        
                        # Add dialog modal
                        Dialog(
                            Div(
                                H3("LLM Response", cls="text-lg font-bold"),
                                P(response, cls="py-4 whitespace-pre-wrap text-sm font-mono bg-black p-2 rounded overflow-auto max-h-96"),
                                Div(
                                    Form(
                                        Button("Close", cls="btn"),
                                        method="dialog"
                                    ),
                                    cls="modal-action"
                                ),
                                cls="modal-box"
                            ),
                            id=modal_id,
                            cls="modal"
                        ),
                        
                        cls="flex items-center mb-4"
                    ),
                    
                    # Collapsible context section
                    Div(
                        # Collapse title
                        Div(
                            "View Context",
                            cls="collapse-title font-semibold"
                        ),
                        
                        # Collapse content
                        Div(
                            # Context document
                            (Div(
                                H4("Context Document", cls="text-lg font-semibold text-white mb-2"),
                                try_read_pdf_image(top_source.get('image_key', '')),
                                cls="mb-4"
                            ) if top_source else ""),
                            
                            # Context paragraphs if available
                            (Div(
                                H4("Retrieved Context", cls="text-lg font-semibold text-white mb-2"),
                                P(context_paragraphs[0] if context_paragraphs else "No context available", 
                                cls="text-white text-sm bg-zinc-700 p-3 rounded-md"),
                                cls="mb-4"
                            ) if context_paragraphs else ""),
                            
                            # Token map for "Classify" token
                            (Div(
                                H4("Token Similarity Map", cls="text-lg font-semibold text-white mb-2"),
                                (Img(
                                    src=f"/heatmap-image/{current_token_maps[0]['path']}" if current_token_maps else "",
                                    cls="w-full max-h-80 object-contain rounded-md"
                                ) if current_token_maps else 
                                Div("No token maps available", cls="text-zinc-400 text-center p-4")),
                                cls="mb-4"
                            ) if current_token_maps else ""),
                            
                            cls="collapse-content text-sm"
                        ),
                        
                        tabindex="0",
                        cls="bg-zinc-800 text-white focus:bg-zinc-700 collapse rounded-md",
                    ),
                    
                    id=f"classification-section-{i}",
                    cls=f"mb-6 {'hidden' if i > 0 else ''}"
                )
            else:
                # Simplified section when no context is available
                context_section = Div(
                    # Classification badge only
                    Div(
                        classification,
                        cls=f"badge {badge_color} text-lg p-3"
                    ),
                    id=f"classification-section-{i}",
                    cls=f"mb-6 {'hidden' if i > 0 else ''}"
                )
                
            classification_sections.append(context_section)
        
        # Create the complete carousel component
        return Div(
            H2("Insect Classification Results", cls="text-2xl font-bold text-white mb-4 text-center"),
            
            # Main carousel container with images
            Div(
                *carousel_items,
                cls="carousel w-full rounded-lg overflow-hidden mb-4"
            ),
            
            # Carousel indicators
            Div(
                *carousel_indicators,
                cls="flex justify-center w-full gap-2 py-2"
            ),
            
            # Classification sections container
            Div(
                *classification_sections,
                id="classification-sections",
                cls="mt-4"
            ),
            
            # Process another batch button
            Div(
                Button(
                    "Classify More Insects",
                    hx_get="/batch-upload",
                    hx_target="#main-content",
                    cls="btn btn-warning w-full max-w-xs"
                ),
                cls="mt-8 text-center w-full"
            ),
            
            # JavaScript to handle carousel navigation and show appropriate classification section
            Script("""
            document.addEventListener('DOMContentLoaded', function() {
                // Function to show the right classification section based on active carousel item
                const updateClassificationSection = function() {
                    const id = window.location.hash.substring(1);
                    if (!id) return;
                    
                    // Get the index from the item id (item1, item2, etc.)
                    const index = parseInt(id.replace('item', '')) - 1;
                    
                    // Hide all classification sections
                    document.querySelectorAll('[id^="classification-section-"]').forEach(section => {
                        section.classList.add('hidden');
                    });
                    
                    // Show the corresponding section
                    const currentSection = document.getElementById(`classification-section-${index}`);
                    if (currentSection) {
                        currentSection.classList.remove('hidden');
                    }
                    
                    // Update active indicator
                    document.querySelectorAll('[href^="#item"]').forEach(indicator => {
                        if (indicator.getAttribute('href') === '#' + id) {
                            indicator.classList.add('btn-active');
                        } else {
                            indicator.classList.remove('btn-active');
                        }
                    });
                };
                
                // Listen for hash changes
                window.addEventListener('hashchange', updateClassificationSection);
                
                // Initial update - if no hash, set to first item
                if (!window.location.hash) {
                    window.location.hash = 'item1';
                } else {
                    updateClassificationSection();
                }
            });
            """),
            
            id="batch-results",
            cls="w-full flex flex-col items-center bg-zinc-900 rounded-md p-6 fade-in"
        )

    # Add this helper function to directly embed images in the UI
    def try_read_image(image_path):
        """Attempt to read an image and return it as an embedded element"""
        try:
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_data = f.read()
                    base64_img = base64.b64encode(img_data).decode('utf-8')
                
                # Determine image type from extension
                ext = os.path.splitext(image_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    media_type = "image/jpeg"
                elif ext == '.png':
                    media_type = "image/png"
                elif ext == '.gif':
                    media_type = "image/gif"
                else:
                    media_type = "image/jpeg"  # Default to JPEG
                
                # Display as embedded base64 image
                return Img(
                    src=f"data:{media_type};base64,{base64_img}",
                    cls="mx-auto max-h-96 w-full object-contain rounded-lg border border-zinc-700"
                )
            else:
                # Try to find any image with the same analysis ID prefix
                analysis_id = os.path.basename(image_path).split('_')[0]
                directory = os.path.dirname(image_path)
                
                if os.path.exists(directory):
                    files = os.listdir(directory)
                    matching_files = [f for f in files if f.startswith(analysis_id)]
                    
                    if matching_files:
                        alt_path = os.path.join(directory, matching_files[0])
                        if os.path.exists(alt_path):
                            with open(alt_path, "rb") as f:
                                img_data = f.read()
                                base64_img = base64.b64encode(img_data).decode('utf-8')
                            
                            # Determine image type from extension
                            alt_ext = os.path.splitext(matching_files[0])[1].lower()
                            if alt_ext in ['.jpg', '.jpeg']:
                                media_type = "image/jpeg"
                            elif alt_ext == '.png':
                                media_type = "image/png"
                            elif alt_ext == '.gif':
                                media_type = "image/gif"
                            else:
                                media_type = "image/jpeg"  # Default to JPEG
                            
                            return Img(
                                src=f"data:{media_type};base64,{base64_img}",
                                cls="mx-auto max-h-96 w-full object-contain rounded-lg border border-zinc-700"
                            )
                
                # If all fails, show placeholder
                return Div(
                    P("Image not found", cls="text-red-500 text-center"),
                    cls="w-full h-64 bg-zinc-800 rounded-lg border border-zinc-700 flex items-center justify-center"
                )
        except Exception as e:
            logging.error(f"Error embedding image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return Div(
                P(f"Error displaying image: {str(e)}", cls="text-red-500 text-center"),
                cls="w-full h-64 bg-zinc-800 rounded-lg border border-zinc-700 flex items-center justify-center"
            )

    # Helper function to read PDF images directly
    def try_read_pdf_image(image_key):
        """Attempt to read a PDF image and return it as an embedded element"""
        global page_images
        
        try:
            if image_key in page_images:
                image_path = page_images[image_key]
                if os.path.exists(image_path):
                    with open(image_path, "rb") as f:
                        img_data = f.read()
                        base64_img = base64.b64encode(img_data).decode('utf-8')
                    
                    # Display as embedded base64 image
                    return Img(
                        src=f"data:image/png;base64,{base64_img}",
                        cls="w-full rounded-lg border border-zinc-700 max-h-80 object-contain mx-auto"
                    )
            
            # If path not found, try alternative paths
            parts = image_key.split('_')
            if len(parts) >= 2:
                filename = '_'.join(parts[:-1])
                page_num = int(parts[-1]) if parts[-1].isdigit() else 0
                potential_paths = [
                    os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                    os.path.join(PDF_IMAGES_DIR, f"{filename}", f"page_{page_num}.png"),
                    os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
                ]
                
                for path in potential_paths:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            img_data = f.read()
                            base64_img = base64.b64encode(img_data).decode('utf-8')
                        
                        # Display as embedded base64 image
                        return Img(
                            src=f"data:image/png;base64,{base64_img}",
                            cls="w-full rounded-lg border border-zinc-700 max-h-80 object-contain mx-auto"
                        )
            
            # If all fails, show a placeholder
            return Div(
                P("Context document image not found", cls="text-red-500 text-center"),
                cls="w-full h-64 bg-zinc-800 rounded-lg border border-zinc-700 flex items-center justify-center"
            )
        except Exception as e:
            logging.error(f"Error embedding PDF image {image_key}: {e}")
            import traceback
            traceback.print_exc()
            return Div(
                P(f"Error displaying context document: {str(e)}", cls="text-red-500 text-center"),
                cls="w-full h-64 bg-zinc-800 rounded-lg border border-zinc-700 flex items-center justify-center"
            )

    # BATCH PROCESSING ROUTES
    # Updated main route function
    @rt("/")
    def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        logging.info(f"New session: {session['session_id']} - showing batch upload form")
        
        return (
            Title("Insect Classification"),
            Main(
                # Loading indicator with better visibility
                Div(
                    Div(cls="loading loading-spinner loading-lg text-warning"),
                    Div("Processing your insect images...", cls="text-white mt-4 text-lg"),
                    id="loading-indicator",
                    cls="htmx-indicator fixed top-0 left-0 w-full h-full bg-black bg-opacity-80 flex flex-col items-center justify-center z-50"
                ),
                
                # Page header 
                H1("Insect Classifier", cls="text-3xl font-bold mb-4 text-white"),
                
                # Header info
                Div(
                    P("Upload insect images for instant AI classification", 
                      cls="text-white text-center mb-6"),
                    cls="w-full max-w-2xl"
                ),
                
                # Main content area - DIRECTLY SHOW BATCH FORM
                Div(
                    batch_upload_form(),
                    id="main-content",
                    cls="w-full max-w-2xl"
                ),
                
                # Results area - will be populated by process-batch
                Div(id="analysis-results", cls="w-full max-w-5xl mt-8"),
                
                cls="flex flex-col items-center min-h-screen bg-black p-4",
            )
        )

    # Add this explicit route for batch upload
    @rt("/batch-upload", methods=["GET"])
    def get_batch_upload(session):
        """Show the batch upload form"""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            
        logging.info(f"Showing batch upload form for session: {session['session_id']}")
        return batch_upload_form()

    # Process batch of images
    @rt("/process-batch", methods=["POST"])
    async def process_batch(request: Request):
        """Process a batch of uploaded images for insect classification"""
        form = await request.form()
        
        # Extract all uploaded images
        image_files = []
        for key in form.keys():
            if key.startswith('image_'):
                image_files.append(form.get(key))
        
        # If no image_X fields, try the multiple file field
        if not image_files and form.get("image_files"):
            image_files = form.getlist("image_files")
        
        # Limit to 10 images
        image_files = image_files[:10]
        
        if not image_files:
            return Div("No insect images uploaded", cls="text-red-500 text-center p-4")
        
        logging.info(f"Processing batch of {len(image_files)} insect images")
        
        # Set the query for insect classification
        query = "Classify this insect"
        share_context = form.get("share_context", "false") == "true"
        logging.info(f"Using query: {query}, Share context: {share_context}")
        
        # For shared context, retrieve documents once
        shared_context = None
        shared_top_sources = None
        
        if share_context:
            logging.info(f"Retrieving shared context for query: {query}")
            shared_context, shared_top_sources = await retrieve_relevant_documents(query)
            logging.info(f"Retrieved {len(shared_context) if shared_context else 0} shared context paragraphs")
        
        # Create directory for temporary uploads if it doesn't exist
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        
        # Process each image
        batch_results = []
        token_maps_by_image = {}
        
        for i, image_file in enumerate(image_files):
            if not image_file:
                continue
                
            try:
                # Generate a unique ID for this analysis
                analysis_id = str(uuid.uuid4())
                
                # Get filename or create default
                filename = getattr(image_file, 'filename', f"image_{i}.jpg")
                # Sanitize filename
                safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                image_path = os.path.join(TEMP_UPLOAD_DIR, f"{analysis_id}_{safe_filename}")
                
                logging.info(f"Processing insect image {i+1}/{len(image_files)}: {filename} → {image_path}")
                
                # Save the uploaded image
                content = await image_file.read()
                with open(image_path, "wb") as f:
                    f.write(content)
                
                # Verify the file was saved
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Failed to save image to {image_path}")
                    
                # Open the image for processing
                image = Image.open(image_path)
                
                # Get context - either shared or unique per image
                if not share_context:
                    logging.info(f"Retrieving unique context for insect image {i+1}")
                    retrieved_paragraphs, top_sources = await retrieve_relevant_documents(query)
                else:
                    logging.info(f"Using shared context for insect image {i+1}")
                    retrieved_paragraphs, top_sources = shared_context, shared_top_sources
                
                # Generate context text
                context_text = "\n\n".join(retrieved_paragraphs) if retrieved_paragraphs else ""
                
                # Generate token maps for top document
                image_token_maps = {}
                if top_sources:
                    top_source = top_sources[0]
                    image_key = top_source.get('image_key')
                    
                    # Only generate token maps if we haven't already for this context
                    if image_key and image_key not in token_maps_by_image:
                        logging.info(f"Generating token maps for context document: {image_key}")
                        image_heatmaps = await generate_similarity_maps(query, image_key)
                        if image_heatmaps:
                            token_maps_by_image[image_key] = image_heatmaps
                            image_token_maps[image_key] = image_heatmaps
                    elif image_key:
                        # Reuse existing token maps
                        image_token_maps[image_key] = token_maps_by_image[image_key]
                
                # Process with Mistral using insect classification system prompt
                logging.info(f"Processing insect image {i+1} with Mistral")
                response_text = await process_with_mistral(image, query, context_text, analysis_id)
                
                # Save to database
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Insert basic info
                    cursor.execute(
                        "INSERT INTO image_analyses (analysis_id, image_path, analysis_type, query, response) VALUES (?, ?, ?, ?, ?)",
                        (analysis_id, image_path, "insect_classification", query, response_text)
                    )
                    
                    # Add context source if available
                    if top_sources:
                        top_source = top_sources[0]
                        context_source = f"{top_source['filename']} (page {top_source['page']})"
                        
                        # Check if the column exists
                        cursor.execute("PRAGMA table_info(image_analyses)")
                        columns = [column[1] for column in cursor.fetchall()]
                        if 'context_source' in columns:
                            cursor.execute(
                                "UPDATE image_analyses SET context_source = ? WHERE analysis_id = ?",
                                (context_source, analysis_id)
                            )
                    
                    conn.commit()
                    conn.close()
                    logging.info(f"Saved analysis {analysis_id} to database")
                except Exception as db_error:
                    logging.error(f"Database error for insect image {i+1}: {db_error}")
                
                # Add to batch results
                batch_results.append({
                    "analysis_id": analysis_id,
                    "image_path": image_path,
                    "response": response_text,
                    "context_paragraphs": retrieved_paragraphs,
                    "top_sources": top_sources,
                    "token_maps": image_token_maps
                })
                
            except Exception as e:
                logging.error(f"Error processing insect image {i+1}: {e}")
                import traceback
                traceback.print_exc()
                
                # Add error result
                batch_results.append({
                    "error": True,
                    "message": f"Error: {str(e)}",
                    "analysis_id": f"error_{i}",
                    "image_path": ""
                })
        
        # Ensure volume is committed
        try:
            bee_volume.commit()
        except Exception as e:
            logging.error(f"Error committing volume: {e}")
        
        # Return carousel UI with all results
        return carousel_ui(batch_results)

    @rt("/token-map/{image_key}/{token_idx}")
    async def get_token_map(image_key: str, token_idx: int):
        """Return the token map image HTML for display in the UI"""
        # Get the file path
        heatmap_filename = f"{image_key}_token_{token_idx}.png"
        heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
        logging.info(f"Looking for token map: {heatmap_path}")
        
        if os.path.exists(heatmap_path):
            logging.info(f"Found token map at: {heatmap_path}")
            # Encode the image directly to base64 for inline display
            try:
                with open(heatmap_path, "rb") as f:
                    img_data = f.read()
                    base64_img = base64.b64encode(img_data).decode('utf-8')
                    
                # Use an inline data URL instead of a separate request
                return Div(
                    Img(
                        src=f"data:image/png;base64,{base64_img}",
                        cls="mx-auto max-h-96 w-full object-contain token-map"
                    ),
                    cls="p-2 flex items-center justify-center min-h-[300px]"
                )
            except Exception as e:
                logging.error(f"Error reading token map file: {e}")
                import traceback
                traceback.print_exc()
                return Div(f"Error reading token map: {str(e)}", cls="text-red-500 p-4")
        else:
            logging.error(f"Token map not found at: {heatmap_path}")
            return Div(f"Token map not found for {image_key} token {token_idx}", cls="text-red-500 p-4")

    @fasthtml_app.get("/temp-image/{filename}")
    async def serve_temp_image(filename: str):
        """Serve temporary images (uploads and similarity maps)"""
        image_path = os.path.join(TEMP_UPLOAD_DIR, filename)
        logging.info(f"Requested temp image: {filename}, looking at: {image_path}")
        
        if os.path.exists(image_path):
            logging.info(f"Found temp image at: {image_path}, size: {os.path.getsize(image_path)} bytes")
            try:
                # Open and read the file directly
                with open(image_path, "rb") as f:
                    content = f.read()
                
                # Determine media type based on file extension
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    media_type = "image/jpeg"
                elif ext == '.png':
                    media_type = "image/png"
                elif ext == '.gif':
                    media_type = "image/gif"
                else:
                    media_type = "application/octet-stream"
                
                # Return as binary response
                return Response(
                    content=content,
                    media_type=media_type
                )
            except Exception as e:
                logging.error(f"Error reading temp image file: {e}")
                import traceback
                traceback.print_exc()
                return Response(
                    content=f"Error reading image: {str(e)}",
                    media_type="text/plain",
                    status_code=500
                )
        else:
            logging.error(f"Temp image not found: {image_path}")
            
            # Check for files with similar name patterns
            try:
                files = os.listdir(TEMP_UPLOAD_DIR)
                
                # Look for files with the analysis_id prefix
                analysis_id = filename.split('_')[0]
                matching_files = [f for f in files if f.startswith(analysis_id)]
                
                if matching_files:
                    logging.info(f"Found similar files: {matching_files}")
                    
                    # Try to use the first matching file instead
                    if len(matching_files) > 0:
                        alt_path = os.path.join(TEMP_UPLOAD_DIR, matching_files[0])
                        logging.info(f"Trying alternative file: {alt_path}")
                        
                        if os.path.exists(alt_path):
                            logging.info(f"Using alternative file: {alt_path}")
                            with open(alt_path, "rb") as f:
                                content = f.read()
                            
                            # Determine media type based on file extension
                            alt_ext = os.path.splitext(matching_files[0])[1].lower()
                            if alt_ext in ['.jpg', '.jpeg']:
                                media_type = "image/jpeg"
                            elif alt_ext == '.png':
                                media_type = "image/png"
                            elif alt_ext == '.gif':
                                media_type = "image/gif"
                            else:
                                media_type = "application/octet-stream"
                            
                            return Response(
                                content=content,
                                media_type=media_type
                            )
                
                # If we get here, no good alternative was found
                logging.info(f"All files in {TEMP_UPLOAD_DIR}: {files[:20]}")
            except Exception as e:
                logging.error(f"Error listing directory: {e}")
            
            return Response(
                content=f"Image not found: {filename}",
                media_type="text/plain",
                status_code=404
            )

    @fasthtml_app.get("/pdf-image/{image_key}")
    async def get_pdf_image(image_key: str):
        """Serve PDF page images"""
        logging.info(f"Image request for key: {image_key}")
        try:
            if image_key in page_images:
                image_path = page_images[image_key]
                logging.info(f"Found image path: {image_path}")
                if os.path.exists(image_path):
                    logging.info(f"Image file exists, serving from: {image_path}")
                    try:
                        # Open and read the file directly
                        with open(image_path, "rb") as f:
                            content = f.read()
                        
                        # Return as binary response
                        return Response(
                            content=content,
                            media_type="image/png"
                        )
                    except Exception as e:
                        logging.error(f"Error reading PDF image file: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logging.error(f"Image file does not exist at path: {image_path}")
                    # Try to find the image by reconstructing path patterns
                    parts = image_key.split('_')
                    if len(parts) >= 2:
                        filename = '_'.join(parts[:-1])
                        page_num = int(parts[-1]) if parts[-1].isdigit() else 0
                        potential_paths = [
                            # Check different path patterns
                            os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                            os.path.join(PDF_IMAGES_DIR, f"{filename}", f"page_{page_num}.png"),
                            os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
                        ]
                        
                        for path in potential_paths:
                            logging.info(f"Trying alternative path: {path}")
                            if os.path.exists(path):
                                logging.info(f"Found image at alternative path: {path}")
                                with open(path, "rb") as f:
                                    content = f.read()
                                return Response(
                                    content=content,
                                    media_type="image/png"
                                )
            else:
                logging.error(f"Image key '{image_key}' not found in page_images dictionary")
                # List available keys for debugging
                available_keys = list(page_images.keys())[:10]  # First 10 keys
                logging.info(f"Available keys (first 10): {available_keys}")
        except Exception as e:
            logging.error(f"Error in get_pdf_image: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Return a placeholder image or error response
        return Response(
            content=f"Image not found for key: {image_key}", 
            media_type="text/plain", 
            status_code=404
        )

    @fasthtml_app.get("/heatmap-image/{filename}")
    async def get_heatmap_image(filename: str):
        """Serve heatmap images"""
        heatmap_path = os.path.join(HEATMAP_DIR, filename)
        logging.info(f"Looking for heatmap image: {heatmap_path}")
        
        if os.path.exists(heatmap_path):
            logging.info(f"Found heatmap at: {heatmap_path}, size: {os.path.getsize(heatmap_path)} bytes")
            try:
                # Open and read the file directly
                with open(heatmap_path, "rb") as f:
                    content = f.read()
                    
                # Return as binary response
                return Response(
                    content=content,
                    media_type="image/png"
                )
            except Exception as e:
                logging.error(f"Error reading heatmap file: {e}")
                import traceback
                traceback.print_exc()
                return Response(
                    content=f"Error reading heatmap: {str(e)}",
                    media_type="text/plain",
                    status_code=500
                )
        else:
            logging.error(f"Heatmap image not found: {heatmap_path}")
            # List what files do exist in the directory
            try:
                files = os.listdir(HEATMAP_DIR)
                matching_files = [f for f in files if f.startswith(filename.split('_token_')[0])]
                logging.info(f"Files in {HEATMAP_DIR} that match prefix: {matching_files}")
            except Exception as e:
                logging.error(f"Error listing directory: {e}")
                
            return Response(
                content=f"Heatmap not found: {filename}",
                media_type="text/plain",
                status_code=404
            )

    @fasthtml_app.get("/debug-info")
    async def debug_info():
        """Return debugging information about the environment"""
        info = {
            "temp_upload_dir_exists": os.path.exists(TEMP_UPLOAD_DIR),
            "temp_upload_files": os.listdir(TEMP_UPLOAD_DIR)[:20] if os.path.exists(TEMP_UPLOAD_DIR) else [],
            "heatmap_dir_exists": os.path.exists(HEATMAP_DIR),
            "heatmap_files": os.listdir(HEATMAP_DIR)[:20] if os.path.exists(HEATMAP_DIR) else [],
            "pdf_images_dir_exists": os.path.exists(PDF_IMAGES_DIR),
            "pdf_images_subfolders": os.listdir(PDF_IMAGES_DIR)[:20] if os.path.exists(PDF_IMAGES_DIR) else [],
            "page_images_count": len(page_images),
            "page_images_sample": list(page_images.keys())[:10]
        }
        
        return Div(
            H2("Debugging Information", cls="text-xl font-semibold text-white mb-4"),
            Pre(json.dumps(info, indent=2), cls="bg-zinc-800 p-4 rounded-md text-white"),
            cls="w-full max-w-2xl mx-auto bg-zinc-900 rounded-md p-6"
        )

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
