from components.assets import arrow_circle_icon, github_icon
from components.chat import chat, chat_form, chat_message
import asyncio
import modal
from fasthtml.common import *
import fastapi
import logging
from transformers import AutoTokenizer
import uuid
from modal import Secret
from fastlite import Database
from starlette.middleware.sessions import SessionMiddleware
import aiohttp
import os
import sqlite3
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map

# Constants
QWEN_MODELS_DIR = "/Qwen"  # existing volume
COLQWEN_MODELS_DIR = "/ColQwen"  # new volume for ColQwen2 models
DATA_DIR = "/bee_pdf"
UPLOADED_PDFS_DIR = "/bee_pdf/uploaded_pdfs"
PDF_IMAGES_DIR = "/bee_pdf/pdf_images"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Default model names
DEFAULT_QWEN_NAME = "Qwen/Qwen2.5-7B-Instruct-1M"
DEFAULT_COLQWEN_NAME = "vidore/colqwen2-v1.0"
USERNAME = "c123ian"
APP_NAME = "polliknow-rag"
DATABASE_DIR = "/db_rag_advan"

db_path = os.path.join(DATABASE_DIR, 'chat_history.db')
os.makedirs(DATABASE_DIR, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations_history_table_sqlalchemy_v2 (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        top_source_headline TEXT,
        top_source_url TEXT,
        cosine_sim_score REAL, 
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()

db = Database(db_path)
conversations = db['conversations']

logging.basicConfig(level=logging.INFO)

# Set up volumes
try:
    qwen_volume = modal.Volume.lookup("Qwen", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download Qwen models first with the appropriate script")

try:
    colqwen_volume = modal.Volume.lookup("ColQwen", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download ColQwen models first with the appropriate script")

# Update: add colpali-engine to the install list
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

# Already created via embedding_colpali.py
bee_volume = modal.Volume.from_name("bee_pdf", create_if_missing=False)

try:
    db_volume = modal.Volume.lookup("db_data", create_if_missing=True)
except modal.exception.NotFoundError:
    db_volume = modal.Volume.persisted("db_data")

app = modal.App(APP_NAME)

@app.function(
    image=image,
    gpu=modal.gpu.A10G(count=1),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={
        QWEN_MODELS_DIR: qwen_volume,
        COLQWEN_MODELS_DIR: colqwen_volume,
        DATA_DIR: bee_volume
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

    print(f"Initializing AsyncLLMEngine with model path: {model_path} and tokenizer path: {tokenizer_path}")

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
            request_id = str(uuid.uuid4())

            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens,
                stop=["User:", "Assistant:", "\n\n"],
            )

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
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app

@app.function(
    image=image,
    volumes={
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume,
        COLQWEN_MODELS_DIR: colqwen_volume  # Add ColQwen volume
    },
    gpu=modal.gpu.A10G(count=1),  # Explicitly request GPU for ColQwen
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    # Removing faiss dependency as it's not needed when using ColQwen directly
    import os
    import pickle
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    import logging
    from starlette.middleware import Middleware
    from starlette.websockets import WebSocket
    import uuid
    import asyncio
    from sqlalchemy import create_engine, Column, String, DateTime, Float
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    import datetime
    from rerankers import Reranker
    from nltk.tokenize import word_tokenize
    import nltk
    import numpy as np
    from rank_bm25 import BM25Okapi
    from fastapi.responses import FileResponse, Response, HTMLResponse
    from io import BytesIO
    import base64
    from PIL import Image
    from pdf2image import convert_from_path
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map
    
    # Helper function to ensure ColQwen model is loaded from volume or HuggingFace
    def ensure_colqwen_model_loaded():
        """Ensure the ColQwen2 model is properly loaded"""
        global colpali_model, colpali_processor, colpali_embeddings
        
        # Verify CUDA is available
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA is NOT available, ColQwen will run on CPU which is slower")
        
        # Check if we need to load the model
        if 'colpali_model' not in globals() or colpali_model is None:
            print(f"Loading ColQwen2 model...")
            try:
                # Initialize model directly from HuggingFace for now
                # We'll preload to volume in a separate step
                colpali_model = ColQwen2.from_pretrained(
                    DEFAULT_COLQWEN_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
                colpali_processor = ColQwen2Processor.from_pretrained(DEFAULT_COLQWEN_NAME)
                
                print(f"ColQwen2 model loaded successfully on device: {colpali_model.device}")
                
                # Load embeddings if needed
                if 'colpali_embeddings' not in globals() or colpali_embeddings is None:
                    colpali_embeddings_path = os.path.join(DATA_DIR, "colpali_embeddings.pkl")
                    print(f"Loading ColQwen2 embeddings from {colpali_embeddings_path}")
                    with open(colpali_embeddings_path, "rb") as f:
                        colpali_embeddings = pickle.load(f)
                    print(f"Loaded {len(colpali_embeddings)} ColQwen2 embeddings")
                
                return True
            except Exception as e:
                print(f"Error loading ColQwen2 model: {e}")
                import traceback
                traceback.print_exc()
                return False
        return True

    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

    print(f"Contents of DATA_DIR ({DATA_DIR}):")
    if os.path.exists(DATA_DIR):
        print(f"  Directory exists, contains: {os.listdir(DATA_DIR)}")
    else:
        print(f"  Directory does not exist!")

    print(f"Contents of PDF_IMAGES_DIR ({PDF_IMAGES_DIR}):")
    if os.path.exists(PDF_IMAGES_DIR):
        print(f"  Directory exists, contains: {os.listdir(PDF_IMAGES_DIR)}")
    else:
        print(f"  Directory does not exist!")

    # Path definitions
    DATA_PICKLE_PATH = os.path.join(DATA_DIR, "data.pkl")
    PDF_PAGE_IMAGES_PATH = os.path.join(DATA_DIR, "pdf_page_image_paths.pkl")
    COLPALI_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "colpali_embeddings.pkl")

    # Check if we're using ColPali embeddings
    using_colpali = os.path.exists(COLPALI_EMBEDDINGS_PATH)
    
    if using_colpali:
        print(f"Loading ColPali embeddings from {COLPALI_EMBEDDINGS_PATH}")
        with open(COLPALI_EMBEDDINGS_PATH, "rb") as f:
            colpali_embeddings = pickle.load(f)
        print(f"Loaded {len(colpali_embeddings)} ColPali embeddings")
        
        # Make sure the model is loaded (call the helper function)
        ensure_colqwen_model_loaded()
    else:
        print("ColPali embeddings not found")
        # Load original embedding model
        emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Load BM25 index 
        try:
            with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f:
                bm25_index = pickle.load(f)
            with open(os.path.join(DATA_DIR, "tokenized_paragraphs.pkl"), "rb") as f:
                tokenized_docs = pickle.load(f)
        except Exception as e:
            print(f"Error loading BM25 index: {e}")
            def create_bm25_index(documents):
                tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
                bm25_index = BM25Okapi(tokenized_docs)
                return bm25_index, tokenized_docs
            docs = []
            bm25_index, tokenized_docs = create_bm25_index(docs)
    
    # Load data frame with metadata
    print(f"Loading DataFrame from {DATA_PICKLE_PATH}")
    print(f"  File exists: {os.path.exists(DATA_PICKLE_PATH)}")
    df = pd.read_pickle(DATA_PICKLE_PATH)
    
    # Load image paths
    print(f"Loading image paths from {PDF_PAGE_IMAGES_PATH}")
    print(f"  File exists: {os.path.exists(PDF_PAGE_IMAGES_PATH)}")
    page_images = {}
    try:
        with open(PDF_PAGE_IMAGES_PATH, "rb") as f:
            page_images = pickle.load(f)
        print(f"  Loaded {len(page_images)} image paths")
    except Exception as e:
        print(f"  Error loading image paths: {e}")
        logging.error(f"Error loading PDF page images: {e}")
    
    # Function to create token selector component - define for reference but we'll use direct HTML generation
    def token_list_component(tokens, image_key):
        """Create the token selector component with the fetched tokens"""
        return Select(
            *[Option(f"{t['idx']}: {t['token']}", value=t['idx']) for t in tokens],
            id=f"token-selector-{image_key}",
            cls="select select-bordered w-full max-w-xs bg-zinc-800 text-white mb-2"
        )

    # Function to create similarity map viewer component
    def similarity_map_viewer(image_key):
        """Create a UI component for viewing similarity maps"""
        return Div(
            Div("Token Similarity Maps", cls="text-zinc-400 text-sm font-semibold mb-2"),
            
            # Token selector
            Div(
                Select(
                    Option("Loading tokens...", value="", disabled=True),
                    id=f"token-selector-{image_key}",
                    cls="select select-bordered w-full max-w-xs bg-zinc-800 text-white mb-2",
                    hx_get=f"/token-list/{image_key}",
                    hx_trigger="load",
                    hx_swap="innerHTML"
                ),
                cls="mb-4"
            ),
            
            # Map display area - empty initially
            Div(
                Div("Select a token to view its similarity map", cls="text-zinc-400"),
                id=f"similarity-map-container-{image_key}",
                cls="w-full h-96 flex items-center justify-center bg-zinc-800 rounded-md overflow-hidden"
            ),
            
            # Script to handle token selection
            Script(
                f"""
                document.addEventListener('htmx:afterSwap', function(evt) {{
                    if (evt.detail.target.id === 'token-selector-{image_key}') {{
                        const tokenSelector = document.getElementById('token-selector-{image_key}');
                        
                        if (tokenSelector) {{
                            tokenSelector.addEventListener('change', function() {{
                                const selectedToken = this.value;
                                const mapContainer = document.getElementById('similarity-map-container-{image_key}');
                                
                                if (mapContainer && selectedToken) {{
                                    // Show loading indicator
                                    mapContainer.innerHTML = '<div class="text-white">Loading map...</div>';
                                    
                                    // Load the similarity map image
                                    mapContainer.innerHTML = `<img src="/similarity-map/{image_key}/${{selectedToken}}" 
                                                                class="max-w-full max-h-full object-contain" />`;
                                }}
                            }});
                        }}
                    }}
                }});
                """
            ),
            
            cls="w-full bg-zinc-800 rounded-md p-4 mt-4"
        )

    # Function to display top sources with similarity maps
    def chat_top_sources(top_sources):
        # Just show the top 1 most relevant source
        top_sources = top_sources[:1]
        
        carousel_items = [
            Div(
                Img(
                    src=f"/pdf-image/{source['image_key']}",
                    cls="w-full rounded-lg border border-zinc-700"
                ),
                id=f"item{i+1}",
                cls="carousel-item w-full"
            )
            for i, source in enumerate(top_sources)
        ]

        # Simplified for just one image - no carousel controls needed
        document_display = Div(
            Div(*carousel_items, cls="w-full"),
            cls="flex flex-col w-full"
        )
        
        # Create a more direct token similarity UI for the top source
        token_exploration = None
        if using_colpali and top_sources:
            source = top_sources[0]
            image_key = source['image_key']
            
            # Create hardcoded default tokens for immediate use
            default_token_options = """
                <option value="0">0: Query</option>
                <option value="1">1: classify</option>
                <option value="2">2: honeybee</option>
                <option value="3">3: document</option>
                <option value="4">4: image</option>
            """
            
            token_exploration = Div(
                H3("Explore Token Similarity", cls="text-lg font-semibold text-white mb-3"),
                P("Select a token to see which parts of the document the model focuses on for that word:", cls="text-zinc-300 mb-4"),
                
                # Token selector with default options already populated
                Div(
                    Select(
                        NotStr(default_token_options),
                        id=f"token-selector-{image_key}",
                        cls="select select-bordered w-full bg-zinc-800 text-white mb-4",
                        hx_get=f"/token-list/{image_key}",
                        hx_trigger="load",
                        hx_target="this",
                        hx_swap="outerHTML"
                    ),
                    cls="token-selector-container"
                ),
                
                # Map display area - auto-load the first token map
                Div(
                    Img(
                        src=f"/similarity-map/{image_key}/1",
                        cls="max-w-full max-h-full object-contain opacity-80 hover:opacity-100 transition-opacity"
                    ),
                    id=f"similarity-map-container-{image_key}",
                    cls="w-full h-96 flex items-center justify-center bg-zinc-800 rounded-md overflow-hidden"
                ),
                
                # Script to handle token selection
                Script(
                    f"""
                    document.addEventListener('htmx:afterSettle', function(evt) {{
                        // Check if our dropdown was updated
                        if (evt.detail.target && evt.detail.target.id === "token-selector-{image_key}") {{
                            console.log("Token selector loaded");
                            // Initialize with first token
                            var mapContainer = document.getElementById('similarity-map-container-{image_key}');
                            if (mapContainer) {{
                                var imgElement = mapContainer.querySelector('img');
                                if (!imgElement) {{
                                    mapContainer.innerHTML = '<img src="/similarity-map/{image_key}/1" class="max-w-full max-h-full object-contain" />';
                                }}
                            }}
                        }}
                        
                        // Set up event listener for dropdown changes
                        const tokenSelector = document.getElementById('token-selector-{image_key}');
                        if (tokenSelector && !tokenSelector.hasAttribute('data-initialized')) {{
                            tokenSelector.setAttribute('data-initialized', 'true');
                            tokenSelector.addEventListener('change', function() {{
                                const selectedToken = this.value;
                                const mapContainer = document.getElementById('similarity-map-container-{image_key}');
                                
                                if (mapContainer && selectedToken) {{
                                    // Show loading indicator
                                    mapContainer.innerHTML = '<div class="text-white">Loading map...</div>';
                                    
                                    // Load the similarity map image
                                    mapContainer.innerHTML = `<img src="/similarity-map/{image_key}/${{selectedToken}}" 
                                                              class="max-w-full max-h-full object-contain" />`;
                                }}
                            }});
                        }}
                    }});
                    
                    // Initialize immediately
                    (function() {{
                        const tokenSelector = document.getElementById('token-selector-{image_key}');
                        if (tokenSelector && !tokenSelector.hasAttribute('data-initialized')) {{
                            tokenSelector.setAttribute('data-initialized', 'true');
                            tokenSelector.addEventListener('change', function() {{
                                const selectedToken = this.value;
                                const mapContainer = document.getElementById('similarity-map-container-{image_key}');
                                
                                if (mapContainer && selectedToken) {{
                                    // Show loading indicator
                                    mapContainer.innerHTML = '<div class="text-white">Loading map...</div>';
                                    
                                    // Load the similarity map image
                                    mapContainer.innerHTML = `<img src="/similarity-map/{image_key}/${{selectedToken}}" 
                                                              class="max-w-full max-h-full object-contain" />`;
                                }}
                            }});
                        }}
                    }})();
                    """
                ),
                
                cls="w-full bg-zinc-800 rounded-md p-4 mt-6"
            )

        return Div(
            Div(
                Div("Retrieved Document", cls="text-zinc-300 text-lg font-semibold mb-4"),
                document_display,
                token_exploration if token_exploration else "",
                cls="flex flex-col w-full gap-6"
            ),
            cls="w-full max-w-2xl mx-auto bg-zinc-800 rounded-md mt-6 p-6"
        )

    fasthtml_app, rt = fast_app(
        hdrs=(
            Script(src="https://cdn.tailwindcss.com"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"),
        ),
        ws_hdr=True,
        middleware=[
            Middleware(
                SessionMiddleware,
                secret_key=os.environ.get('YOUR_KEY'),
                session_cookie="secure_session",
                max_age=86400,
                same_site="strict",
                https_only=True
            )
        ]
    )

    session_messages = {}
    Base = declarative_base()

    class Conversation(Base):
        __tablename__ = 'conversations_history_table_sqlalchemy_v2'
        message_id = Column(String, primary_key=True)
        session_id = Column(String, nullable=False)
        role = Column(String, nullable=False)
        content = Column(String, nullable=False)
        top_source_headline = Column(String)
        top_source_url = Column(String)
        cosine_sim_score = Column(Float)
        created_at = Column(DateTime, default=datetime.datetime.utcnow)

    db_engine = create_engine(f'sqlite:///{os.path.join(DATABASE_DIR, "chat_history.db")}')
    Session = sessionmaker(bind=db_engine)
    sqlalchemy_session = Session()

    async def load_chat_history(session_id):
        if not isinstance(session_id, str):
            logging.warning(f"Invalid session_id type: {type(session_id)}. Converting to string.")
            session_id = str(session_id)
        if session_id not in session_messages:
            try:
                session_history = sqlalchemy_session.query(Conversation)\
                    .filter(Conversation.session_id == session_id)\
                    .order_by(Conversation.created_at)\
                    .all()
                session_messages[session_id] = [
                    {"role": msg.role, "content": msg.content} for msg in session_history
                ]
            except Exception as e:
                logging.error(f"Database error in load_chat_history: {e}")
                session_messages[session_id] = []
        return session_messages[session_id]

    @fasthtml_app.get("/")
    async def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        messages = await load_chat_history(session_id)
        return Div(
            H1("Chat with PDF Documents", cls="text-3xl font-bold mb-4 text-white"),
            Div(f"Session ID: {session_id}", cls="text-white mb-4"),
            chat(session_id=session_id, messages=messages),
            Div(Span("Model status: "), Span("⚫", id="model-status-emoji"), cls="model-status text-white mt-4"),
            Div(id="top-sources"),
            cls="flex flex-col items-center min-h-screen bg-black",
        )

    @fasthtml_app.get("/token-list/{image_key}")
    async def get_token_list(image_key: str):
        from fastapi.responses import HTMLResponse, JSONResponse
        
        logging.info(f"Token list request for image: {image_key}")
        
        if not using_colpali:
            return HTMLResponse(
                content="<div>Token visualization not available - requires ColPali/ColQwen model</div>",
                status_code=400
            )
        
        try:
            # Make sure model is loaded
            if not ensure_colqwen_model_loaded():
                return HTMLResponse(
                    content="<div>Failed to load ColQwen model</div>",
                    status_code=500
                )
                
            # Use a hardcoded default query if we can't find a recent one
            default_query = "analyze this document"
            
            # Get the most recent query
            latest_query = default_query
            try:
                for conv in sqlalchemy_session.query(Conversation).filter(
                    Conversation.role == 'user'
                ).order_by(Conversation.created_at.desc()):
                    latest_query = conv.content
                    break
            except Exception as db_err:
                logging.error(f"Database error when fetching latest query: {db_err}")
                # Continue with default query
            
            # Process the query with ColPali to get tokens
            processed_query = colpali_processor.process_queries([latest_query]).to(colpali_model.device)
            
            # Extract and decode query tokens
            query_content = colpali_processor.decode(processed_query.input_ids[0]).replace(
                colpali_processor.tokenizer.pad_token, "")
            
            # Handle query augmentation token if it exists
            query_augmentation_token = getattr(colpali_processor, 'query_augmentation_token', None)
            if query_augmentation_token is not None:
                query_content = query_content.replace(query_augmentation_token, "").strip()
                
            # Tokenize the query
            query_tokens = colpali_processor.tokenizer.tokenize(query_content)
            
            # Create token list for the selector - use only first 20 tokens if there are many
            max_tokens = 20
            tokens = [{"idx": idx, "token": token} for idx, token in enumerate(query_tokens[:max_tokens])]
            
            # Generate HTML directly instead of using token_list_component
            options_html = "\n".join([f'<option value="{t["idx"]}">{t["idx"]}: {t["token"]}</option>' for t in tokens])
            selector_html = f'<select id="token-selector-{image_key}" class="select select-bordered w-full max-w-xs bg-zinc-800 text-white mb-2">{options_html}</select>'
            return HTMLResponse(content=selector_html)
            
        except Exception as e:
            logging.error(f"Error getting token list: {e}")
            import traceback
            traceback.print_exc()
            # Return a basic selector with a few default options as fallback
            fallback_html = f'''
            <select id="token-selector-{image_key}" class="select select-bordered w-full max-w-xs bg-zinc-800 text-white mb-2">
                <option value="0">0: Query</option>
                <option value="1">1: First token</option>
                <option value="2">2: Second token</option>
            </select>
            '''
            return HTMLResponse(content=fallback_html)

    @fasthtml_app.get("/similarity-map/{image_key}/{token_idx}")
    async def get_similarity_map(image_key: str, token_idx: int):
        logging.info(f"Similarity map request for image: {image_key}, token: {token_idx}")
        
        # Check if we're using ColPali
        if not using_colpali:
            return Response(content="Similarity maps only available with ColPali/ColQwen models", 
                        media_type="text/plain", status_code=400)
        
        # Ensure model is loaded
        if not ensure_colqwen_model_loaded():
            return Response(content="Failed to load ColQwen model", 
                        media_type="text/plain", status_code=500)
                        
        try:
            # Get the original image
            if image_key in page_images:
                image_path = page_images[image_key]
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                else:
                    return Response(content=f"Image file not found: {image_path}", 
                                media_type="text/plain", status_code=404)
            else:
                return Response(content=f"Image key not found: {image_key}", 
                            media_type="text/plain", status_code=404)
            
            # Use a hardcoded default query based on recent queries in logs
            default_query = "classify honeybee"
            
            # Try to get the most recent query
            latest_query = default_query
            try:
                for conv in sqlalchemy_session.query(Conversation).filter(
                    Conversation.role == 'user'
                ).order_by(Conversation.created_at.desc()).limit(1):
                    latest_query = conv.content
                    break
            except Exception as db_err:
                logging.error(f"Database error when fetching latest query: {db_err}")
                # Continue with default query
            
            # Process the query and image with ColPali
            processed_image = colpali_processor.process_images([image]).to(colpali_model.device)
            processed_query = colpali_processor.process_queries([latest_query]).to(colpali_model.device)
            
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
            
            # Get the image mask
            image_mask = colpali_processor.get_image_mask(processed_image)
            
            # Generate the similarity maps
            similarity_maps = get_similarity_maps_from_embeddings(
                image_embeddings=image_embedding,
                query_embeddings=query_embedding,
                n_patches=n_patches,
                image_mask=image_mask
            )[0]  # Get the first (and only) item from the batch
            
            # Get tokens for reference
            query_content = colpali_processor.decode(processed_query.input_ids[0]).replace(
                colpali_processor.tokenizer.pad_token, "")
                
            # Handle query augmentation token if it exists
            query_augmentation_token = getattr(colpali_processor, 'query_augmentation_token', None)
            if query_augmentation_token is not None:
                query_content = query_content.replace(query_augmentation_token, "").strip()
                
            # Tokenize the query
            query_tokens = colpali_processor.tokenizer.tokenize(query_content)
            
            # Convert token_idx to int and use a default if out of range
            try:
                token_idx = int(token_idx)
                if token_idx < 0 or token_idx >= len(query_tokens):
                    token_idx = 0  # Use the first token as fallback
            except ValueError:
                token_idx = 0  # Default to first token
            
            # Safety check for similarity maps size
            if similarity_maps.shape[0] <= token_idx:
                token_idx = 0  # Use the first token as fallback
                
            # Get the similarity map for the specific token
            token_similarity_map = similarity_maps[token_idx]
            
            # Generate the visualization
            fig, ax = plot_similarity_map(
                image=image,
                similarity_map=token_similarity_map,
                figsize=(10, 10),
                show_colorbar=True
            )
            
            # Add a title with the token
            token_text = query_tokens[token_idx] if token_idx < len(query_tokens) else "Unknown"
            max_sim = token_similarity_map.max().item()
            ax.set_title(f"Token: '{token_text}' (MaxSim: {max_sim:.2f})")
            
            # Convert plot to image
            buf = io.BytesIO()
            canvas = FigureCanvas(fig)
            canvas.print_png(buf)
            buf.seek(0)
            
            # Return the image
            return Response(content=buf.getvalue(), media_type="image/png")
            
        except Exception as e:
            logging.error(f"Error generating similarity map: {e}")
            import traceback
            traceback.print_exc()
            
            # Create a simple error image
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, f"Error generating similarity map:\n{str(e)}", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            
            return Response(content=buf.getvalue(), media_type="image/png")

    @fasthtml_app.get("/pdf-image/{image_key}")  # Use FastAPI's native route decorator
    async def get_pdf_image(image_key: str):
        logging.info(f"Image request for key: {image_key}")
        try:
            if image_key in page_images:
                image_path = page_images[image_key]
                logging.info(f"Found image path: {image_path}")
                if os.path.exists(image_path):
                    logging.info(f"Image file exists, serving...")
                    return FileResponse(image_path, media_type="image/png")
                else:
                    logging.error(f"Image file does not exist at path: {image_path}")
                    parts = image_key.split('_')
                    if len(parts) >= 2:
                        pdf_name = '_'.join(parts[:-1])
                        page_num = int(parts[-1])
                        pdf_rows = df[df['filename'] == pdf_name]
                        if not pdf_rows.empty:
                            pdf_path = pdf_rows.iloc[0]['full_path']
                            logging.info(f"Found PDF at {pdf_path}, generating image for page {page_num}")
                            try:
                                if os.path.exists(pdf_path):
                                    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150)
                                    if images:
                                        img = images[0]
                                        buffer = BytesIO()
                                        img.save(buffer, format='PNG')
                                        buffer.seek(0)
                                        return Response(content=buffer.getvalue(), media_type="image/png")
                            except Exception as e:
                                logging.error(f"Error generating image on-the-fly: {e}")
                                return Response(content=f"Error generating image: {str(e)}", 
                                            media_type="text/plain", 
                                            status_code=500)
            else:
                logging.error(f"Image key '{image_key}' not found in page_images dictionary")
                return Response(content=f"Image key not found: {image_key}", 
                            media_type="text/plain", 
                            status_code=404)
        except Exception as e:
            logging.error(f"Unexpected error in get_pdf_image: {e}")
            return Response(content=f"Server error: {str(e)}", 
                        media_type="text/plain", 
                        status_code=500)
        
        return Response(content="Image not found", media_type="text/plain", status_code=404)

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, session_id: str, send):
        logging.info(f"WebSocket received - msg: {msg}, session_id: {session_id}")
        if not session_id:
            logging.error("No session_id received in WebSocket connection!")
            return
        messages = await load_chat_history(session_id)
        response_received = asyncio.Event()
        max_tokens = 6000

        async def update_model_status():
            await asyncio.sleep(3)
            if not response_received.is_set():
                for _ in range(25):
                    if response_received.is_set():
                        break
                    await send(Span("🟡", id="model-status-emoji", hx_swap_oob="innerHTML"))
                    await asyncio.sleep(1)
                    if response_received.is_set():
                        break
                    await send(Span("⚫", id="model-status-emoji", hx_swap_oob="innerHTML"))
                    await asyncio.sleep(1)
                else:
                    if not response_received.is_set():
                        await send(Span("🔴", id="model-status-emoji", hx_swap_oob="innerHTML"))
            if response_received.is_set():
                await send(Span("🟢", id="model-status-emoji", hx_swap_oob="innerHTML"))
                await asyncio.sleep(600)
                await send(Span("⚫", id="model-status-emoji", hx_swap_oob="innerHTML"))

        asyncio.create_task(update_model_status())

        messages.append({"role": "user", "content": msg})
        message_index = len(messages) - 1

        new_message = Conversation(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            role='user',
            content=msg
        )
        sqlalchemy_session.add(new_message)
        sqlalchemy_session.commit()

        await send(chat_form(disabled=True))
        await send(Div(chat_message(message_index, messages=messages), id="messages", hx_swap_oob="beforeend"))

        # Document retrieval section - use ColPali if available
        if using_colpali:
            # Process query with ColPali
            logging.info("Using ColPali for query embedding...")
            processed_query = colpali_processor.process_queries([msg]).to(colpali_model.device)
            with torch.no_grad():
                query_embeddings = colpali_model(**processed_query)
            
            # Calculate similarities with all pages
            similarities = []
            for idx, page_emb in enumerate(colpali_embeddings):
                # Convert page embedding to tensor
                page_tensor = torch.tensor(page_emb, device=colpali_model.device, dtype=torch.bfloat16)
                
                # Score using ColPali's scoring method
                score = float(colpali_processor.score_multi_vector(
                    query_embeddings,
                    page_tensor.unsqueeze(0)  # Add batch dimension
                )[0])
                
                similarities.append((idx, score))
            
            # Sort by similarity and get top K=1 (just the most relevant document)
            similarities.sort(key=lambda x: x[1], reverse=True)
            K = 1  # We're simplifying to just show the top result
            
            # Get documents based on top indices
            retrieved_paragraphs = []
            top_sources_data = []
            
            for rank, (idx, score) in enumerate(similarities[:K]):
                if idx < len(df):
                    filename = df.iloc[idx]['filename']
                    page_num = df.iloc[idx]['page']
                    image_key = df.iloc[idx]['image_key']
                    paragraph_text = df.iloc[idx]['text']
                    
                    retrieved_paragraphs.append(paragraph_text)
                    top_sources_data.append({
                        'filename': filename,
                        'page': page_num,
                        'semantic_score': score,
                        'keyword_score': 0.0,  # Not using BM25 anymore
                        'combined_score': score,
                        'image_key': image_key,
                        'idx': idx,
                        'reranker_score': score
                    })
            
            # Use the top result for context
            final_top_sources = top_sources_data[:1]
            context = "\n\n".join(retrieved_paragraphs[:1])
            
        else:
            # Original retrieval logic
            query_embedding = emb_model.encode([msg], normalize_embeddings=True).astype('float32')
            K = 10
            distances, indices = index.search(query_embedding, K)
            tokenized_query = word_tokenize(msg.lower())
            bm25_scores = bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[-K:][::-1]

            all_candidate_indices = list(set(indices[0].tolist() + top_bm25_indices.tolist()))

            retrieved_paragraphs = []
            top_sources_data = []
            docs_for_reranking = []
            semantic_scores = {}
            keyword_scores = {}

            for idx in all_candidate_indices:
                paragraph_text = df.iloc[idx]['text']
                pdf_filename = df.iloc[idx]['filename']
                page_num = df.iloc[idx]['page']
                image_key = df.iloc[idx]['image_key']

                if idx in indices[0]:
                    i = np.where(indices[0] == idx)[0][0]
                    semantic_score = float(1 - distances[0][i])
                    semantic_scores[idx] = semantic_score
                else:
                    semantic_scores[idx] = 0.0

                keyword_score = float(bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0)
                keyword_scores[idx] = keyword_score

                alpha = 0.6
                combined_score = alpha * semantic_scores[idx] + (1 - alpha) * keyword_scores[idx]

                retrieved_paragraphs.append(paragraph_text)
                top_sources_data.append({
                    'filename': pdf_filename,
                    'page': page_num,
                    'semantic_score': semantic_scores[idx],
                    'keyword_score': keyword_scores[idx],
                    'combined_score': combined_score,
                    'image_key': image_key,
                    'idx': idx
                })
                docs_for_reranking.append(paragraph_text)

            ranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2', model_type="cross-encoder", verbose=0)
            ranked_results = ranker.rank(query=msg, docs=docs_for_reranking)
            top_ranked_docs = ranked_results.top_k(3)

            final_retrieved_paragraphs = []
            final_top_sources = []
            for ranked_doc in top_ranked_docs:
                ranked_idx = docs_for_reranking.index(ranked_doc.text)
                final_retrieved_paragraphs.append(ranked_doc.text)
                source_info = top_sources_data[ranked_idx]
                source_info['reranker_score'] = ranked_doc.score
                final_top_sources.append(source_info)

            context = "\n\n".join(retrieved_paragraphs[:2])

        def build_conversation(messages, max_length=2000):
            conversation = ''
            total_length = 0
            for message in reversed(messages):
                role = message['role']
                content = message['content']
                message_text = f"{role.capitalize()}: {content}\n"
                total_length += len(message_text)
                if total_length > max_length:
                    break
                conversation = message_text + conversation
            return conversation

        conversation_history = build_conversation(messages)

        def build_prompt(system_prompt, context, conversation_history):
            return f"""{system_prompt}

Context Information:
{context}

Conversation History:
{conversation_history}
Assistant:"""

        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
            "If you don't know the answer, just say that you don't know."
            "Use three sentences maximum and keep the answer concise."
        )

        prompt = build_prompt(system_prompt, context, conversation_history)
        print(f"Final Prompt being passed to the LLM:\n{prompt}\n")

        vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
        payload = {
            "prompt": prompt,
            "max_tokens": 2000,
            "stream": True
        }

        async with aiohttp.ClientSession() as client_session:
            async with client_session.post(vllm_url, json=payload) as response:
                messages.append({"role": "assistant", "content": ""})
                message_index = len(messages) - 1
                await send(Div(chat_message(message_index, messages=messages), id="messages", hx_swap_oob="beforeend"))

        async with aiohttp.ClientSession() as client_session:
            async with client_session.post(vllm_url, json=payload) as response:
                if response.status == 200:
                    response_received.set()
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            text = chunk.decode('utf-8').strip()
                            if text:
                                if not text.startswith(' ') and messages[message_index]["content"] and not messages[message_index]["content"].endswith(' '):
                                    text = ' ' + text
                                messages[message_index]["content"] += text
                                await send(Span(text, hx_swap_oob="beforeend", id=f"msg-content-{message_index}"))
                    new_assistant_message = Conversation(
                        message_id=str(uuid.uuid4()),
                        session_id=session_id,
                        role='assistant',
                        content=messages[message_index]["content"],
                        top_source_headline=final_top_sources[0]['filename'] if final_top_sources else None,
                        top_source_url=None,
                        cosine_sim_score=final_top_sources[0].get('similarity_score', 0) if final_top_sources else None
                    )
                    sqlalchemy_session.add(new_assistant_message)
                    sqlalchemy_session.commit()
                    logging.info(f"Assistant message committed to DB successfully - Content: {messages[message_index]['content'][:50]}...")
                else:
                    error_message = "Error: Unable to get response from LLM."
                    messages.append({"role": "assistant", "content": error_message})
                    await send(Div(chat_message(len(messages) - 1, messages=messages), id="messages", hx_swap_oob="beforeend"))

        await send(Div(chat_top_sources(final_top_sources), id="top-sources", hx_swap_oob="innerHTML", cls="flex gap-4"))
        await send(chat_form(disabled=False))

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
