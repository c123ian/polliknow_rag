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

# Image definition with environment variables
image = modal.Image.debian_slim(python_version="3.10") \
    .apt_install("libgl1-mesa-glx","libglib2.0-0","libsm6","libxrender1","libxext6","poppler-utils") \
    .pip_install(
        "vllm==0.8.1",
        "mistral_common>=1.5.4",
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
        "colpali-engine[interpretability]>=0.3.2",
        "torch",
        "matplotlib"
    ) \
    .env({"VLLM_USE_V1": "0"})  # Set VLLM_USE_V1=0 to fix Mistral compatibility

app = modal.App(APP_NAME)

# Helper function to convert image to data URI
def format_image(image):
    """Convert PIL Image to data URI for multimodal API"""
    buffered = io.BytesIO()
    # Convert to RGB if it has alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_data_uri = f'data:image/jpeg;base64,{img_base64}'
    return image_data_uri

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=20,
    volumes={
        MISTRAL_MODELS_DIR: mistral_volume,
        COLQWEN_MODELS_DIR: colqwen_volume,
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume
    }
)
@modal.asgi_app()
def serve_vllm():
    import os
    import asyncio
    import fastapi
    import uuid
    import datetime
    import base64
    import io
    from fastapi.responses import StreamingResponse, JSONResponse
    from typing import Optional, List, Dict, Any
    from PIL import Image

    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    # Log environment configuration
    logging.info(f"VLLM_USE_V1 environment variable: {os.environ.get('VLLM_USE_V1', 'not set')}")

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {DEFAULT_MISTRAL_NAME} server",
        description="Multimodal LLM server for insect classification",
        version="0.0.1",
        docs_url="/docs",
    )

    def find_model_path(base_dir):
        for root, _, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    model_path = find_model_path(MISTRAL_MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MISTRAL_MODELS_DIR}")

    # Initialize the LLM
    logging.info(f"Initializing LLM with model path: {model_path}")
    
    try:
        # Initialize with multimodal support
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            trust_remote_code=True,
            max_model_len=4096,
            dtype="float16"
        )
        logging.info("LLM initialized successfully!")
    except Exception as init_error:
        logging.error(f"Error initializing LLM: {init_error}")
        import traceback
        traceback.print_exc()
        raise

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
            
            # Process multimodal messages without stripping images
            logging.info(f"Received chat completion request with {len(messages)} messages")
            
            # Debug log the first message content structure
            if messages and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg.get("content"), list):
                    content_types = [part.get("type") for part in first_msg["content"] if "type" in part]
                    logging.info(f"First message contains content types: {content_types}")
            
            # Use the LLM's chat method for multimodal support
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=["User:", "System:"]
            )
            
            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
            
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                response_text = outputs[0].outputs[0].text
                
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
                logging.error("No output generated from LLM.chat()")
                return JSONResponse(status_code=500, content={"error": "No output generated"})
        except Exception as e:
            logging.error(f"Error in chat_completions: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    # Add completions endpoint for text-only requests
    @web_app.post("/v1/completions")
    async def completions(request: fastapi.Request) -> JSONResponse:
        """Handle basic completions - useful for text-only requests"""
        try:
            body = await request.json()
            prompt = body.get("prompt", "")
            max_tokens = body.get("max_tokens", 500)
            temperature = body.get("temperature", 0.7)
            request_id = str(uuid.uuid4())
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            outputs = llm.generate(prompt, sampling_params)
            
            if outputs and len(outputs) > 0 and len(outputs[0].outputs) > 0:
                response_text = outputs[0].outputs[0].text
                
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
                return JSONResponse(status_code=500, content={"error": "No output generated"})
        except Exception as e:
            logging.error(f"Error in completions: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    # Health check endpoint
    @web_app.get("/health")
    async def health_check():
        """Check if the server is running and the model is loaded"""
        return JSONResponse(content={
            "status": "healthy", 
            "model": DEFAULT_MISTRAL_NAME,
            "supports_multimodal": True,
            "max_tokens": 4096
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

    # Helper function to get context image from PDF pages
    def get_context_image(top_sources):
        """Get the context image from retrieved PDF pages"""
        global page_images
        
        if not top_sources or len(top_sources) == 0:
            return None
            
        # Get the top source document
        top_source = top_sources[0]
        image_key = top_source.get('image_key')
        
        if not image_key or image_key not in page_images:
            logging.warning(f"Context image not found for key: {image_key}")
            
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
                        logging.info(f"Found context image at: {potential_path}")
                        return Image.open(potential_path)
                
            return None
            
        try:
            image_path = page_images[image_key]
            if os.path.exists(image_path):
                context_image = Image.open(image_path)
                return context_image
            else:
                logging.error(f"Context image file not found: {image_path}")
        except Exception as e:
            logging.error(f"Error loading context image: {e}")
        
        return None

    # Updated process_with_mistral function to handle multimodal API properly
    async def process_with_mistral(image, query, context_text="", analysis_id=""):
        """Process the insect image with Mistral, using context image if available"""
        logging.info(f"Processing image {analysis_id} with Mistral using query: {query}")
        
        try:
            # Retrieve relevant documents using ColPali
            retrieved_paragraphs, top_sources = await retrieve_relevant_documents(query)
            
            # Get context image from retrieved documents
            context_image = None
            if top_sources:
                context_image = get_context_image(top_sources)
                if context_image:
                    logging.info(f"Found context image from document: {top_sources[0].get('filename', 'unknown')}, page {top_sources[0].get('page', 'unknown')}")
                else:
                    logging.warning(f"No context image found for {top_sources[0].get('image_key', 'unknown')}")
            
            # Convert main insect image to data URI
            insect_image_uri = format_image(image)
            logging.info(f"Converted insect image to data URI (length: {len(insect_image_uri)})")
            
            # Convert context image to data URI if available
            context_image_uri = None
            if context_image:
                context_image_uri = format_image(context_image)
                logging.info(f"Converted context image to data URI (length: {len(context_image_uri)})")
            
            # Create system prompt that focuses on the insect image
            system_prompt = (
                "You are an expert biologist. Classify the insect in the provided image into "
                "ONE of these categories (bumblebee, honeybee, wasp, solitary bee, hoverfly, other flies, "
                "butterfly & moths, other insect). Focus primarily on the visual characteristics "
                "in the insect image. If a second reference image is provided, use it as secondary information only. "
                "After analysis, provide the answer as a single word, for example 'Answer: honeybee'."
            )
            
            # Create message content based on available images
            message_content = []
            
            # Always add the task description text first
            task_text = "Classify this insect shown in the image."
            if retrieved_paragraphs:
                task_text += " Additional reference information: " + context_text
            
            message_content.append({
                "type": "text", 
                "text": task_text
            })
            
            # Add the main insect image
            message_content.append({
                "type": "image_url",
                "image_url": {"url": insect_image_uri}
            })
            
            # If we have a context image, add it with clear instructions
            if context_image_uri:
                message_content.append({
                    "type": "text",
                    "text": "Here is an additional reference image that may help with classification. Focus primarily on the insect in the first image."
                })
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": context_image_uri}
                })
            
            # Assemble the full message payload
            payload = {
                "model": DEFAULT_MISTRAL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message_content}
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            # Send to the multimodal endpoint
            multimodal_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/chat/completions"
            
            async with aiohttp.ClientSession() as client_session:
                logging.info(f"Sending multimodal request to {multimodal_url}")
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
                        logging.error(f"Error response from multimodal API: {error_text}")
                        return f"Error processing image: {error_text}"
        
        except Exception as e:
            logging.error(f"Exception in process_with_mistral: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

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
            logging.error(f"Error in document retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

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
        categories = ["bumblebee", "honeybee", "wasp", "solitary bee", "hoverfly", "other flies", "butterfly", "moth", "other"]
        
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
            
            # Create a unique modal ID for this result
            modal_id = f"modal_{analysis_id}"
            
            # Create carousel item for the image WITH its classification elements
            carousel_items.append(
                Div(
                    # Image display
                    try_read_image(image_path),
                    
                    # Classification elements directly under each image
                    Div(
                        # Badge and controls section
                        Div(
                            # Classification badge
                            Div(
                                classification,
                                cls=f"badge {badge_color} text-lg p-3 mr-4"
                            ),
                            
                            # Swap component
                            Label(
                                # Hidden checkbox controls the state
                                Input(type="checkbox"),
                                # Swap-on shows when checked (thumbs down)
                                Div("👎", cls="swap-on"),
                                # Swap-off shows when unchecked (thumbs up)
                                Div("👍", cls="swap-off"),
                                cls="swap swap-flip text-3xl mx-2"
                            ),
                            
                            # View raw output modal button
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
                            
                            cls="flex items-center mt-4 mb-2 justify-center"
                        ),
                        
                        # Collapsible context section - only if context exists
                        (Div(
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
                                    try_read_pdf_image(top_sources[0].get('image_key', '')),
                                    cls="mb-4"
                                ) if top_sources else ""),
                                
                                # Context paragraphs if available
                                (Div(
                                    H4("Retrieved Context", cls="text-lg font-semibold text-white mb-2"),
                                    P(context_paragraphs[0] if context_paragraphs else "No context available", 
                                    cls="text-white text-sm bg-zinc-700 p-3 rounded-md"),
                                    cls="mb-4"
                                ) if context_paragraphs else ""),
                                
                                # Token map
                                (Div(
                                    H4("Token Similarity Map", cls="text-lg font-semibold text-white mb-2"),
                                    (Img(
                                        src=f"/heatmap-image/{token_maps.get(top_sources[0].get('image_key', ''), [])[0]['path']}" 
                                            if top_sources and token_maps.get(top_sources[0].get('image_key', ''), []) else "",
                                        cls="w-full max-h-80 object-contain rounded-md"
                                    ) if top_sources and token_maps.get(top_sources[0].get('image_key', ''), []) else 
                                    Div("No token maps available", cls="text-zinc-400 text-center p-4")),
                                    cls="mb-4"
                                ) if top_sources and token_maps else ""),
                                
                                cls="collapse-content text-sm"
                            ),
                            
                            tabindex="0",
                            cls="bg-zinc-800 text-white focus:bg-zinc-700 collapse rounded-md",
                        ) if top_sources else ""),
                        
                        cls="w-full px-4 pb-4"
                    ),
                    
                    id=f"item{i+1}",
                    cls="carousel-item w-full flex flex-col"
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
        
        # Create the complete carousel component
        return Div(
            H2("Insect Classification Results", cls="text-2xl font-bold text-white mb-4 text-center"),
            
            # Main carousel container with images and their classification elements
            Div(
                *carousel_items,
                cls="carousel w-full rounded-lg overflow-hidden mb-4"
            ),
            
            # Carousel indicators
            Div(
                *carousel_indicators,
                cls="flex justify-center w-full gap-2 py-2"
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
            
            # Simplified JavaScript - we no longer need to toggle sections visibility
            Script("""
            document.addEventListener('DOMContentLoaded', function() {
                // Update active indicator on hash change
                const updateActiveIndicator = function() {
                    const id = window.location.hash.substring(1);
                    if (!id) return;
                    
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
                window.addEventListener('hashchange', updateActiveIndicator);
                
                // Initial update - if no hash, set to first item
                if (!window.location.hash) {
                    window.location.hash = 'item1';
                } else {
                    updateActiveIndicator();
                }
            });
            """),
            
            id="batch-results",
            cls="w-full flex flex-col items-center bg-zinc-900 rounded-md p-6 fade-in"
        )

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
                
                # Process with Mistral using multimodal capabilities
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

    # Routes for token maps, image handling, and more...
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
    
    # Modified generate_similarity_maps function to create token similarity maps
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

    # Main route function
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

    # Add batch upload route
    @rt("/batch-upload", methods=["GET"])
    def get_batch_upload(session):
        """Show the batch upload form"""
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            
        logging.info(f"Showing batch upload form for session: {session['session_id']}")
        return batch_upload_form()

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()
