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
MODELS_DIR = "/Qwen"
MODEL_NAME = "Qwen2.5-7B-Instruct-1M"
DATA_DIR = "/bee_pdf"
UPLOADED_PDFS_DIR = "/bee_pdf/uploaded_pdfs"
PDF_IMAGES_DIR = "/bee_pdf/pdf_images"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLPALI_MODEL_NAME = "vidore/colqwen2-v1.0"  # ColQwen2 model name
USERNAME = "c123ian"
APP_NAME = "polliknow-rag"
DATABASE_DIR = "/db_rag_advan"

db_path = os.path.join(DATABASE_DIR, 'chat_history.db')
os.makedirs(DATABASE_DIR, exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''
    DROP TABLE IF EXISTS conversations_history_table_sqlalchemy_v2
''')
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

try:
    volume = modal.Volume.lookup("Qwen", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

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
# already created via embedding_colpali.py
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
    volumes={MODELS_DIR: volume},
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

    MODEL_NAME = "Qwen2.5-7B-Instruct-1M"
    MODELS_DIR = "/Qwen"

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
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

    models = OpenAIServingModels(engine_client=engine, model_config=model_config, base_model_paths={MODEL_NAME: model_path})
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
        DATABASE_DIR: db_volume
    },
    gpu=modal.gpu.A10G(count=1),  # Explicitly request GPU for ColQwen
    secrets=[modal.Secret.from_name("my-custom-secret-3")]
)
@modal.asgi_app()
def serve_fasthtml():
    import faiss
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
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.pyplot as plt
    
    # Helper function to ensure ColQwen model is loaded
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
            print(f"Loading ColQwen2 model ({COLPALI_MODEL_NAME})...")
            try:
                colpali_model = ColQwen2.from_pretrained(
                    COLPALI_MODEL_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
                colpali_processor = ColQwen2Processor.from_pretrained(COLPALI_MODEL_NAME)
                print(f"ColQwen2 model loaded successfully on device: {colpali_model.device}")
                
                # Load embeddings if needed
                if 'colpali_embeddings' not in globals() or colpali_embeddings is None:
                    print(f"Loading ColQwen2 embeddings from {COLPALI_EMBEDDINGS_PATH}")
                    with open(COLPALI_EMBEDDINGS_PATH, "rb") as f:
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
        
        # Hard-code the query for similarity maps
        hardcoded_query = "Classify the species"
        
        return Div(
            H1("Document Visual Analysis", cls="text-3xl font-bold mb-4 text-white"),
            Div(f"Session ID: {session_id}", cls="text-white mb-4"),
            
            # Instead of chat, just show the top image with similarity map
            Div(id="top-source-view", cls="flex flex-col w-full max-w-2xl items-center"),
            
            # Add a button to process the query
            Button(
                "Analyze with query: 'Classify the species'",
                hx_post="/process-hardcoded-query",
                hx_target="#top-source-view",
                cls="bg-green-500 hover:bg-green-600 text-white rounded-md py-2 px-4 mt-4"
            ),
            
            Div(Span("Model status: "), Span("⚫", id="model-status-emoji"), cls="model-status text-white mt-4"),
            cls="flex flex-col items-center min-h-screen bg-black",
        )

    @fasthtml_app.post("/process-hardcoded-query")
    async def process_hardcoded_query():
        # Use the hardcoded query
        query = "Classify the species bumblebee"
        
        # Process query with ColPali/ColQwen
        processed_query = colpali_processor.process_queries([query]).to(colpali_model.device)
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
        
        # Sort by similarity and get top K (just 1 in this case)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_idx, score = similarities[0]
        
        # Get document metadata
        filename = df.iloc[top_idx]['filename']
        page_num = df.iloc[top_idx]['page']
        image_key = df.iloc[top_idx]['image_key']
        
        # Get the image path
        image_path = page_images.get(image_key)
        if not image_path or not os.path.exists(image_path):
            return Div("Image not found", cls="text-white")
        
        # Open the image
        image = Image.open(image_path)
        
        # Process the image with ColPali
        processed_image = colpali_processor.process_images([image]).to(colpali_model.device)
        
        # Forward pass
        with torch.no_grad():
            image_embedding = colpali_model(**processed_image)
        
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
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask
        )[0]  # Get the first item from the batch
        
        # Get tokens for reference
        query_content = colpali_processor.decode(processed_query.input_ids[0]).replace(
            colpali_processor.tokenizer.pad_token, "")
        query_content = query_content.replace(
            getattr(colpali_processor, 'query_augmentation_token', ''), "").strip()
        query_tokens = colpali_processor.tokenizer.tokenize(query_content)
        
        # Find "Classify" token index
        classify_idx = None
        for idx, token in enumerate(query_tokens):
            if 'Classify' in token or 'classify' in token:
                classify_idx = idx
                break
        
        # If not found, just use the first token
        if classify_idx is None:
            classify_idx = 0
            
        # Get the similarity map for the specific token
        token_similarity_map = similarity_maps[classify_idx]
        
        # Generate the visualization
        fig, ax = plot_similarity_map(
            image=image,
            similarity_map=token_similarity_map,
            figsize=(10, 10),
            show_colorbar=True
        )
        
        # Add a title with the token
        token_text = query_tokens[classify_idx]
        max_sim = token_similarity_map.max().item()
        ax.set_title(f"Token: '{token_text}' (MaxSim: {max_sim:.2f})")
        
        # Convert plot to image
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        canvas.print_png(buf)
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        
        # Create the response HTML with the image and its similarity map
        return Div(
            H2(f"Top result: {filename}, page {page_num}", cls="text-xl font-semibold text-white mb-4"),
            Div(f"Similarity score: {score:.4f}", cls="text-green-400 mb-4"),
            
            # Original document image
            Div(
                H3("Document Image:", cls="text-lg font-medium text-white mb-2"),
                Img(src=f"/pdf-image/{image_key}", cls="w-full max-w-xl rounded-lg border border-zinc-700 mb-4"),
                cls="w-full"
            ),
            
            # Similarity map
            Div(
                H3(f"Similarity Map for Token: '{token_text}'", cls="text-lg font-medium text-white mb-2"), 
                Img(src=f"data:image/png;base64,{img_base64}", cls="w-full max-w-xl rounded-lg border border-zinc-700"),
                cls="w-full"
            ),
            
            cls="flex flex-col items-center w-full"
        )

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
        # Keeping this as a placeholder to maintain compatibility
        pass

    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()
    serve_fasthtml()