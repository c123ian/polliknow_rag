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
from starlette.responses import FileResponse, Response as StarletteResponse
from fastapi import FastAPI, Request, Response as FastAPIResponse
from fastapi.responses import FileResponse as FastAPIFileResponse
from starlette.responses import Response
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import json
import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from io import BytesIO

# Constants
MISTRAL_MODELS_DIR = "/Mistral"  # volume for Mistral models
COLQWEN_MODELS_DIR = "/Qwen"  # Changed volume name to match what's created in download_qwen.py
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

# Set up volumes - ensure correct volume names
try:
    mistral_volume = modal.Volume.lookup("Mistral", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download Mistral models first with the appropriate script")

try:
    # Changed to match what's created in download_qwen.py
    colqwen_volume = modal.Volume.lookup("Qwen", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download Qwen models first with download_qwen.py")

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
        "transformers>=4.48.3",      # Use latest transformers
        "rerankers",
        "sqlite-minutils",
        "rank-bm25",
        "nltk",
        "sqlalchemy",
        "pdf2image",
        "colpali-engine[interpretability]>=0.3.2",
        "torch",
        "matplotlib",
        "accelerate",
        "einops",
        "timm",
        "pillow"
    )

app = modal.App(APP_NAME)

# Create app function for Mistral LLM serving & ColQwen2 operations
@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
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
    from typing import Optional, Dict, Any, List

    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels
    from vllm.entrypoints.logger import RequestLogger
    from vllm.sampling_params import SamplingParams
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from colpali_engine.interpretability import get_similarity_maps_from_embeddings, plot_similarity_map

    # Models directories
    MODELS_DIR = MISTRAL_MODELS_DIR
    COLQWEN_DIR = COLQWEN_MODELS_DIR

    # Log directory contents for debugging
    logging.info(f"COLQWEN_DIR contents: {os.listdir(COLQWEN_DIR)}")
    
    # Initialize FastAPI app
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {DEFAULT_MISTRAL_NAME} server with ColQwen2 embedding support",
        description="Run an OpenAI-compatible LLM server with vLLM and ColQwen2",
        version="0.0.1",
        docs_url="/docs",
    )

    # Helper functions for model loading
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
        # For Mistral 3.1, the tokenizer might be in the model directory
        tokenizer_path = model_path

    logging.info(f"Initializing AsyncLLMEngine with model path: {model_path} and tokenizer path: {tokenizer_path}")

    # Following the specific Mistral recommendations from the model card
    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer=tokenizer_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=32768,
        dtype="bfloat16",
        # Mistral-specific parameters
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        # Enable vision model support
        limit_mm_per_prompt={"image": 10},
    )

    # Initialize Mistral engine
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logging.info("Successfully initialized AsyncLLMEngine")
    except Exception as e:
        logging.error(f"Error initializing AsyncLLMEngine: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to direct vLLM approach if the AsyncLLMEngine fails
        try:
            logging.info("Attempting alternative vLLM initialization...")
            from vllm import LLM
            
            # Initialize with simpler parameters
            llm = LLM(
                model=model_path,
                tokenizer=tokenizer_path,
                tokenizer_mode="mistral",
                trust_remote_code=True,
                dtype="bfloat16",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.95,
            )
            
            # Create a compatibility wrapper that mimics AsyncLLMEngine
            class CompatibilityEngine:
                def __init__(self, llm):
                    self.llm = llm
                    
                async def generate(self, prompt, sampling_params, request_id):
                    # Convert to synchronous call
                    outputs = self.llm.generate(prompt, sampling_params)
                    return outputs
                    
            engine = CompatibilityEngine(llm)
            logging.info("Initialized alternative LLM engine")
        except Exception as fallback_error:
            logging.error(f"Failed to initialize alternative engine: {fallback_error}")
            raise

    # Set up model configuration
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
        chat_template_content_format="string",  
    )

    # Initialize ColQwen2 model
    colqwen_model = None
    colqwen_processor = None

    def load_colqwen_model():
        """Load ColQwen2 model if it's not already loaded"""
        global colqwen_model, colqwen_processor
        
        if colqwen_model is not None:
            return colqwen_model, colqwen_processor
            
        # Log directory contents for debugging
        try:
            model_dir = os.path.join(COLQWEN_DIR, os.path.basename(DEFAULT_COLQWEN_NAME))
            logging.info(f"Looking for ColQwen2 model in: {model_dir}")
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                logging.info(f"Directory exists and contains: {os.listdir(model_dir)}")
                colqwen_model = ColQwen2.from_pretrained(
                    model_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
                colqwen_processor = ColQwen2Processor.from_pretrained(model_dir)
                logging.info(f"Loaded ColQwen2 model from volume: {model_dir}")
            else:
                logging.info(f"Directory not found, downloading from HuggingFace: {DEFAULT_COLQWEN_NAME}")
                colqwen_model = ColQwen2.from_pretrained(
                    DEFAULT_COLQWEN_NAME,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
                colqwen_processor = ColQwen2Processor.from_pretrained(DEFAULT_COLQWEN_NAME)
                logging.info(f"Loaded ColQwen2 model from HuggingFace")
        except Exception as e:
            logging.error(f"Error loading ColQwen2 model: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        return colqwen_model, colqwen_processor
    
    # === Mistral Vision/LLM Endpoints ===
    
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
                temperature=0.15,  # Lower for Mistral
                max_tokens=max_tokens,
                stop=["User:", "Assistant:", "\n\n"],
            )

            if stream:
                async def generate_text():
                    # Simplified streaming implementation that properly uses async iteration
                    response_text = ""
                    
                    # Use proper async iteration over the generator
                    async for output in engine.generate(prompt, sampling_params, request_id):
                        if len(output.outputs) > 0:
                            new_text = output.outputs[0].text
                            
                            # Remove Assistant: prefix if present
                            if "Assistant:" in new_text and not response_text:
                                new_text = new_text.split("Assistant:")[-1].lstrip()
                                
                            # Calculate the difference to send
                            delta = new_text[len(response_text):]
                            if delta:
                                yield delta
                                response_text = new_text

                return StreamingResponse(generate_text(), media_type="text/plain")
            else:
                # Non-streaming response using proper async handling
                response_text = ""
                async for output in engine.generate(prompt, sampling_params, request_id):
                    if len(output.outputs) > 0:
                        response_text = output.outputs[0].text
                
                # Remove Assistant: prefix if present
                if "Assistant:" in response_text:
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
                
        except Exception as e:
            logging.error(f"Error in completion_generator: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.post("/v1/chat/completions")
    async def chat_completions(request: fastapi.Request) -> JSONResponse:
        """Handle multimodal chat completions for Mistral Vision"""
        try:
            body = await request.json()
            messages = body.get("messages", [])
            model = body.get("model", DEFAULT_MISTRAL_NAME)
            max_tokens = body.get("max_tokens", 500)
            temperature = body.get("temperature", 0.15)  # Mistral recommends 0.15
            top_p = body.get("top_p", 0.95)
            request_id = str(uuid.uuid4())
            
            logging.info(f"Received chat completion request for model {model}")
            
            # Format messages according to Mistral's expected format
            formatted_messages = []
            for message in messages:
                if message.get("role") == "system":
                    # For Mistral 3.1, add system prompt using special tokens
                    system_content = message.get("content", "")
                    if isinstance(system_content, list):
                        # Handle multimodal system prompt if needed
                        system_text = " ".join([item.get("text", "") for item in system_content if item.get("type") == "text"])
                    else:
                        system_text = system_content
                    
                    # Add formatted system prompt
                    formatted_messages.append({
                        "role": "system",
                        "content": system_text
                    })
                else:
                    # Pass user and assistant messages as is
                    formatted_messages.append(message)
            
            # Build sampling params
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=body.get("stop", None)
            )
            
            # Try Direct OpenAI API Compatible Format first
            try:
                response = await openai_serving_chat.create_chat_completion(
                    request_id=request_id,
                    model=model,
                    messages=formatted_messages,
                    sampling_params=sampling_params,
                    stream=False
                )
                return JSONResponse(content=response)
            except Exception as api_error:
                logging.error(f"Error in OpenAI chat completion: {api_error}")
                import traceback
                traceback.print_exc()
                
                # Fall back to manual prompt construction
                logging.info("Falling back to manual prompt construction")
                
                # Build a prompt in Mistral's expected format
                # Format: <s>[SYSTEM_PROMPT]<system prompt>[/SYSTEM_PROMPT][INST]<user message>[/INST]<assistant response></s>[INST]<user message>[/INST]
                
                system_content = ""
                for message in messages:
                    if message.get("role") == "system":
                        content = message.get("content", "")
                        if isinstance(content, list):
                            # Get only text from system prompt
                            system_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                        else:
                            system_content = content
                        break
                
                # Build the conversation prompt with Mistral's format
                prompt = f"<s>[SYSTEM_PROMPT]{system_content}[/SYSTEM_PROMPT]"
                
                # Add conversation messages
                user_content = None
                for message in messages:
                    role = message.get("role")
                    content = message.get("content")
                    
                    # Skip system messages as they're already handled
                    if role == "system":
                        continue
                        
                    if role == "user":
                        # For the last user message, save it for special handling
                        user_content = content
                        continue
                    elif role == "assistant":
                        if isinstance(content, list):
                            # Get only text from assistant
                            text_content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
                            prompt += f"[INST]{user_content if user_content else ''}[/INST]{text_content}</s>"
                        else:
                            prompt += f"[INST]{user_content if user_content else ''}[/INST]{content}</s>"
                        user_content = None
                
                # Add the final user message if there is one
                if user_content is not None:
                    if isinstance(user_content, list):
                        # Handle multimodal content specially
                        text_parts = []
                        has_images = False
                        
                        for item in user_content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image_url":
                                has_images = True
                                # Just note that there's an image - we'll add it to the model inputs directly
                                text_parts.append("[IMAGE]")
                        
                        text_content = " ".join(text_parts)
                        prompt += f"[INST]{text_content}[/INST]"
                        
                        if has_images:
                            logging.warning("Direct image handling not supported in manual mode - using text-only prompt")
                    else:
                        prompt += f"[INST]{user_content}[/INST]"
                
                # Use proper async handling with the generator
                response_text = ""
                async for output in engine.generate(prompt, sampling_params, request_id):
                    if output and hasattr(output, 'outputs') and len(output.outputs) > 0:
                        response_text = output.outputs[0].text.strip()
                
                # Create OpenAI-compatible response
                response_json = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(datetime.datetime.now().timestamp()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": -1,  # Not available
                        "completion_tokens": -1,  # Not available
                        "total_tokens": -1  # Not available
                    }
                }
                
                return JSONResponse(content=response_json)
                
        except Exception as e:
            logging.error(f"Error in chat_completions: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    # === ColQwen2 Endpoints ===
    
    @web_app.post("/v1/embeddings")
    async def generate_embeddings(request: fastapi.Request) -> JSONResponse:
        """Generate embeddings for queries or images"""
        try:
            # Load the model if needed
            model, processor = load_colqwen_model()
            
            body = await request.json()
            embedding_type = body.get("type", "query")  # 'query' or 'image'
            
            if embedding_type == "query":
                # Process queries
                queries = body.get("queries", [])
                if not queries:
                    return JSONResponse(status_code=400, content={"error": "No queries provided"})
                
                processed_queries = processor.process_queries(queries).to(model.device)
                with torch.no_grad():
                    query_embeddings = model(**processed_queries)
                
                # Convert embeddings to list for JSON serialization
                result = []
                for i, emb in enumerate(query_embeddings):
                    # Convert to CPU and standard float for serialization
                    numpy_emb = emb.cpu().to(torch.float32).numpy().tolist()
                    result.append({
                        "index": i,
                        "embedding": numpy_emb,
                        "query": queries[i]
                    })
                
                return JSONResponse(content={"embeddings": result})
                
            elif embedding_type == "image":
                # Process images from base64
                images_data = body.get("images", [])
                if not images_data:
                    return JSONResponse(status_code=400, content={"error": "No images provided"})
                
                # Convert base64 to PIL Images
                images = []
                for img_data in images_data:
                    try:
                        # Expect format like: data:image/jpeg;base64,/9j/4AAQSkZJR...
                        if "base64," in img_data:
                            img_data = img_data.split("base64,")[1]
                        
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(BytesIO(img_bytes))
                        images.append(img)
                    except Exception as e:
                        logging.error(f"Error processing image: {e}")
                        return JSONResponse(status_code=400, content={"error": f"Invalid image data: {str(e)}"})
                
                # Process images with ColQwen2
                processed_images = processor.process_images(images).to(model.device)
                with torch.no_grad():
                    image_embeddings = model(**processed_images)
                
                # Convert embeddings to list for JSON serialization
                result = []
                for i, emb in enumerate(image_embeddings):
                    # Convert to CPU and standard float for serialization
                    numpy_emb = emb.cpu().to(torch.float32).numpy().tolist()
                    result.append({
                        "index": i,
                        "embedding": numpy_emb
                    })
                
                return JSONResponse(content={"embeddings": result})
                
            else:
                return JSONResponse(status_code=400, content={"error": f"Invalid embedding type: {embedding_type}"})
                
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.post("/v1/score_embeddings")
    async def score_embeddings(request: fastapi.Request) -> JSONResponse:
        """Score similarity between query and document embeddings"""
        try:
            # Load the model if needed
            _, processor = load_colqwen_model()
            
            body = await request.json()
            query_embeddings_data = body.get("query_embeddings", [])
            image_embeddings_data = body.get("image_embeddings", [])
            
            if not query_embeddings_data or not image_embeddings_data:
                return JSONResponse(status_code=400, content={"error": "Both query and image embeddings must be provided"})
            
            # Convert lists back to tensors
            query_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in query_embeddings_data]
            image_embeddings = [torch.tensor(emb, dtype=torch.float32) for emb in image_embeddings_data]
            
            # Calculate scores
            scores = []
            for q_idx, query_emb in enumerate(query_embeddings):
                query_scores = []
                for img_idx, img_emb in enumerate(image_embeddings):
                    # Use processor's scoring method
                    score = float(processor.score_multi_vector(
                        query_emb.unsqueeze(0),
                        img_emb.unsqueeze(0)
                    )[0])
                    query_scores.append({
                        "image_index": img_idx,
                        "score": score
                    })
                scores.append({
                    "query_index": q_idx,
                    "scores": query_scores
                })
            
            return JSONResponse(content={"scores": scores})
            
        except Exception as e:
            logging.error(f"Error scoring embeddings: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    @web_app.post("/v1/similarity_maps")
    async def generate_similarity_maps(request: fastapi.Request) -> JSONResponse:
        """Generate token similarity heatmaps for images and queries"""
        try:
            # Load the model if needed
            model, processor = load_colqwen_model()
            
            body = await request.json()
            query = body.get("query", "")
            image_data = body.get("image", "")
            max_tokens = body.get("max_tokens", 6)  # Limit number of tokens for maps
            
            if not query or not image_data:
                return JSONResponse(status_code=400, content={"error": "Both query and image must be provided"})
            
            # Convert base64 to PIL Image
            try:
                # Expect format like: data:image/jpeg;base64,/9j/4AAQSkZJR...
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                
                img_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(img_bytes))
            except Exception as e:
                logging.error(f"Error processing image data: {e}")
                return JSONResponse(status_code=400, content={"error": f"Invalid image data: {str(e)}"})
            
            # Process query and image with ColQwen2
            processed_query = processor.process_queries([query]).to(model.device)
            batch_images = processor.process_images([image]).to(model.device)
            
            # Forward passes to get embeddings
            with torch.no_grad():
                query_embeddings = model(**processed_query)
                image_embeddings = model(**batch_images)
            
            # Get the number of image patches
            n_patches = processor.get_n_patches(
                image_size=image.size,
                patch_size=model.patch_size,
                spatial_merge_size=getattr(model, 'spatial_merge_size', None)
            )
            
            # Get image mask
            image_mask = processor.get_image_mask(batch_images)
            
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
            query_tokens = processor.tokenizer.tokenize(query)
            
            # Filter to meaningful tokens
            token_sims = []
            stopwords = set(["<bos>", "<eos>", "<pad>", "a", "an", "the", "in", "on", "at", "of", "for", "with", "by", "to", "from"])
            
            for token_idx, token in enumerate(query_tokens):
                if token_idx >= similarity_maps.shape[0]:
                    continue
                    
                # Skip stopwords and short tokens
                if token in stopwords or len(token) <= 1:
                    continue
                    
                token_clean = token[1:] if token.startswith("▁") else token
                if token_clean and len(token_clean) > 1:
                    max_sim = similarity_maps[token_idx].max().item()
                    token_sims.append((token_idx, token, max_sim))
            
            # Sort by similarity score and take top tokens
            token_sims.sort(key=lambda x: x[2], reverse=True)
            top_tokens = token_sims[:max_tokens]  # Limit to max_tokens
            
            # Generate and return heatmaps as base64
            token_maps = []
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
                token_display = token[1:] if token.startswith("▁") else token
                ax.set_title(f"Token: '{token_display}', Score: {score:.2f}", fontsize=12)
                
                # Convert figure to base64
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png", bbox_inches='tight')
                img_buffer.seek(0)
                img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                
                # Add to response
                token_maps.append({
                    "token": token_display,
                    "token_idx": int(token_idx),
                    "score": float(score),
                    "heatmap_base64": img_data
                })
            
            return JSONResponse(content={
                "query": query,
                "token_maps": token_maps
            })
            
        except Exception as e:
            logging.error(f"Error generating similarity maps: {str(e)}")
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e)})

    return web_app

# Create app function for FastHTML UI - No GPU required now
@app.function(
    image=image,
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    volumes={
        DATA_DIR: bee_volume,
        DATABASE_DIR: db_volume
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
    
    # Global variables - data only, no models
    df = None
    page_images = {}
    bm25_index = None
    tokenized_docs = None
    colpali_embeddings = None
    
    # Function to load data
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

    # Load data at startup
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
                
                /* Prevent token map UI flicker */
                .token-tab {
                    min-width: 60px;
                    text-align: center;
                }
                
                #token-map-display {
                    min-height: 300px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                #token-map-display img {
                    transition: opacity 0.2s;
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

    # API calls to serve_vllm for embeddings and similarity maps
    async def call_colqwen_api(endpoint, payload):
        """Call ColQwen2 API endpoints in serve_vllm"""
        api_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/{endpoint}"
        logging.info(f"Calling ColQwen2 API endpoint: {api_url}")
        
        async with aiohttp.ClientSession() as client_session:
            try:
                async with client_session.post(api_url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logging.error(f"Error from ColQwen2 API: {error_text}")
                        return {"error": error_text}
            except Exception as e:
                logging.error(f"Error calling ColQwen2 API: {str(e)}")
                import traceback
                traceback.print_exc()
                return {"error": str(e)}

    # Retrieve relevant documents
    async def retrieve_relevant_documents(query, top_k=5):
        """Retrieve most relevant documents using ColPali embeddings and BM25"""
        global colpali_embeddings, df, bm25_index, tokenized_docs
        
        if colpali_embeddings is None or df is None or len(df) == 0:
            logging.error("No documents or embeddings available for retrieval")
            return [], []
            
        retrieved_paragraphs = []
        top_sources_data = []
        
        # ColPali retrieval (vector search)
        try:
            # Generate query embeddings via API
            query_results = await call_colqwen_api("v1/embeddings", {
                "type": "query",
                "queries": [query]
            })
            
            if "error" in query_results:
                logging.error(f"Error generating query embeddings: {query_results['error']}")
                return [], []
            
            # Extract the query embedding
            query_embedding = query_results["embeddings"][0]["embedding"]
            
            # Calculate similarities with all pages
            similarities = []
            for idx, page_emb in enumerate(colpali_embeddings):
                # Convert embeddings to proper format
                payload = {
                    "query_embeddings": [query_embedding],
                    "image_embeddings": [page_emb.tolist() if hasattr(page_emb, "tolist") else page_emb]
                }
                
                # Call API to score embeddings
                score_results = await call_colqwen_api("v1/score_embeddings", payload)
                
                if "error" in score_results:
                    logging.error(f"Error scoring embeddings: {score_results['error']}")
                    continue
                
                # Extract the score
                score = score_results["scores"][0]["scores"][0]["score"]
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

    # Generate similarity maps for context document
    async def generate_similarity_maps(query, image_key):
        """Generate token similarity maps for a retrieved document"""
        global page_images
        
        if not image_key or image_key not in page_images:
            logging.error(f"Invalid image key: {image_key}")
            return []
            
        try:
            # Get image path and load image
            image_path = page_images[image_key]
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                return []
                
            # Load the image
            image = Image.open(image_path)
            
            # Convert image to base64
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            base64_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Call API to generate similarity maps
            result = await call_colqwen_api("v1/similarity_maps", {
                "query": query,
                "image": f"data:image/png;base64,{base64_img}",
                "max_tokens": 6
            })
            
            if "error" in result:
                logging.error(f"Error generating similarity maps: {result['error']}")
                return []
            
            # Process and save the maps locally
            token_maps = result.get("token_maps", [])
            image_heatmaps = []
            
            for token_map in token_maps:
                token = token_map["token"]
                token_idx = token_map["token_idx"]
                score = token_map["score"]
                
                # Save base64 image to file
                heatmap_filename = f"{image_key}_token_{token_idx}.png"
                heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
                
                # Decode and save
                heatmap_data = base64.b64decode(token_map["heatmap_base64"])
                with open(heatmap_path, "wb") as f:
                    f.write(heatmap_data)
                
                # Add to list
                image_heatmaps.append({
                    "token": token,
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

    # Process image with Mistral Vision - New implementation without ColQwen image descriptions
    async def process_with_mistral_vision(image, query, context_text="", analysis_id=""):
        """Process the image with Mistral Vision including context from retrieved documents"""
        logging.info(f"Processing image {analysis_id} with Mistral Vision using query: {query}")
        
        try:
            # Convert image to base64 for transmission
            img_buffer = BytesIO()
            image.save(img_buffer, format=image.format or "JPEG")
            img_data = img_buffer.getvalue()
            base64_img = base64.b64encode(img_data).decode("utf-8")
            
            # Build the system prompt for Mistral
            system_prompt = """You are an expert biologist analyzing visual specimens. 
            Provide detailed, scientific analysis with proper taxonomic terminology.
            Always be thorough but concise, keeping responses under 5 sentences."""
            
            # Log image info for debugging
            logging.info(f"Image for analysis: {image.format} {image.size}, {len(base64_img)} bytes base64")
            
            # Build user content with text and image
            user_content = []
            
            # Add text content
            text_content = query
            if context_text:
                text_content += f"\n\nRelevant context from scientific literature:\n{context_text}\n\nExamine the image and provide a detailed, scientifically accurate response using the context where relevant."
            
            user_content.append({"type": "text", "text": text_content})
            
            # Add image content - Mistral expects this format
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
            
            # Create the chat completion request in Mistral's format
            multimodal_message = {
                "model": DEFAULT_MISTRAL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 500,
                "temperature": 0.15  # Mistral recommends low temperature
            }
            
            # Call the multimodal endpoint
            multimodal_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/chat/completions"
            logging.info(f"Calling Mistral Vision endpoint: {multimodal_url}")
            
            async with aiohttp.ClientSession() as client_session:
                try:
                    async with client_session.post(multimodal_url, json=multimodal_message, timeout=120) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                multimodal_response = result["choices"][0]["message"]["content"]
                                logging.info(f"Received Mistral Vision response: {multimodal_response[:100]}...")
                                return multimodal_response.strip()
                        
                        # If we get here, there was an error with the response
                        logging.warning(f"Mistral Vision endpoint failed with status {response.status}")
                        error_text = await response.text()
                        logging.error(f"Error response: {error_text}")
                except Exception as e:
                    logging.warning(f"Mistral Vision endpoint error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Fallback to text-only endpoint with clear messaging
            logging.info("Falling back to text-only endpoint")
            
            # Format the text prompt with Mistral's expected format
            system_part = f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT]"
            
            # Build prompt without image description - just explain the limitation
            user_part = f"Query: {query}\n\n"
            if context_text:
                user_part += f"Relevant context from scientific literature:\n{context_text}\n\n"
            user_part += "Note: I couldn't analyze the image directly, but I can provide information based on the query and context."
            
            # Complete the prompt in Mistral format
            text_prompt = f"<s>{system_part}[INST]{user_part}[/INST]"
            
            # Use text-only API
            vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
            payload = {
                "prompt": text_prompt,
                "max_tokens": 500,
                "temperature": 0.15,
                "stream": False
            }
            
            logging.info(f"Sending text request to endpoint: {vllm_url}")
            
            async with aiohttp.ClientSession() as client_session:
                async with client_session.post(vllm_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            text_response = result["choices"][0]["text"]
                            logging.info(f"Received text response: {text_response[:100]}...")
                            return text_response.strip()
                        else:
                            logging.error(f"Invalid response format: {result}")
                    else:
                        error_text = await response.text()
                        logging.error(f"Error from Mistral text API: {error_text}")
                        return f"Error processing image: {error_text}"
        
        except Exception as e:
            logging.error(f"Exception in process_with_mistral_vision: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error analyzing image: {str(e)}"
    
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
                    cls="mx-auto max-h-64 rounded-lg border border-zinc-700"
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
                                cls="mx-auto max-h-64 rounded-lg border border-zinc-700"
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

    def token_maps_ui(image_key, heatmaps):
        """Create token map UI with tabs and display area - improved to avoid flicker"""
        if not heatmaps:
            return Div("No token maps available", cls="text-zinc-400 text-center")
            
        # Create a list of button elements first
        tab_buttons = []
        
        # Create the tabs
        for i, heatmap in enumerate(heatmaps):
            token_class = "btn btn-sm token-tab " + ("btn-primary active" if i == 0 else "btn-outline")
            token_tab = Button(
                heatmap['token'],
                cls=token_class,
                # Use data attributes for client-side tab switching to reduce HTTP requests
                data_token_idx=f"{heatmap['token_idx']}",
                data_image_key=f"{image_key}",
                id=f"token-tab-{i}"
            )
            # Add the button to our list
            tab_buttons.append(token_tab)
        
        # Create token tabs Div with all buttons at once
        token_tabs = Div(*tab_buttons, cls="flex flex-wrap gap-2 mb-4")
        
        # Pre-load all token maps as hidden images
        token_maps_container = Div(id="token-maps-container", cls="hidden")
        for heatmap in heatmaps:
            heatmap_path = f"/heatmap-image/{image_key}_token_{heatmap['token_idx']}.png"
            token_maps_container.children.append(
                Img(
                    src=heatmap_path,
                    cls="hidden",
                    id=f"token-map-{heatmap['token_idx']}",
                    data_token_idx=f"{heatmap['token_idx']}"
                )
            )
        
        # Initial token map display - show the first one
        initial_token_map = None
        if heatmaps:
            initial_token_idx = heatmaps[0]['token_idx']
            initial_heatmap_path = f"/heatmap-image/{image_key}_token_{initial_token_idx}.png"
            initial_token_map = Img(
                src=initial_heatmap_path,
                cls="mx-auto max-h-96 w-full object-contain token-map",
                id="active-token-map"
            )
        else:
            initial_token_map = Div("Select a token above to see its heatmap", cls="text-zinc-400 text-center py-8")
        
        # Create the full token maps container
        return Div(
            H3("Token Similarity Maps", cls="text-xl font-semibold text-white mb-2"),
            P("See how the model focuses on different parts of the context document for each word in the query:", 
              cls="text-zinc-300 mb-3"),
            
            # Token tabs navigation - using our complete Div with all buttons
            token_tabs,
            
            # Hidden container with all maps
            token_maps_container,
            
            # Token map display area - just shows the active map
            Div(
                initial_token_map,
                id="token-map-display",
                cls="w-full bg-zinc-700 rounded-md overflow-hidden"
            ),
            
            # JavaScript to handle tab interaction client-side
            Script('''
            document.addEventListener('DOMContentLoaded', function() {
                // Add click handlers to all token tabs
                document.querySelectorAll('[id^="token-tab-"]').forEach(function(tab) {
                    tab.addEventListener('click', function(e) {
                        // Get the token index and image key from data attributes
                        const tokenIdx = this.getAttribute('data-token-idx');
                        const imageKey = this.getAttribute('data-image-key');
                        
                        // Remove active class from all tabs
                        document.querySelectorAll('[id^="token-tab-"]').forEach(function(t) {
                            t.classList.remove('btn-primary', 'active');
                            t.classList.add('btn-outline');
                        });
                        
                        // Add active class to clicked tab
                        this.classList.remove('btn-outline');
                        this.classList.add('btn-primary', 'active');
                        
                        // Update the displayed token map
                        const mapPath = `/heatmap-image/${imageKey}_token_${tokenIdx}.png`;
                        const activeMap = document.getElementById('active-token-map');
                        if (activeMap) {
                            // Just update the src attribute to avoid reflow
                            activeMap.src = mapPath;
                        }
                    });
                });
                
                // Auto-select first tab on load
                setTimeout(function() {
                    const firstTab = document.getElementById('token-tab-0');
                    if (firstTab) firstTab.click();
                }, 100);
            });
            '''),
            
            cls="mt-6"
        )

    def context_document_ui(top_source):
        """Create UI for displaying the context document"""
        if not top_source:
            return Div("No context document available", cls="text-zinc-400 text-center")
            
        # Get document info
        image_key = top_source['image_key']
        score = top_source.get('score', 0.0)
        
        # Create the document display
        return Div(
            H3("Context Document", cls="text-xl font-semibold text-white mb-2"),
            P(f"The AI is using this document to help analyze your image (Score: {score:.2f}):", 
              cls="text-zinc-300 mb-3"),
            
            # Document image - embed directly as base64
            Div(
                try_read_pdf_image(image_key),
                cls="w-full mb-4"
            ),
            
            cls="context-container mt-6"
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

    def analysis_results_ui(image_path, mistral_response, context_paragraphs, top_sources, token_maps, analysis_id):
        """Create UI to display analysis results with context and token maps"""
        
        # Get the image filename for the UI
        image_filename = os.path.basename(image_path)
        
        # Check if the file exists, log info about it
        if os.path.exists(image_path):
            logging.info(f"Image exists for UI: {image_path}, size: {os.path.getsize(image_path)} bytes")
        else:
            logging.error(f"Image does not exist for UI: {image_path}")
        
        # Context section
        context_ui = None
        if top_sources and context_paragraphs:
            top_source = top_sources[0]
            context_text = context_paragraphs[0] if context_paragraphs else ""
            
            context_ui = Div(
                H3("Retrieved Context", cls="text-lg font-semibold text-white mb-2"),
                
                # Context container
                Div(
                    Div("The AI used this information to enhance its analysis:", cls="context-header"),
                    P(context_text, cls="text-white text-sm"),
                    cls="context-container"
                ),
                
                # Context document display
                context_document_ui(top_source),
                
                cls="mb-6"
            )
        
        # Token maps section - with improved error handling
        token_maps_section = None
        if top_sources and token_maps and len(top_sources) > 0:
            try:
                top_source = top_sources[0]
                image_key = top_source.get('image_key')
                
                if image_key and image_key in token_maps and token_maps[image_key]:
                    token_maps_section = token_maps_ui(image_key, token_maps[image_key])
            except Exception as e:
                logging.error(f"Error rendering token maps: {e}")
                token_maps_section = Div(
                    H3("Token Similarity Maps", cls="text-xl font-semibold text-white mb-2"),
                    P(f"Error displaying token maps: {str(e)}", cls="text-red-500"),
                    cls="bg-zinc-800 p-4 rounded-md"
                )
        
        if token_maps_section is None:
            token_maps_section = Div(
                H3("Token Similarity Maps", cls="text-xl font-semibold text-white mb-2"),
                P("No token maps available for this analysis.", cls="text-zinc-400"),
                cls="bg-zinc-800 p-4 rounded-md"
            )
        
        return Div(
            H2("Analysis Results", cls="text-2xl font-bold text-white mb-4"),
            
            # Results layout - side by side for larger screens
            Div(
                # Left side - image & analysis text
                Div(
                    # Display the original image - Read the file directly instead of using a URL
                    Div(
                        H3("Uploaded Image", cls="text-xl font-semibold text-white mb-2"),
                        try_read_image(image_path),  # Use our custom function to get the image
                        cls="mb-6"
                    ),
                    
                    # Display Mistral's response
                    Div(
                        H3("AI Analysis", cls="text-xl font-semibold text-white mb-2"),
                        Div(
                            mistral_response,
                            cls="bg-zinc-700 rounded-md p-4 text-white"
                        ),
                        cls="mb-6"
                    ),
                    
                    # Display context if available
                    context_ui if context_ui else "",
                    
                    cls="lg:w-1/2 lg:pr-4 mb-6 lg:mb-0"
                ),
                
                # Right side - token maps (with error handling)
                Div(
                    token_maps_section,
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
                // Set up form behaviors
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
                
                // Handle token map interactions
                document.addEventListener('click', function(e) {
                    if (e.target.closest('[id^="token-tab-"]')) {
                        const tab = e.target.closest('[id^="token-tab-"]');
                        const tokenIdx = tab.getAttribute('data-token-idx');
                        const imageKey = tab.getAttribute('data-image-key');
                        
                        // Update tab styling
                        document.querySelectorAll('[id^="token-tab-"]').forEach(t => {
                            t.classList.remove('btn-primary', 'active');
                            t.classList.add('btn-outline');
                        });
                        
                        tab.classList.remove('btn-outline');
                        tab.classList.add('btn-primary', 'active');
                        
                        // Find the right image and show it
                        const mapPath = `/heatmap-image/${imageKey}_token_${tokenIdx}.png`;
                        const display = document.getElementById('token-map-display');
                        
                        if (display) {
                            // Look for an existing active map
                            let activeMap = document.getElementById('active-token-map');
                            
                            if (activeMap) {
                                activeMap.src = mapPath;
                            } else {
                                // Create a new map if needed
                                activeMap = document.createElement('img');
                                activeMap.id = 'active-token-map';
                                activeMap.src = mapPath;
                                activeMap.className = 'mx-auto max-h-96 w-full object-contain token-map';
                                
                                // Clear the display and add the new map
                                display.innerHTML = '';
                                display.appendChild(activeMap);
                            }
                        }
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
        # Make sure we have a clean filename with no spaces or special characters
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        if safe_filename != filename:
            logging.info(f"Sanitized filename from '{filename}' to '{safe_filename}'")
            filename = safe_filename
        
        image_path = os.path.join(TEMP_UPLOAD_DIR, f"{analysis_id}_{filename}")
        
        # Save the uploaded image
        logging.info(f"Saving image to {image_path}")
        try:
            # Read the content of the uploaded file
            content = await image_file.read()
            with open(image_path, "wb") as f:
                f.write(content)
            
            # Verify the file was saved
            if os.path.exists(image_path):
                file_size = os.path.getsize(image_path)
                logging.info(f"Image saved successfully: {image_path} (size: {file_size} bytes)")
            else:
                logging.error(f"Failed to save image, file doesn't exist: {image_path}")
                return Div(f"Error saving image: File doesn't exist after write operation", cls="text-red-500")
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()
            return Div(f"Error saving image: {str(e)}", cls="text-red-500")
        
        # Open the image for processing
        try:
            image = Image.open(image_path)
            logging.info(f"Image opened: {image.size}, {image.mode}")
        except Exception as e:
            logging.error(f"Error opening image: {e}")
            import traceback
            traceback.print_exc()
            return Div(f"Error opening image: {str(e)}", cls="text-red-500")
        
        # Get the appropriate query based on analysis type
        query = get_query_for_analysis(analysis_type)
        logging.info(f"Using query: {query}")
        
        # Save analysis to database
        try:
            # Use direct SQLite for more reliable database operations
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO image_analyses (analysis_id, image_path, analysis_type, query) VALUES (?, ?, ?, ?)",
                (analysis_id, image_path, analysis_type, query)
            )
            conn.commit()
            conn.close()
            logging.info(f"Analysis record created in database: {analysis_id}")
        except Exception as db_error:
            logging.error(f"Database error: {db_error}")
            import traceback
            traceback.print_exc()
        
        # Retrieve relevant documents for context
        retrieved_paragraphs, top_sources = await retrieve_relevant_documents(query)
        
        if not top_sources:
            logging.warning("No relevant documents found for context")
        else:
            logging.info(f"Found {len(top_sources)} relevant documents, top score: {top_sources[0].get('score', 0)}")
        
        # Generate context text
        context_text = "\n\n".join(retrieved_paragraphs) if retrieved_paragraphs else ""
        
        # Generate similarity maps for top document
        token_maps = {}
        if top_sources:
            top_source = top_sources[0]
            image_key = top_source.get('image_key')
            
            if image_key:
                logging.info(f"Generating similarity maps for image key: {image_key}")
                
                # Update database with context source if column exists
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    # Check if the column exists first
                    cursor.execute("PRAGMA table_info(image_analyses)")
                    columns = [column[1] for column in cursor.fetchall()]
                    if 'context_source' in columns:
                        cursor.execute(
                            "UPDATE image_analyses SET context_source = ? WHERE analysis_id = ?",
                            (f"{top_source['filename']} (page {top_source['page']})", analysis_id)
                        )
                        conn.commit()
                    conn.close()
                except Exception as db_error:
                    logging.error(f"Database error updating context source: {db_error}")
                
                # Generate token maps
                image_heatmaps = await generate_similarity_maps(query, image_key)
                if image_heatmaps:
                    logging.info(f"Generated {len(image_heatmaps)} token heatmaps")
                    token_maps[image_key] = image_heatmaps
                else:
                    logging.warning(f"No token heatmaps generated for {image_key}")
            else:
                logging.warning("Top source missing image_key")
        
        # Process with Mistral Vision (renamed the function)
        mistral_response = await process_with_mistral_vision(image, query, context_text, analysis_id)
        
        # Update database with response
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE image_analyses SET response = ? WHERE analysis_id = ?",
                (mistral_response, analysis_id)
            )
            conn.commit()
            conn.close()
            logging.info(f"Updated analysis record with response: {analysis_id}")
        except Exception as db_error:
            logging.error(f"Database error updating response: {db_error}")
        
        # Add a log just before returning UI
        logging.info(f"Rendering UI with: image={os.path.basename(image_path)}, token_maps={len(token_maps)}, top_sources={len(top_sources)}")
        
        # Return the analysis results UI
        return analysis_results_ui(
            image_path=image_path, 
            mistral_response=mistral_response, 
            context_paragraphs=retrieved_paragraphs,
            top_sources=top_sources, 
            token_maps=token_maps, 
            analysis_id=analysis_id
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
            "page_images_sample": list(page_images.keys())[:10],
            "volumes_mounted": {
                "data_dir": os.path.exists(DATA_DIR),
                "db_dir": os.path.exists(DATABASE_DIR)
            }
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
