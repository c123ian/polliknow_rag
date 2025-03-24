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

# Constants
MODELS_DIR = "/Qwen"
MODEL_NAME = "Qwen2.5-7B-Instruct-1M"
FAISS_DATA_DIR = "/faiss_data_pdfs"
UPLOADED_PDFS_DIR = "/faiss_data_pdfs/uploaded_pdfs"
PDF_IMAGES_DIR = "/faiss_data_pdfs/pdf_images"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
USERNAME = "c123ian"
APP_NAME = "rag"
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

# Update: add pdf2image to the install list
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
        "pdf2image"
    )

# Replace the old faiss_volume definition with the new one
faiss_volume = modal.Volume.from_name("faiss_data_pdfs", create_if_missing=True)

try:
    db_volume = modal.Volume.lookup("db_data", create_if_missing=True)
except modal.exception.NotFoundError:
    db_volume = modal.Volume.persisted("db_data")

app = modal.App(APP_NAME)

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="40GB"),
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
        max_model_len=367584
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
    # openai_serving_chat = OpenAIServingChat(engine, model_config, models, "assistant", request_logger, None, "string")

    openai_serving_chat = OpenAIServingChat(
    engine_client=engine,  # ✅ First positional argument
    model_config=model_config,  # ✅ Second positional argument
    models=models,  # ✅ Added models argument
    response_role="assistant",  # ✅ Third positional argument
    request_logger=request_logger,  # ✅ Fourth positional argument
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
        FAISS_DATA_DIR: faiss_volume,
        DATABASE_DIR: db_volume
    },
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

    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

    print(f"Contents of FAISS_DATA_DIR ({FAISS_DATA_DIR}):")
    if os.path.exists(FAISS_DATA_DIR):
        print(f"  Directory exists, contains: {os.listdir(FAISS_DATA_DIR)}")
    else:
        print(f"  Directory does not exist!")

    print(f"Contents of PDF_IMAGES_DIR ({PDF_IMAGES_DIR}):")
    if os.path.exists(PDF_IMAGES_DIR):
        print(f"  Directory exists, contains: {os.listdir(PDF_IMAGES_DIR)}")
    else:
        print(f"  Directory does not exist!")

    FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
    DATA_PICKLE_PATH = os.path.join(FAISS_DATA_DIR, "data.pkl")
    PDF_PAGE_IMAGES_PATH = os.path.join(FAISS_DATA_DIR, "pdf_page_image_paths.pkl")

    print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
    print(f"  File exists: {os.path.exists(FAISS_INDEX_PATH)}")
    index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"Loading DataFrame from {DATA_PICKLE_PATH}")
    print(f"  File exists: {os.path.exists(DATA_PICKLE_PATH)}")
    df = pd.read_pickle(DATA_PICKLE_PATH)
    docs = df['text'].tolist()

    print(f"Loading image paths from {PDF_PAGE_IMAGES_PATH}")
    print(f"  File exists: {os.path.exists(PDF_PAGE_IMAGES_PATH)}")
    page_images = {}
    try:
        with open(PDF_PAGE_IMAGES_PATH, "rb") as f:
            page_images = pickle.load(f)
        print(f"  Loaded {len(page_images)} image paths")
        for key, path in list(page_images.items())[:2]:
            print(f"  Image key: {key}, path: {path}")
            print(f"  Path exists: {os.path.exists(path)}")
    except Exception as e:
        print(f"  Error loading image paths: {e}")
        logging.error(f"Error loading PDF page images: {e}")

    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def create_bm25_index(documents):
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        bm25_index = BM25Okapi(tokenized_docs)
        return bm25_index, tokenized_docs

    bm25_index, tokenized_docs = create_bm25_index(docs)

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

    def chat_top_sources(top_sources):
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

        # Navigation buttons for each image
        carousel_controls = Div(
            *[
                A(str(i+1), href=f"#item{i+1}", cls="btn btn-xs")
                for i in range(len(top_sources))
            ],
            cls="flex w-full justify-center gap-2 py-2"
        )

        return Div(
            Div(
                Div("Top Sources", cls="text-zinc-400 text-sm font-semibold"),
                Div(
                    Div(*carousel_items, cls="carousel w-full"),
                    carousel_controls,  # Buttons for navigation
                    cls="flex flex-col w-full"
                ),
                cls="flex flex-col w-full gap-6"
            ),
            cls="w-full max-w-2xl mx-auto bg-zinc-800 rounded-md mt-6 p-6"
        )





    @rt("/debug")
    async def debug_info():
        html = ["<h1>Debug Information</h1>"]
        directories = [
            FAISS_DATA_DIR,
            PDF_IMAGES_DIR,
            UPLOADED_PDFS_DIR
        ]
        html.append("<h2>Directories:</h2><ul>")
        for directory in directories:
            if os.path.exists(directory):
                items = os.listdir(directory)
                html.append(f"<li>✅ {directory}: {len(items)} items</li>")
                html.append("<ul>")
                for item in items[:10]:
                    full_path = os.path.join(directory, item)
                    is_dir = os.path.isdir(full_path)
                    html.append(f"<li>{'📁' if is_dir else '📄'} {item}</li>")
                if len(items) > 10:
                    html.append(f"<li>... and {len(items) - 10} more</li>")
                html.append("</ul>")
            else:
                html.append(f"<li>❌ {directory}: Not found</li>")
        html.append("</ul>")

        html.append("<h2>Image Paths:</h2>")
        if page_images:
            html.append(f"<p>Found {len(page_images)} image paths</p>")
            html.append("<ul>")
            for key, path in list(page_images.items())[:5]:
                exists = os.path.exists(path)
                html.append(f"<li>{key} → {path} {'✅' if exists else '❌'}</li>")
            html.append("</ul>")

            if page_images:
                test_key = next(iter(page_images.keys()))
                test_path = page_images[test_key]
                html.append(f"<h3>Testing image: {test_key}</h3>")
                if os.path.exists(test_path):
                    try:
                        with open(test_path, "rb") as img_file:
                            img_data = img_file.read()
                            base64_data = base64.b64encode(img_data).decode('utf-8')
                            html.append("<p>✅ Image loaded successfully</p>")
                            html.append(f'<img src="data:image/png;base64,{base64_data}" style="max-width:300px; border:1px solid #ccc;">')
                    except Exception as e:
                        html.append(f"<p>❌ Error reading image: {e}</p>")
                else:
                    html.append(f"<p>❌ Image not found at path: {test_path}</p>")
        else:
            html.append("<p>No image paths loaded</p>")

        return HTMLResponse(content="<br>".join(html))

    @rt("/")
    async def get(session):
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        messages = await load_chat_history(session_id)
        return Div(
            H1("Chat with PDF Documents", cls="text-3xl font-bold mb-4 text-white"),
            Div(f"Session ID: {session_id}", cls="text-white mb-4"),
            A("Debug Info", href="/debug", cls="text-blue-500 underline mb-4", target="_blank"),
            chat(session_id=session_id, messages=messages),
            Div(Span("Model status: "), Span("⚫", id="model-status-emoji"), cls="model-status text-white mt-4"),
            Div(id="top-sources"),
            cls="flex flex-col items-center min-h-screen bg-black",
        )

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

        await send(chat_form(disabled=False))
        await send(Div(chat_message(message_index, messages=messages), id="messages", hx_swap_oob="beforeend"))

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
