import modal
import os
import io
import re
import pickle
import base64
from typing import List
from concurrent.futures import ThreadPoolExecutor

FAISS_DATA_DIR = "/faiss_data_pdfs"
UPLOADED_PDFS_DIR = "/faiss_data_pdfs/uploaded_pdfs"
PDF_IMAGES_DIR = "/faiss_data_pdfs/pdf_images"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

pandas_faiss_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx","libglib2.0-0","libsm6","libxrender1","libxext6","poppler-utils")
    .pip_install(
        "faiss-cpu","pandas","numpy","huggingface_hub","sentence-transformers",
        "langchain","langchain-community","pypdf","Pillow","rapidocr-onnxruntime",
        "opencv-python-headless","rank-bm25","nltk","pdf2image","pymupdf"
    )
)

app = modal.App("process_pdfs_and_store_embeddings")
faiss_volume = modal.Volume.from_name("faiss_data_pdfs", create_if_missing=True)

def generate_unique_folder_name(pdf_path: str) -> str:
    import hashlib
    filename = os.path.basename(pdf_path)
    hash_obj = hashlib.md5(pdf_path.encode())
    return f"{filename.split('.')[0]}_{hash_obj.hexdigest()[:8]}"

def split_into_paragraphs(text: str, max_paragraph_size: int = 512, min_paragraph_size: int = 100) -> List[str]:
    raw_paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    processed_paragraphs = []
    current_paragraph = []
    current_size = 0
    for raw_paragraph in raw_paragraphs:
        if len(raw_paragraph) > max_paragraph_size:
            sentences = re.split(r'(?<=[.!?])\s+', raw_paragraph)
            for sentence in sentences:
                sentence_len = len(sentence)
                if current_size + sentence_len <= max_paragraph_size:
                    current_paragraph.append(sentence)
                    current_size += sentence_len
                else:
                    if current_size >= min_paragraph_size:
                        processed_paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = [sentence]
                    current_size = sentence_len
        else:
            if current_size + len(raw_paragraph) > max_paragraph_size:
                if current_size >= min_paragraph_size:
                    processed_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [raw_paragraph]
                current_size = len(raw_paragraph)
            else:
                current_paragraph.append(raw_paragraph)
                current_size += len(raw_paragraph)
    if current_paragraph and current_size >= min_paragraph_size:
        processed_paragraphs.append(' '.join(current_paragraph))
    return processed_paragraphs

@app.function(image=pandas_faiss_image,volumes={FAISS_DATA_DIR: faiss_volume})
def upload_pdf(local_pdf_path):
    os.makedirs(UPLOADED_PDFS_DIR, exist_ok=True)
    filename = os.path.basename(local_pdf_path)
    dest_path = os.path.join(UPLOADED_PDFS_DIR, filename)
    with open(local_pdf_path, 'rb') as src_file:
        content = src_file.read()
        with open(dest_path, 'wb') as dest_file:
            dest_file.write(content)
    print(f"Uploaded PDF: {filename} to {dest_path}")
    faiss_volume.commit()
    return dest_path

@app.function(image=pandas_faiss_image,volumes={FAISS_DATA_DIR: faiss_volume},timeout=4*60*60)
def process_pdfs_and_store_embeddings():
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.document_loaders.parsers import RapidOCRBlobParser
    import faiss
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import nltk
    from nltk.tokenize import word_tokenize
    from pdf2image import convert_from_path
    from PIL import Image

    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    os.makedirs(UPLOADED_PDFS_DIR, exist_ok=True)
    os.makedirs(PDF_IMAGES_DIR, exist_ok=True)
    print(f"UPLOADED_PDFS_DIR: {UPLOADED_PDFS_DIR}")
    print(f"PDF_IMAGES_DIR: {PDF_IMAGES_DIR}")

    pdf_files = [f for f in os.listdir(UPLOADED_PDFS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {UPLOADED_PDFS_DIR}")
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    all_paragraphs = []
    all_metadata = []
    page_images = {}

    for pdf_file in pdf_files:
        pdf_path = os.path.join(UPLOADED_PDFS_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        pdf_dir_name = pdf_file.replace('.pdf','')
        pdf_images_dir = os.path.join(PDF_IMAGES_DIR, pdf_dir_name)
        os.makedirs(pdf_images_dir, exist_ok=True)
        print(f"Images will be stored in: {pdf_images_dir}")

        # Use the recommended PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        def process_page(page_num):
            try:
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=150)
                if not images: return None
                image = images[0]
                w, h = image.size
                new_w = min(w, 1000)
                new_h = int(h*(new_w/w))
                resized_image = image.resize((new_w, new_h), Image.LANCZOS)
                image_path = os.path.join(pdf_images_dir, f"{page_num}.png")
                resized_image.save(image_path, format='PNG', quality=85)
                print(f"Saved image for {pdf_file} page {page_num} to {image_path}")
                return image_path
            except Exception as e:
                print(f"Error processing page {page_num}: {e}")
                return None

        print("Converting PDF pages to images...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_page = {executor.submit(process_page, i): i for i in range(len(docs))}
            for future in future_to_page:
                p_num = future_to_page[future]
                try:
                    img_path = future.result()
                    if img_path:
                        image_key = f"{pdf_file}_{p_num}"
                        page_images[image_key] = img_path
                        print(f"Stored image path for key {image_key}: {img_path}")
                except Exception as e:
                    print(f"Error saving image for page {p_num}: {e}")

        for doc in docs:
            page_num = doc.metadata.get('page', 0)
            page_paragraphs = split_into_paragraphs(doc.page_content, max_paragraph_size=512, min_paragraph_size=100)
            for paragraph in page_paragraphs:
                all_paragraphs.append(paragraph)
                all_metadata.append({
                    'page': page_num,
                    'source_file': pdf_path,
                    'paragraph_size': len(paragraph),
                    'image_key': f"{pdf_file}_{page_num}"
                })

    print(f"Generating embeddings for {len(all_paragraphs)} paragraphs...")
    embeddings = embedding_model.encode(all_paragraphs,convert_to_tensor=False,show_progress_bar=True,batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss_index_path = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")
    print(f"File exists: {os.path.exists(faiss_index_path)}")

    print("Creating BM25 index...")
    tokenized_paragraphs = [word_tokenize(par.lower()) for par in all_paragraphs]
    bm25_index = BM25Okapi(tokenized_paragraphs)
    bm25_path = os.path.join(FAISS_DATA_DIR, "bm25_index.pkl")
    with open(bm25_path,"wb") as f:
        pickle.dump(bm25_index,f)
    print(f"BM25 index saved to {bm25_path}")
    print(f"File exists: {os.path.exists(bm25_path)}")

    tokenized_path = os.path.join(FAISS_DATA_DIR, "tokenized_paragraphs.pkl")
    with open(tokenized_path,"wb") as f:
        pickle.dump(tokenized_paragraphs,f)
    print(f"Tokenized paragraphs saved to {tokenized_path}")
    print(f"File exists: {os.path.exists(tokenized_path)}")

    print("Saving PDF page image paths...")
    for key, path in list(page_images.items())[:3]:
        print(f"Image key: {key}, path: {path}, exists: {os.path.exists(path)}")
    images_path = os.path.join(FAISS_DATA_DIR, "pdf_page_image_paths.pkl")
    with open(images_path,"wb") as f:
        pickle.dump(page_images,f)
    print(f"PDF page image paths saved to {images_path}")
    print(f"File exists: {os.path.exists(images_path)}")

    print("Saving metadata...")
    import pandas as pd
    df = pd.DataFrame({
        "filename":[os.path.basename(m['source_file']) for m in all_metadata],
        "page":[m['page'] for m in all_metadata],
        "paragraph_size":[m['paragraph_size'] for m in all_metadata],
        "text":all_paragraphs,
        "image_key":[m['image_key'] for m in all_metadata],
        "full_path":[m['source_file'] for m in all_metadata]
    })
    data_path = os.path.join(FAISS_DATA_DIR, "data.pkl")
    df.to_pickle(data_path)
    print(f"DataFrame saved to {data_path}")
    print(f"File exists: {os.path.exists(data_path)}")

    print("\nContents of FAISS_DATA_DIR:")
    for item in os.listdir(FAISS_DATA_DIR):
        print(f"  {item}")

    print("\nContents of PDF_IMAGES_DIR:")
    for item in os.listdir(PDF_IMAGES_DIR):
        print(f"  {item}")

    print("\nCommitting volume...")
    faiss_volume.commit()

    print("\n✅ Processing complete! FAISS index, BM25 index, PDF images, and text saved.")
    print(f"Total paragraphs processed: {len(all_paragraphs)}")
    print(f"Average paragraph size: {sum(m['paragraph_size'] for m in all_metadata)/len(all_metadata):.2f} characters")
    print(f"Total PDF pages stored as images: {len(page_images)}")

@app.local_entrypoint()
def main(command="process", pdf_path=None):
    if command == "process":
        print("Processing all PDFs in volume...")
        process_pdfs_and_store_embeddings.remote()
    elif command == "upload" and pdf_path:
        print(f"Uploading PDF: {pdf_path}")
        upload_pdf.remote(pdf_path)
    else:
        print("Usage:")
        print("  python embedding.py process")
        print("  python embedding.py upload /path/to/your.pdf")
