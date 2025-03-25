import modal
import os
import io
import re
import pickle
import base64
from typing import List
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np

DATA_DIR = "/bee_pdf"
UPLOADED_PDFS_DIR = "/bee_pdf/uploaded_pdfs"
PDF_IMAGES_DIR = "/bee_pdf/pdf_images"
COLPALI_MODEL_NAME = "vidore/colqwen2-v1.0"  # Using ColQwen2 model

# Update the image to include colpali-engine
bee_image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx","libglib2.0-0","libsm6","libxrender1","libxext6","poppler-utils")
    .pip_install(
        "faiss-cpu","pandas","numpy","huggingface_hub","sentence-transformers",
        "langchain","langchain-community","pypdf","Pillow","rapidocr-onnxruntime",
        "opencv-python-headless","rank-bm25","nltk","pdf2image","pymupdf",
        "colpali-engine","transformers","torch"  # Added these new dependencies
    )
)

app = modal.App("process_pdfs_and_store_embeddings")
bee_volume = modal.Volume.from_name("bee_pdf", create_if_missing=True)

def generate_unique_folder_name(pdf_path: str) -> str:
    import hashlib
    filename = os.path.basename(pdf_path)
    hash_obj = hashlib.md5(pdf_path.encode())
    return f"{filename.split('.')[0]}_{hash_obj.hexdigest()[:8]}"

@app.function(image=bee_image,
              volumes={DATA_DIR: bee_volume},
              cpu=2.0)

def upload_pdf(local_pdf_path):
    os.makedirs(UPLOADED_PDFS_DIR, exist_ok=True)
    filename = os.path.basename(local_pdf_path)
    dest_path = os.path.join(UPLOADED_PDFS_DIR, filename)
    with open(local_pdf_path, 'rb') as src_file:
        content = src_file.read()
        with open(dest_path, 'wb') as dest_file:
            dest_file.write(content)
    print(f"Uploaded PDF: {filename} to {dest_path}")
    bee_volume.commit()
    return dest_path

@app.function(image=bee_image,volumes={DATA_DIR: bee_volume},timeout=4*60*60)
def process_pdfs_and_store_embeddings():
    import torch
    from pdf2image import convert_from_path
    from PIL import Image
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    import pandas as pd
    import nltk
    import numpy as np
    
    # Setup NLTK for tokenization (still needed for other parts of your pipeline)
    NLTK_DATA_DIR = "/tmp/nltk_data"
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(NLTK_DATA_DIR)
    nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)

    # Load ColPali model
    print("Loading ColPali model...")
    model = ColQwen2.from_pretrained(
        COLPALI_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    ).eval()
    processor = ColQwen2Processor.from_pretrained(COLPALI_MODEL_NAME)
    print("ColPali model loaded successfully!")

    os.makedirs(UPLOADED_PDFS_DIR, exist_ok=True)
    os.makedirs(PDF_IMAGES_DIR, exist_ok=True)
    print(f"UPLOADED_PDFS_DIR: {UPLOADED_PDFS_DIR}")
    print(f"PDF_IMAGES_DIR: {PDF_IMAGES_DIR}")

    pdf_files = [f for f in os.listdir(UPLOADED_PDFS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {UPLOADED_PDFS_DIR}")
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    all_embeddings = []  # Store all multi-vector embeddings
    all_metadata = []    # Store metadata about each page
    page_images = {}     # Store image paths
    page_contents = []   # Store page text (we'll extract it from PDFs for backward compatibility)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(UPLOADED_PDFS_DIR, pdf_file)
        print(f"Processing {pdf_file}...")
        pdf_dir_name = pdf_file.replace('.pdf','')
        pdf_images_dir = os.path.join(PDF_IMAGES_DIR, pdf_dir_name)
        os.makedirs(pdf_images_dir, exist_ok=True)
        print(f"Images will be stored in: {pdf_images_dir}")

        # Convert PDF to images (as before)
        print("Converting PDF pages to images...")
        images = convert_from_path(pdf_path, dpi=150)
        
        # Process images in batches to avoid OOM
        batch_size = 4  # Adjust based on GPU memory
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_image_paths = []
            
            # First, save the images (as you were doing before)
            for j, image in enumerate(batch_images):
                page_num = i + j
                if page_num < len(images):
                    # Save image
                    w, h = image.size
                    new_w = min(w, 1000)
                    new_h = int(h*(new_w/w))
                    resized_image = image.resize((new_w, new_h), Image.LANCZOS)
                    image_path = os.path.join(pdf_images_dir, f"{page_num}.png")
                    resized_image.save(image_path, format='PNG', quality=85)
                    
                    # Store image path
                    image_key = f"{pdf_file}_{page_num}"
                    page_images[image_key] = image_path
                    batch_image_paths.append(image_path)
                    
                    # Store metadata
                    all_metadata.append({
                        'page': page_num,
                        'source_file': pdf_path,
                        'paragraph_size': 0,  # Not relevant for ColPali
                        'image_key': image_key,
                        'full_path': pdf_path,
                        'filename': pdf_file
                    })
            
            # Process batch with ColPali
            print(f"Generating embeddings for batch {i//batch_size + 1}...")
            processed_batch = processor.process_images(batch_images).to(model.device)
            
            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = model(**processed_batch)
            
            # Store embeddings
            for j, emb in enumerate(batch_embeddings):
                page_num = i + j
                if page_num < len(images):
                    # Convert to numpy and store
                    page_emb = emb.cpu().numpy()
                    all_embeddings.append(page_emb)
                    page_contents.append(f"[Image content from page {page_num} of {pdf_file}]")

    # Save the embeddings
    print(f"Saving {len(all_embeddings)} ColPali embeddings...")
    embeddings_path = os.path.join(DATA_DIR, "colpali_embeddings.pkl")
    with open(embeddings_path, "wb") as f:
        pickle.dump(all_embeddings, f)
    print(f"Embeddings saved to {embeddings_path}")
    
    # Save page images paths
    images_path = os.path.join(DATA_DIR, "pdf_page_image_paths.pkl")
    with open(images_path, "wb") as f:
        pickle.dump(page_images, f)
    print(f"PDF page image paths saved to {images_path}")
    
    # Save metadata
    df = pd.DataFrame({
        "filename": [os.path.basename(m['source_file']) for m in all_metadata],
        "page": [m['page'] for m in all_metadata],
        "paragraph_size": [m['paragraph_size'] for m in all_metadata],
        "text": page_contents,  # Include page text for backward compatibility
        "image_key": [m['image_key'] for m in all_metadata],
        "full_path": [m['source_file'] for m in all_metadata]
    })
    data_path = os.path.join(DATA_DIR, "data.pkl")
    df.to_pickle(data_path)
    print(f"DataFrame saved to {data_path}")
    
    # Create dummy FAISS index for backward compatibility
    # We won't actually use this for retrieval, but existing code might expect it
    #import faiss
    #index = faiss.IndexFlatIP(1)  # Dummy index
    #faiss_index_path = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
    #faiss.write_index(index, faiss_index_path)
    #print(f"Dummy FAISS index saved to {faiss_index_path} for backward compatibility")
    
    # For BM25, create a placeholder
    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "wb") as f:
        pickle.dump(None, f)
    with open(os.path.join(DATA_DIR, "tokenized_paragraphs.pkl"), "wb") as f:
        pickle.dump([], f)
    
    bee_volume.commit()
    
    print("\n✅ Processing complete! ColPali embeddings, PDF images, and metadata saved.")
    print(f"Total pages processed: {len(all_metadata)}")

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