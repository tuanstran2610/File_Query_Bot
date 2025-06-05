import os
import json
import fitz  # PyMuPDF
import re
import cv2
import tempfile
from pdf2image import convert_from_path
import pytesseract as pyt
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.utils import embedding_functions

# Set Tesseract path for OCR
pyt.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Supported file extensions
FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".jpg", ".png", ".jpeg"]

def check_image(filepath):
    """Check if a PDF is image-based (no selectable text)."""
    doc = fitz.open(filepath)
    for page in doc:
        text = page.get_text()
        if text.strip():
            doc.close()
            return False
    doc.close()
    return True

def clean_text(raw_text):
    """Clean text by removing unnecessary line breaks and extra whitespace."""
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_text.strip())

def extract_text_with_ocr(file_path):
    """Extract text from image-based files (PDF, PNG, JPG, JPEG)."""
    text = ""
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path)
        for image in images:
            temp_path = tempfile.mktemp(suffix='.png')
            image.save(temp_path, 'PNG')
            img = cv2.imread(temp_path)
            text += pyt.image_to_string(img, lang="eng") + "\n"
            os.unlink(temp_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        text += pyt.image_to_string(img, lang="eng")
    return clean_text(text)

def extract_text(file_path):
    """Extract text from a file based on its type."""
    file_name = os.path.basename(file_path)
    if file_path.lower().endswith(('.jpg', '.png', '.jpeg')) or (file_path.lower().endswith('.pdf') and check_image(file_path)):
        return extract_text_with_ocr(file_path), file_name
    elif file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return clean_text(text), file_name
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return clean_text(text), file_name
    elif file_path.lower().endswith('.docx'):
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return clean_text(text), file_name
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def semantic_chunking(text, embed_model):
    """Chunk text semantically using LangChain's SemanticChunker."""
    try:
        text_splitter = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",  # Options: 'percentile', 'standard_deviation', 'interquartile'
            breakpoint_threshold_amount=90  # Adjust threshold for chunk size
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error during semantic chunking: {e}")
        return []

def save_chunks_to_json(chunks, file_name, output_dir="./output"):
    """Save chunks to a JSON file for inspection."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"file_name": file_name, "chunks": chunks}, f, ensure_ascii=False, indent=4)
    return output_path

def store_in_chromadb(chunks, file_name, db_path="./ChromaDB", collection_name="collections"):
    """Store chunks in ChromaDB with embeddings."""
    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=db_path)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = db.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Store each chunk with metadata
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"file_name": file_name, "chunk_id": i}],
            ids=[f"{file_name}_{i}"]
        )

def process_file(file_path):
    """Main function to process a file: extract, chunk, embed, store, and save chunks."""
    # Initialize embedding model for chunking
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Extract text
    text, file_name = extract_text(file_path)
    if not text:
        return f"No text extracted from {file_path}"

    # Perform semantic chunking
    chunks = semantic_chunking(text, embed_model)
    if not chunks:
        return f"No chunks created for {file_path}"

    # Save chunks to JSON
    json_path = save_chunks_to_json(chunks, file_name)
    print(f"Chunks saved to {json_path}")

    # Store chunks in ChromaDB
    store_in_chromadb(chunks, file_name)
    return f"Processed {file_path}: {len(chunks)} chunks created and stored."

def main():
    """Main loop to get user input and process files."""
    while True:
        file_path = input("Enter the path to the file (or 'quit' to exit): ")
        if file_path.lower() == 'quit':
            break
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        if not any(file_path.lower().endswith(ext) for ext in FILE_EXTENSIONS):
            print(f"Unsupported file type. Supported extensions: {', '.join(FILE_EXTENSIONS)}")
            continue
        result = process_file(file_path)
        print(result)

if __name__ == "__main__":
    main()