import os
import json
import fitz
import re
import cv2
import tempfile
from pdf2image import convert_from_path
import easyocr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid
import numpy as np
import math

FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".jpg", ".png", ".jpeg"]
QDRANT_COLLECTION_NAME = "test_chunking"


def check_image(filepath):
    doc = fitz.open(filepath)
    for page in doc:
        text = page.get_text()
        if text.strip():
            doc.close()
            return False
    doc.close()
    return True


def clean_text(raw_text):
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_text.strip())


def preprocess_text(text):
    # Loại bỏ dòng chứa số trang (Page 1, Trang 2, - 3 -, v.v.)
    text = re.sub(r'(?:Page|Trang)?\s*-?\s*\d+\s*-?', '', text, flags=re.IGNORECASE)

    # Loại bỏ các ký tự đặc biệt không cần thiết (giữ lại chữ, số, dấu cơ bản)
    text = re.sub(r"[^\w\s.,!?%\-–()]", "", text)

    text = re.sub(r'\n{2,}', '\n', text)  # dòng trống liên tiếp
    text = re.sub(r'[ \t]+', ' ', text)  # tab, nhiều khoảng trắng
    text = re.sub(r' +\n', '\n', text)  # khoảng trắng cuối dòng

    return text.strip()


def extract_text_with_ocr(file_path):
    reader = easyocr.Reader(['vi', 'en'])
    text = ""
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path)
        for image in images:
            temp_path = tempfile.mktemp(suffix='.png')
            image.save(temp_path, 'PNG')
            results = reader.readtext(temp_path, detail=0)
            text += "\n".join(results) + "\n"
            os.unlink(temp_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = reader.readtext(file_path, detail=0)
        text += "\n".join(results)
    return preprocess_text(clean_text(text))


def extract_text(file_path):
    file_name = os.path.basename(file_path)
    if file_path.lower().endswith(('.jpg', '.png', '.jpeg')) or (
            file_path.lower().endswith('.pdf') and check_image(file_path)):
        return extract_text_with_ocr(file_path), file_name
    elif file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return preprocess_text(clean_text(text)), file_name
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return preprocess_text(clean_text(text)), file_name
    elif file_path.lower().endswith('.docx'):
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return preprocess_text(clean_text(text)), file_name
    else:
        raise ValueError(f"Unsupported file type: {file_path}")



def filter_invalid_chunks(chunks, min_length=30):
    """Loại bỏ các chunk quá ngắn hoặc chỉ chứa dấu câu."""
    filtered = []
    for chunk in chunks:
        cleaned = chunk.strip()
        # Loại bỏ chunk chỉ chứa dấu chấm hoặc quá ngắn
        if len(cleaned) >= min_length and not re.fullmatch(r"[.?!,:;\"']+", cleaned):
            filtered.append(cleaned)
    return filtered


def semantic_chunking(text, embed_model):
    try:
        # Bước 1: chia văn bản lớn thành các đoạn nhỏ có overlap 20%
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        base_chunks = base_splitter.split_text(text)

        # Bước 2: áp dụng SemanticChunker lên từng đoạn nhỏ
        semantic_splitter = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )

        final_chunks = []
        for chunk in base_chunks:
            try:
                semantic_chunks = semantic_splitter.split_text(chunk)
                final_chunks.extend(semantic_chunks)
            except Exception as e:
                print(f"Semantic split failed on chunk: {e}")
                final_chunks.append(chunk)

        # Bước 3: loại bỏ chunk quá ngắn hoặc vô nghĩa
        filtered_chunks = filter_invalid_chunks(final_chunks)
        return filtered_chunks

    except Exception as e:
        print(f"Error during semantic chunking: {e}")
        return []


def save_chunks_to_json(chunks, file_name, output_dir="./output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"file_name": file_name, "chunks": chunks}, f, ensure_ascii=False, indent=4)
    return output_path


def store_in_qdrant(chunks, file_name, collection_name=QDRANT_COLLECTION_NAME):
    client = QdrantClient(url="http://localhost:6333")
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embed_model.client.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )

    embeddings = embed_model.embed_documents(chunks)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"file_name": file_name, "chunk_id": i, "text": chunk}
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=collection_name, points=points)


def process_file(file_path):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text, file_name = extract_text(file_path)
    if not text:
        return f"No text extracted from {file_path}"

    chunks = semantic_chunking(text, embed_model)
    if not chunks:
        return f"No chunks created for {file_path}"

    json_path = save_chunks_to_json(chunks, file_name)
    print(f"Chunks saved to {json_path}")

    store_in_qdrant(chunks, file_name)
    return f"Processed {file_path}: {len(chunks)} chunks created and stored."


def main():
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
