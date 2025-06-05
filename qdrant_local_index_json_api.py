# from flask import Flask, request, jsonify
# import os
# import json
# import re
# import uuid
# import logging
# import asyncio
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import PointStruct, VectorParams, Distance
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
#
# QDRANT_COMMON_COLLECTION = "all_documents"
# FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".jpg", ".png", ".jpeg"]
#
# # Cache embedding model globally to avoid reinitialization
# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#
# def clean_text(raw_text):
#     """Clean text by removing special characters, extra whitespace, page numbers, and excessive punctuation."""
#     logger.info("Starting text cleaning")
#     try:
#         # Remove page numbers (e.g., "Page 146 of 156", "trang 123")
#         text = re.sub(r'(?i)(?:page|trang)\s*-?\s*\d+\s*(?:of\s*\d+)?\s*-?', '', raw_text)
#
#         # Remove excessive punctuation (e.g., sequences of 3 or more dots, commas, etc.)
#         text = re.sub(r'([.,!?;]){3,}', '', text)  # Remove 3+ repeated punctuation marks
#         text = re.sub(r'[^\w\s.,!?%\-–()]', '', text)  # Remove special characters except allowed ones
#         text = re.sub(r'\n{2,}', '\n', text)  # Normalize newlines
#         text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
#         text = re.sub(r'\s+\n', '\n', text)  # Remove spaces before newlines
#         cleaned = text.strip()
#
#         # Log a warning if the cleaned text is too short or empty
#         if not cleaned:
#             logger.warning("Cleaned text is empty")
#         else:
#             logger.info(f"Text cleaning completed, length: {len(cleaned)} characters")
#         return cleaned
#     except Exception as e:
#         logger.error(f"Text cleaning failed: {str(e)}")
#         raise
#
#
# def filter_invalid_chunks(chunks, min_length=30):
#     """Remove chunks that are too short, contain only punctuation, or lack semantic value."""
#     logger.info(f"Filtering {len(chunks)} chunks")
#     filtered = []
#     for chunk in chunks:
#         chunk = chunk.strip()
#         # Skip chunks that are too short
#         if len(chunk) < min_length:
#             continue
#         # Skip chunks that are mostly punctuation
#         if re.fullmatch(r'[.?!,:;"\']+', chunk):
#             continue
#         # Skip chunks with repetitive metadata-like patterns (e.g., "Title Doc ID SHE-3")
#         if re.match(r'^(Title\s*Doc\s*ID|Revision|Date|Page\s*\d+.*)$', chunk, re.IGNORECASE):
#             continue
#         # Skip chunks that are mostly numbers or single letters (e.g., "Y N Y N")
#         if re.match(r'^(?:\s*[YN]\s*)+$', chunk):
#             continue
#         filtered.append(chunk)
#     logger.info(f"Filtered to {len(filtered)} valid chunks")
#     return filtered
#
#
# def semantic_chunking(text, embed_model, overlap_words=25, min_diff_words=10):
#     """Chunk text semantically, then apply safe overlap of N words without duplication."""
#     logger.info("Starting semantic chunking with manual overlap")
#     try:
#         # Base split
#         base_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=5000,
#             chunk_overlap=0,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#         )
#         base_chunks = base_splitter.split_text(text)
#         logger.info(f"Base split created {len(base_chunks)} chunks")
#
#         # Semantic split
#         semantic_splitter = SemanticChunker(
#             embeddings=embed_model,
#             breakpoint_threshold_type="percentile",
#             breakpoint_threshold_amount=90
#         )
#
#         raw_chunks = []
#         for i, chunk in enumerate(base_chunks):
#             try:
#                 sub_chunks = semantic_splitter.split_text(chunk)
#                 raw_chunks.extend(sub_chunks)
#             except Exception as e:
#                 logger.warning(f"Semantic split failed on chunk {i+1}: {str(e)}")
#                 raw_chunks.append(chunk)
#
#         logger.info(f"Semantic chunking done. Total: {len(raw_chunks)} chunks")
#
#         # Manual overlap with safety check
#         overlapped_chunks = []
#         for i, chunk in enumerate(raw_chunks):
#             words = chunk.strip().split()
#             if i == 0:
#                 overlapped_chunks.append(chunk)
#                 continue
#
#             prev_words = raw_chunks[i - 1].strip().split()
#             if len(prev_words) < overlap_words + min_diff_words:
#                 # Previous chunk too small for safe overlap, skip it
#                 overlapped_chunks.append(chunk)
#                 continue
#
#             overlap = prev_words[-overlap_words:]
#             new_chunk_words = overlap + words
#             new_chunk_text = ' '.join(new_chunk_words)
#
#             # Prevent duplicate chunk (identical to previous)
#             if overlapped_chunks and new_chunk_text == overlapped_chunks[-1]:
#                 logger.debug(f"Skipped redundant chunk at index {i}")
#                 continue
#
#             overlapped_chunks.append(new_chunk_text)
#
#         filtered_chunks = filter_invalid_chunks(overlapped_chunks)
#         logger.info(f"Final chunk count after overlap + filter: {len(filtered_chunks)}")
#         return filtered_chunks
#
#     except Exception as e:
#         logger.error(f"Semantic chunking with overlap failed: {str(e)}")
#         return []
#
#
# def save_chunks_to_json(chunks, file_name, metadata, output_dir="./output"):
#     """Save chunks and metadata to a JSON file."""
#     logger.info(f"Saving chunks to JSON for file: {file_name}")
#     try:
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump({"file_name": file_name, "metadata": metadata, "chunks": chunks}, f, ensure_ascii=False, indent=4)
#         logger.info(f"Chunks saved to {output_path}")
#         return output_path
#     except Exception as e:
#         logger.error(f"Failed to save JSON: {str(e)}")
#         raise
#
#
# async def store_in_qdrant(chunks, file_name, metadata, collection_name, embed_model, client):
#     """Store chunks in a Qdrant collection with metadata asynchronously."""
#     logger.info(f"Storing {len(chunks)} chunks in collection: {collection_name}")
#     try:
#         # Check if collection exists, create if not
#         try:
#             client.get_collection(collection_name)
#         except:
#             client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=VectorParams(
#                     size=embed_model.client.get_sentence_embedding_dimension(),
#                     distance=Distance.COSINE
#                 )
#             )
#             logger.info(f"Created collection: {collection_name}")
#
#         # Generate embeddings
#         logger.info("Generating embeddings")
#         embeddings = embed_model.embed_documents(chunks)
#         logger.info(f"Generated {len(embeddings)} embeddings")
#
#         # Batch upsert points (batch size 100 for efficiency)
#         batch_size = 100
#         points = [
#             PointStruct(
#                 id=str(uuid.uuid4()),
#                 vector=embedding,
#                 payload={"file_name": file_name, "chunk_id": i, "text": chunk, **metadata}
#             )
#             for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
#         ]
#
#         for i in range(0, len(points), batch_size):
#             batch = points[i:i + batch_size]
#             client.upsert(collection_name=collection_name, points=batch)
#             logger.info(f"Upserted batch {i // batch_size + 1} with {len(batch)} points")
#
#         logger.info(f"Completed storing chunks in {collection_name}")
#     except Exception as e:
#         logger.error(f"Failed to store in Qdrant collection {collection_name}: {str(e)}")
#         raise
#
#
# @app.route('/process_document', methods=['POST'])
# async def process_document():
#     """Flask API to process document JSON and store in Qdrant collections."""
#     logger.info("Received /process_document request")
#     try:
#         data = request.get_json()
#         if not data or 'loai_phieu' not in data or 'metadata' not in data or 'content' not in data:
#             logger.error("Invalid JSON format")
#             return jsonify({"error": "Invalid JSON format. Must include 'loai_phieu', 'metadata', and 'content'."}), 400
#
#         loai_phieu = data['loai_phieu']
#         metadata = data['metadata']
#         content = data['content']
#         file_name = metadata.get('file_name', 'unknown_file')
#
#         if not content:
#             logger.error("No content provided")
#             return jsonify({"error": "No content provided."}), 400
#
#         # Warn about very large content (soft limit, no rejection)
#         content_length = len(content)
#         if content_length > 1_000_000:  # 1MB soft limit
#             logger.warning(f"Large content detected: {content_length} characters")
#
#         # Initialize Qdrant client with no timeout
#         client = QdrantClient(url="http://localhost:6333", timeout=None)
#
#         # Clean the content
#         logger.info("Cleaning content")
#         cleaned_content = clean_text(content)
#         if not cleaned_content:
#             logger.error("No valid content after cleaning")
#             return jsonify({"error": "No valid content after cleaning."}), 400
#
#         # Perform semantic chunking
#         logger.info("Starting chunking")
#         chunks = semantic_chunking(cleaned_content, embed_model)
#         if not chunks:
#             logger.error("No chunks created from content")
#             return jsonify({"error": "No chunks created from content."}), 400
#
#         # Save chunks to JSON
#         json_path = save_chunks_to_json(chunks, file_name, metadata)
#
#         # Store in Qdrant: specific collection (loai_phieu) and common collection
#         await store_in_qdrant(chunks, file_name, metadata, loai_phieu, embed_model, client)
#         await store_in_qdrant(chunks, file_name, metadata, QDRANT_COMMON_COLLECTION, embed_model, client)
#
#         response = {
#             "message": f"Processed {file_name}: {len(chunks)} chunks created and stored.",
#             "json_path": json_path,
#             "collections": [loai_phieu, QDRANT_COMMON_COLLECTION],
#             "content_length": content_length,
#             "chunk_count": len(chunks)
#         }
#         logger.info(f"Request completed successfully: {response['message']}")
#         return jsonify(response), 200
#
#     except Exception as e:
#         logger.error(f"Processing failed: {str(e)}")
#         return jsonify({"error": f"Processing failed: {str(e)}"}), 500
#     finally:
#         # Ensure client is closed
#         if 'client' in locals():
#             client.close()
#
#
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify
# import os
# import json
# import re
# import uuid
# import logging
# import asyncio
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import PointStruct, VectorParams, Distance
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# app = Flask(__name__)
#
# QDRANT_COMMON_COLLECTION = "all_documents"
# FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".jpg", ".png", ".jpeg"]
#
# # Cache embedding model globally to avoid reinitialization
# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# VECTOR_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dimensional vectors
#
#
# def clean_text(raw_text):
#     logger.info("Starting text cleaning")
#     try:
#         text = re.sub(r'(?i)(?:page|trang)\s*-?\s*\d+\s*(?:of\s*\d+)?\s*-?', '', raw_text)
#         text = re.sub(r'([.,!?;]){3,}', '', text)
#         text = re.sub(r'[^\w\s.,!?%\-–()]', '', text)
#         text = re.sub(r'\n{2,}', '\n', text)
#         text = re.sub(r'[ \t]+', ' ', text)
#         text = re.sub(r'\s+\n', '\n', text)
#         cleaned = text.strip()
#         if not cleaned:
#             logger.warning("Cleaned text is empty")
#         else:
#             logger.info(f"Text cleaning completed, length: {len(cleaned)} characters")
#         return cleaned
#     except Exception as e:
#         logger.error(f"Text cleaning failed: {str(e)}")
#         raise
#
#
# def filter_invalid_chunks(chunks, min_length=30):
#     logger.info(f"Filtering {len(chunks)} chunks")
#     filtered = []
#     for chunk in chunks:
#         chunk = chunk.strip()
#         if len(chunk) < min_length:
#             continue
#         if re.fullmatch(r'[.?!,:;"\']+', chunk):
#             continue
#         if re.match(r'^(Title\s*Doc\s*ID|Revision|Date|Page\s*\d+.*)$', chunk, re.IGNORECASE):
#             continue
#         if re.match(r'^(?:\s*[YN]\s*)+$', chunk):
#             continue
#         filtered.append(chunk)
#     logger.info(f"Filtered to {len(filtered)} valid chunks")
#     return filtered
#
#
# def semantic_chunking(text, embed_model, overlap_words=25, min_diff_words=10):
#     logger.info("Starting semantic chunking with manual overlap")
#     try:
#         base_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=5000,
#             chunk_overlap=0,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#         )
#         base_chunks = base_splitter.split_text(text)
#         logger.info(f"Base split created {len(base_chunks)} chunks")
#
#         semantic_splitter = SemanticChunker(
#             embeddings=embed_model,
#             breakpoint_threshold_type="percentile",
#             breakpoint_threshold_amount=90
#         )
#
#         raw_chunks = []
#         for i, chunk in enumerate(base_chunks):
#             try:
#                 sub_chunks = semantic_splitter.split_text(chunk)
#                 raw_chunks.extend(sub_chunks)
#             except Exception as e:
#                 logger.warning(f"Semantic split failed on chunk {i+1}: {str(e)}")
#                 raw_chunks.append(chunk)
#
#         logger.info(f"Semantic chunking done. Total: {len(raw_chunks)} chunks")
#
#         overlapped_chunks = []
#         for i, chunk in enumerate(raw_chunks):
#             words = chunk.strip().split()
#             if i == 0:
#                 overlapped_chunks.append(chunk)
#                 continue
#
#             prev_words = raw_chunks[i - 1].strip().split()
#             if len(prev_words) < overlap_words + min_diff_words:
#                 overlapped_chunks.append(chunk)
#                 continue
#
#             overlap = prev_words[-overlap_words:]
#             new_chunk_words = overlap + words
#             new_chunk_text = ' '.join(new_chunk_words)
#
#             if overlapped_chunks and new_chunk_text == overlapped_chunks[-1]:
#                 logger.debug(f"Skipped redundant chunk at index {i}")
#                 continue
#
#             overlapped_chunks.append(new_chunk_text)
#
#         filtered_chunks = filter_invalid_chunks(overlapped_chunks)
#         logger.info(f"Final chunk count after overlap + filter: {len(filtered_chunks)}")
#         return filtered_chunks
#
#     except Exception as e:
#         logger.error(f"Semantic chunking with overlap failed: {str(e)}")
#         return []
#
#
# def save_chunks_to_json(chunks, file_name, metadata, output_dir="./output"):
#     logger.info(f"Saving chunks to JSON for file: {file_name}")
#     try:
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump({"file_name": file_name, "metadata": metadata, "chunks": chunks}, f, ensure_ascii=False, indent=4)
#         logger.info(f"Chunks saved to {output_path}")
#         return output_path
#     except Exception as e:
#         logger.error(f"Failed to save JSON: {str(e)}")
#         raise
#
#
# async def store_in_qdrant(chunks, file_name, metadata, collection_name, embed_model, client):
#     logger.info(f"Storing {len(chunks)} chunks in collection: {collection_name}")
#     try:
#         try:
#             client.get_collection(collection_name)
#         except:
#             client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=VectorParams(
#                     size=VECTOR_DIM,
#                     distance=Distance.COSINE
#                 )
#             )
#             logger.info(f"Created collection: {collection_name}")
#
#         logger.info("Generating embeddings")
#         embeddings = embed_model.embed_documents(chunks)
#         logger.info(f"Generated {len(embeddings)} embeddings")
#
#         batch_size = 100
#         points = [
#             PointStruct(
#                 id=str(uuid.uuid4()),
#                 vector=embedding,
#                 payload={"file_name": file_name, "chunk_id": i, "text": chunk, **metadata}
#             )
#             for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
#         ]
#
#         for i in range(0, len(points), batch_size):
#             batch = points[i:i + batch_size]
#             client.upsert(collection_name=collection_name, points=batch)
#             logger.info(f"Upserted batch {i // batch_size + 1} with {len(batch)} points")
#
#         logger.info(f"Completed storing chunks in {collection_name}")
#     except Exception as e:
#         logger.error(f"Failed to store in Qdrant collection {collection_name}: {str(e)}")
#         raise
#
#
# @app.route('/process_document', methods=['POST'])
# async def process_document():
#     logger.info("Received /process_document request")
#     try:
#         data = request.get_json()
#         if not data or 'loai_phieu' not in data or 'metadata' not in data or 'content' not in data:
#             logger.error("Invalid JSON format")
#             return jsonify({"error": "Invalid JSON format. Must include 'loai_phieu', 'metadata', and 'content'."}), 400
#
#         loai_phieu = data['loai_phieu']
#         metadata = data['metadata']
#         content = data['content']
#         file_name = metadata.get('file_name', 'unknown_file')
#
#         if not content:
#             logger.error("No content provided")
#             return jsonify({"error": "No content provided."}), 400
#
#         content_length = len(content)
#         if content_length > 1_000_000:
#             logger.warning(f"Large content detected: {content_length} characters")
#
#         client = QdrantClient(url="http://localhost:6333", timeout=None)
#
#         logger.info("Cleaning content")
#         cleaned_content = clean_text(content)
#         if not cleaned_content:
#             logger.error("No valid content after cleaning")
#             return jsonify({"error": "No valid content after cleaning."}), 400
#
#         logger.info("Starting chunking")
#         chunks = semantic_chunking(cleaned_content, embed_model)
#         if not chunks:
#             logger.error("No chunks created from content")
#             return jsonify({"error": "No chunks created from content."}), 400
#
#         json_path = save_chunks_to_json(chunks, file_name, metadata)
#
#         await store_in_qdrant(chunks, file_name, metadata, loai_phieu, embed_model, client)
#         await store_in_qdrant(chunks, file_name, metadata, QDRANT_COMMON_COLLECTION, embed_model, client)
#
#         response = {
#             "message": f"Processed {file_name}: {len(chunks)} chunks created and stored.",
#             "json_path": json_path,
#             "collections": [loai_phieu, QDRANT_COMMON_COLLECTION],
#             "content_length": content_length,
#             "chunk_count": len(chunks)
#         }
#         logger.info(f"Request completed successfully: {response['message']}")
#         return jsonify(response), 200
#
#     except Exception as e:
#         logger.error(f"Processing failed: {str(e)}")
#         return jsonify({"error": f"Processing failed: {str(e)}"}), 500
#     finally:
#         if 'client' in locals():
#             client.close()
#
#
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=5000)


import os
import json
import fitz
import re
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid
from docx import Document
from pdf2image import convert_from_path
import easyocr
import tempfile
import torch
from langchain_experimental.text_splitter import SemanticChunker
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".jpg", ".png", ".jpeg"]
GENERAL_COLLECTION_NAME = "general_documents"
reader = easyocr.Reader(['vi', 'en'], gpu=False)  # Global easyocr reader, CPU
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
client = QdrantClient(url="http://localhost:6333")
created_collections = set()  # Cache for collection existence


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
    text = re.sub(r'(?:Page|Trang)?\s*-?\s*\d+\s*-?', '', text, flags=re.IGNORECASE)
    text = re.sub(r"[^\w\s.,!?%\-–()]", "", text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


def extract_text_with_ocr(file_path):
    text = ""
    if file_path.lower().endswith('.pdf'):
        images = convert_from_path(file_path, dpi=150, grayscale=True)
        for image in images:
            temp_path = tempfile.mktemp(suffix='.png')
            image.save(temp_path, 'PNG')
            results = reader.readtext(temp_path, detail=0, low_text=0.3)
            text += "\n".join(results) + "\n"
            os.unlink(temp_path)
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = reader.readtext(file_path, detail=0, low_text=0.3)
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
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return preprocess_text(clean_text(text)), file_name
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def filter_invalid_chunks(chunks, min_length=30):
    filtered = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if len(cleaned) >= min_length and not re.fullmatch(r"[.?!,:;\"']+", cleaned):
            filtered.append(cleaned)
    return filtered


# def semantic_chunking(text, embed_model):
#     try:
#         base_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#         )
#         chunks = base_splitter.split_text(text)
#         return filter_invalid_chunks(chunks, min_length=30)
#     except Exception as e:
#         print(f"Error during chunking: {e}")
#         return []

def semantic_chunking(text, embed_model):
    try:
        # Khởi tạo SemanticChunker để chia văn bản thành các chunk ngữ nghĩa
        semantic_splitter = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        semantic_chunks = semantic_splitter.split_text(text)

        # Khởi tạo RecursiveCharacterTextSplitter với overlap 20%
        overlap_percentage = 0.2
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Kích thước chunk có thể điều chỉnh
            chunk_overlap=int(500 * overlap_percentage),
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            is_separator_regex=False
        )

        # Chia các chunk ngữ nghĩa thành các phần nhỏ hơn với overlap hợp lý
        final_chunks = []
        for chunk in semantic_chunks:
            sub_chunks = splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)

        # Lọc các chunk quá ngắn (dưới 30 ký tự)
        return [chunk for chunk in final_chunks if len(chunk) >= 30]
    except Exception as e:
        print(f"Error during chunking: {e}")
        return []


def ensure_collection(client, collection_name, embed_model):
    if collection_name not in created_collections:
        try:
            client.get_collection(collection_name)
            created_collections.add(collection_name)
        except:
            # Get embedding dimension by embedding a dummy text
            embedding = embed_model.embed_query("test")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(embedding),
                    distance=Distance.COSINE
                )
            )
            created_collections.add(collection_name)


def store_in_qdrant(chunks, file_name, collection_name, form_data, embed_model, client):
    batch_size = 16
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings.extend(embed_model.embed_documents(batch))

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "file_name": file_name,
                "chunk_id": i,
                "text": chunk,
                **(form_data or {})
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    client.upsert(collection_name=collection_name, points=points)


def process_single_file(file_info, loai_phieu, form_data, embed_model, client):
    file_path = file_info.get('path')
    file_name = file_info.get('file_name')
    file_type = file_info.get('file_type')

    if not file_path or not file_name or not file_type:
        return {
            "file_name": file_name or "unknown",
            "status": "error",
            "message": "Missing file information"
        }

    if not os.path.exists(file_path):
        return {
            "file_name": file_name,
            "status": "error",
            "message": f"File not found: {file_path}"
        }

    if not any(file_path.lower().endswith(ext) for ext in FILE_EXTENSIONS):
        return {
            "file_name": file_name,
            "status": "error",
            "message": f"Unsupported file type. Supported extensions: {', '.join(FILE_EXTENSIONS)}"
        }

    try:
        text, extracted_file_name = extract_text(file_path)
        if not text:
            return {
                "file_name": file_name,
                "status": "error",
                "message": "No text extracted"
            }

        chunks = semantic_chunking(text, embed_model)
        if not chunks:
            return {
                "file_name": file_name,
                "status": "error",
                "message": "No chunks created"
            }

        ensure_collection(client, loai_phieu, embed_model)
        ensure_collection(client, GENERAL_COLLECTION_NAME, embed_model)
        store_in_qdrant(chunks, file_name, loai_phieu, form_data, embed_model, client)
        store_in_qdrant(chunks, file_name, GENERAL_COLLECTION_NAME, form_data, embed_model, client)

        return {
            "file_name": file_name,
            "status": "success",
            "message": f"Processed: {len(chunks)} chunks created and stored",
            "content": chunks
        }
    except Exception as e:
        return {
            "file_name": file_name,
            "status": "error",
            "message": f"Error processing file: {str(e)}"
        }


@app.route('/store-documents', methods=['POST'])
def store_documents():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        loai_phieu = data.get('loai_phieu')
        form_data = data.get('formData', {})
        files = data.get('files', [])

        if not loai_phieu:
            return jsonify({"error": "Missing loai_phieu"}), 400
        if not files:
            return jsonify({"error": "No files provided"}), 400

        results = []
        for file_info in files:
            result = process_single_file(file_info, loai_phieu, form_data, embed_model, client)
            results.append(result)

        return jsonify({
            "status": "completed",
            "results": results
        }), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
