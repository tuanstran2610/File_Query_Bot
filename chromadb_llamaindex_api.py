# from flask import Flask, request, jsonify
# import chromadb
# from llama_index.core import Document, VectorStoreIndex, StorageContext
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import uuid
#
# app = Flask(__name__)
#
# # Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
#
# # Initialize embedding model
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# # Common collection name for all files
# COMMON_COLLECTION = "all_files"
#
#
# def get_or_create_collection(collection_name):
#     """Get an existing collection or create a new one if it doesn't exist."""
#     try:
#         return chroma_client.get_collection(name=collection_name)
#     except:
#         return chroma_client.create_collection(name=collection_name)
#
#
# @app.route('/api/documents', methods=['POST'])
# def create_document():
#     """Create a new document in ChromaDB collections."""
#     try:
#         data = request.get_json()
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#         file_id = metadata.get("file_id")
#
#         if not file_id:
#             return jsonify({"error": "file_id is required in metadata"}), 400
#
#         # Get or create collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#
#         # Create a LlamaIndex Document
#         document = Document(
#             text=content,
#             metadata=metadata,
#             id_=file_id
#         )
#
#         # Store in specific collection
#         vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#         storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#         index_specific = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_specific, embed_model=embed_model
#         )
#
#         # Store in common collection
#         vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#         storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#         index_common = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_common, embed_model=embed_model
#         )
#
#         return jsonify({"message": "Document created successfully", "file_id": file_id}), 201
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['PUT'])
# def update_document(file_id):
#     """Update a document based on file_id."""
#     try:
#         data = request.get_json()
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#
#         # Get collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#
#         # Delete existing document
#         specific_collection.delete(ids=[file_id])
#         common_collection.delete(ids=[file_id])
#
#         # Create new document
#         document = Document(
#             text=content,
#             metadata=metadata,
#             id_=file_id
#         )
#
#         # Store in specific collection
#         vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#         storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#         index_specific = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_specific, embed_model=embed_model
#         )
#
#         # Store in common collection
#         vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#         storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#         index_common = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_common, embed_model=embed_model
#         )
#
#         return jsonify({"message": "Document updated successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['DELETE'])
# def delete_document(file_id):
#     """Delete a document based on file_id."""
#     try:
#         # Get list of collection names
#         collection_names = chroma_client.list_collections()
#
#         # Iterate through collection names and delete from each
#         for collection_name in collection_names:
#             try:
#                 collection = chroma_client.get_collection(name=collection_name)
#                 collection.delete(ids=[file_id])  # Delete from the collection
#             except Exception as e:
#                 # Skip collections that don't exist or can't be accessed
#                 continue
#
#         return jsonify({"message": "Document deleted successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['GET'])
# def get_document(file_id):
#     """Retrieve a document based on file_id from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get(ids=[file_id])
#
#         if not result["ids"]:
#             return jsonify({"error": "Document not found"}), 404
#
#         document = {
#             "file_id": result["ids"][0],
#             "metadata": result["metadatas"][0],
#             "content": result["documents"][0]
#         }
#
#         return jsonify(document), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


# from flask import Flask, request, jsonify
# import chromadb
# from llama_index.core import Document, VectorStoreIndex, StorageContext
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import uuid
#
# app = Flask(__name__)
#
# # Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
#
# # Initialize embedding model
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# # Common collection name for all files
# COMMON_COLLECTION = "all_files"
#
#
# def get_or_create_collection(collection_name):
#     """Get an existing collection or create a new one if it doesn't exist."""
#     try:
#         return chroma_client.get_collection(name=collection_name)
#     except:
#         return chroma_client.create_collection(name=collection_name)
#
#
# @app.route('/api/documents', methods=['POST'])
# def create_document():
#     """Create a new document in ChromaDB collections."""
#     try:
#         data = request.get_json()
#         print("Received JSON:", data)  # Debug: Log received JSON
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#         file_id = metadata.get("file_id")
#         print(f"loai_phieu: {loai_phieu}, file_id: {file_id}")  # Debug: Log key fields
#
#         if not file_id:
#             return jsonify({"error": "file_id is required in metadata"}), 400
#
#         # Get or create collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         print(f"Collections created: {loai_phieu}, {COMMON_COLLECTION}")  # Debug: Log collections
#
#         # Create a LlamaIndex Document
#         document = Document(
#             text=content,
#             metadata=metadata,
#             id_=file_id
#         )
#         print("Document created:", document.id_)  # Debug: Log document ID
#
#         # Store in specific collection
#         vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#         storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#         index_specific = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_specific, embed_model=embed_model
#         )
#         print(f"Stored in {loai_phieu} collection")  # Debug: Log storage
#
#         # Store in common collection
#         vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#         storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#         index_common = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_common, embed_model=embed_model
#         )
#         print(f"Stored in {COMMON_COLLECTION} collection")  # Debug: Log storage
#
#         # Verify storage
#         result = common_collection.get(ids=[file_id])
#         if result["ids"]:
#             print("Document verified in all_files:", result["ids"])  # Debug: Log verification
#         else:
#             print("Document not found in all_files after storage")  # Debug: Log failure
#
#         return jsonify({"message": "Document created successfully", "file_id": file_id}), 201
#
#     except Exception as e:
#         print(f"Error in create_document: {str(e)}")  # Debug: Log error
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['PUT'])
# def update_document(file_id):
#     """Update a document based on file_id."""
#     try:
#         data = request.get_json()
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#
#         # Get collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#
#         # Delete existing document
#         specific_collection.delete(ids=[file_id])
#         common_collection.delete(ids=[file_id])
#
#         # Create new document
#         document = Document(
#             text=content,
#             metadata=metadata,
#             id_=file_id
#         )
#
#         # Store in specific collection
#         vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#         storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#         index_specific = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_specific, embed_model=embed_model
#         )
#
#         # Store in common collection
#         vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#         storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#         index_common = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_common, embed_model=embed_model
#         )
#
#         return jsonify({"message": "Document updated successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['DELETE'])
# def delete_document(file_id):
#     """Delete a document based on file_id."""
#     try:
#         # Get list of collection names
#         collection_names = chroma_client.list_collections()
#
#         # Iterate through collection names and delete from each
#         for collection_name in collection_names:
#             try:
#                 collection = chroma_client.get_collection(name=collection_name)
#                 collection.delete(ids=[file_id])  # Delete from the collection
#             except Exception as e:
#                 # Skip collections that don't exist or can't be accessed
#                 continue
#
#         return jsonify({"message": "Document deleted successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['GET'])
# def get_document(file_id):
#     """Retrieve a document based on file_id from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get(ids=[file_id])
#
#         if not result["ids"]:
#             return jsonify({"error": "Document not found"}), 404
#
#         document = {
#             "file_id": result["ids"][0],
#             "metadata": result["metadatas"][0],
#             "content": result["documents"][0]
#         }
#
#         return jsonify(document), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


# from flask import Flask, request, jsonify
# import chromadb
# from llama_index.core import Document, VectorStoreIndex, StorageContext
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import uuid
#
# app = Flask(__name__)
#
# # Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
#
# # Initialize embedding model
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# # Common collection name for all files
# COMMON_COLLECTION = "all_files"
#
# # Toggle between LlamaIndex and direct ChromaDB storage
# USE_LLAMA_INDEX = True  # Set to False to use direct ChromaDB storage
#
#
# def get_or_create_collection(collection_name):
#     """Get an existing collection or create a new one if it doesn't exist."""
#     try:
#         return chroma_client.get_collection(name=collection_name)
#     except:
#         return chroma_client.create_collection(name=collection_name)
#
#
# @app.route('/api/documents', methods=['POST'])
# def create_document():
#     """Create a new document in ChromaDB collections."""
#     try:
#         data = request.get_json()
#         print("Received JSON:", data)
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#         file_id = metadata.get("file_id")
#         print(f"loai_phieu: {loai_phieu}, file_id: {file_id}")
#
#         if not file_id:
#             return jsonify({"error": "file_id is required in metadata"}), 400
#
#         # Get or create collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         print(f"Collections created: {loai_phieu}, {COMMON_COLLECTION}")
#
#         if USE_LLAMA_INDEX:
#             # Create a LlamaIndex Document
#             document = Document(
#                 text=content,
#                 metadata=metadata,
#                 id_=file_id
#             )
#             print("Document created:", document.id_)
#
#             # Store in specific collection
#             vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#             storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#             index_specific = VectorStoreIndex.from_documents(
#                 [document], storage_context=storage_context_specific, embed_model=embed_model
#             )
#             print(f"Stored in {loai_phieu} collection")
#
#             # Store in common collection
#             vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#             storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#             index_common = VectorStoreIndex.from_documents(
#                 [document], storage_context=storage_context_common, embed_model=embed_model
#             )
#             print(f"Stored in {COMMON_COLLECTION} collection")
#         else:
#             # Store directly in ChromaDB
#             specific_collection.add(
#                 ids=[file_id],
#                 documents=[content],
#                 metadatas=[metadata]
#             )
#             print(f"Stored directly in {loai_phieu} collection")
#
#             common_collection.add(
#                 ids=[file_id],
#                 documents=[content],
#                 metadatas=[metadata]
#             )
#             print(f"Stored directly in {COMMON_COLLECTION} collection")
#
#         # Verify storage
#         result = common_collection.get(ids=[file_id])
#         if result["ids"]:
#             print("Document verified in all_files:", result["ids"])
#         else:
#             print("Document not found in all_files after storage")
#
#         return jsonify({"message": "Document created successfully", "file_id": file_id}), 201
#
#     except Exception as e:
#         print(f"Error in create_document: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['PUT'])
# def update_document(file_id):
#     """Update a document based on file_id."""
#     try:
#         data = request.get_json()
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#
#         # Get collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#
#         # Delete existing document from all collections
#         collection_names = chroma_client.list_collections()
#         for collection_name in collection_names:
#             try:
#                 collection = chroma_client.get_collection(name=collection_name)
#                 collection.delete(ids=[file_id])
#             except Exception:
#                 continue
#
#         # Create new document
#         document = Document(
#             text=content,
#             metadata=metadata,
#             id_=file_id
#         )
#
#         # Store in specific collection
#         vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#         storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#         index_specific = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_specific, embed_model=embed_model
#         )
#
#         # Store in common collection
#         vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#         storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#         index_common = VectorStoreIndex.from_documents(
#             [document], storage_context=storage_context_common, embed_model=embed_model
#         )
#
#         return jsonify({"message": "Document updated successfully", "file_id": file_id}), 200
#
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['DELETE'])
# def delete_document(file_id):
#     """Delete a document based on file_id."""
#     try:
#         # Get list of collection names
#         collection_names = chroma_client.list_collections()
#
#         # Iterate through collection names and delete from each
#         for collection_name in collection_names:
#             try:
#                 collection = chroma_client.get_collection(name=collection_name)
#                 collection.delete(ids=[file_id])
#             except Exception as e:
#                 # Skip collections that don't exist or can't be accessed
#                 continue
#
#         return jsonify({"message": "Document deleted successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents/<file_id>', methods=['GET'])
# def get_document(file_id):
#     """Retrieve a document based on file_id from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get(ids=[file_id])
#
#         if not result["ids"]:
#             return jsonify({"error": "Document not found"}), 404
#
#         document = {
#             "file_id": result["ids"][0],
#             "metadata": result["metadatas"][0],
#             "content": result["documents"][0]
#         }
#
#         return jsonify(document), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# @app.route('/api/documents', methods=['GET'])
# def get_documents():
#     """Retrieve a document based on file_id from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get()  # Get all documents
#
#         documents = []
#         for idx in range(len(result["ids"])):
#             documents.append({
#                 "file_id": result["ids"][idx],
#                 "metadata": result["metadatas"][idx],
#                 "content": result["documents"][idx]
#             })
#
#         return jsonify(documents), 200
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)



# from flask import Flask, request, jsonify
# import chromadb
# from llama_index.core import Document, VectorStoreIndex, StorageContext
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# import uuid
#
# app = Flask(__name__)
#
# # Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
#
# # Initialize embedding model
# embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# # Common collection name for all files
# COMMON_COLLECTION = "all_files"
#
# # Toggle between LlamaIndex and direct ChromaDB storage
# USE_LLAMA_INDEX = True  # Set to False to use direct ChromaDB storage for testing
#
# def get_or_create_collection(collection_name):
#     """Get an existing collection or create a new one if it doesn't exist."""
#     try:
#         collection = chroma_client.get_collection(name=collection_name)
#         print(f"Retrieved existing collection: {collection_name}")
#         return collection
#     except:
#         collection = chroma_client.create_collection(name=collection_name)
#         print(f"Created new collection: {collection_name}")
#         return collection
#
# @app.route('/api/documents', methods=['POST'])
# def create_document():
#     """Create a new document in ChromaDB collections."""
#     try:
#         data = request.get_json()
#         print("Received JSON:", data)
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#         file_id = metadata.get("file_id")
#         print(f"loai_phieu: {loai_phieu}, file_id: {file_id}")
#
#         if not file_id:
#             return jsonify({"error": "file_id is required in metadata"}), 400
#
#         # Get or create collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#
#         if USE_LLAMA_INDEX:
#             # Create a LlamaIndex Document
#             document = Document(
#                 text=content,
#                 metadata=metadata,
#                 id_=file_id
#             )
#             print("Document created:", document.id_)
#
#             # Store in specific collection
#             vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#             storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#             index_specific = VectorStoreIndex.from_documents(
#                 [document], storage_context=storage_context_specific, embed_model=embed_model
#             )
#             print(f"Stored in {loai_phieu} collection via LlamaIndex")
#
#             # Store in common collection
#             vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#             storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#             index_common = VectorStoreIndex.from_documents(
#                 [document], storage_context=storage_context_common, embed_model=embed_model
#             )
#             print(f"Stored in {COMMON_COLLECTION} collection via LlamaIndex")
#         else:
#             # Store directly in ChromaDB
#             specific_collection.add(
#                 ids=[file_id],
#                 documents=[content],
#                 metadatas=[metadata],
#                 embeddings=embed_model.embed_query(content) if embed_model else None
#             )
#             print(f"Stored directly in {loai_phieu} collection")
#
#             common_collection.add(
#                 ids=[file_id],
#                 documents=[content],
#                 metadatas=[metadata],
#                 embeddings=embed_model.embed_query(content) if embed_model else None
#             )
#             print(f"Stored directly in {COMMON_COLLECTION} collection")
#
#         # Verify storage in all_files
#         result = common_collection.get(ids=[file_id])
#         if result["ids"]:
#             print("Document verified in all_files:", result["ids"])
#         else:
#             print("Document not found in all_files after storage")
#             return jsonify({"error": "Failed to store document in all_files"}), 500
#
#         return jsonify({"message": "Document created successfully", "file_id": file_id}), 201
#
#     except Exception as e:
#         print(f"Error in create_document: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# # @app.route('/api/documents/<file_id>', methods=['PUT'])
# # def update_document(file_id):
# #     """Update a document based on file_id."""
# #     try:
# #         data = request.get_json()
# #         print("Received JSON for PUT:", data)
# #         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
# #             return jsonify({"error": "Invalid JSON structure"}), 400
# #
# #         loai_phieu = data["loai_phieu"]
# #         metadata = data["metadata"]
# #         content = data["content"]
# #         new_file_id = metadata.get("file_id", file_id)
# #         print(f"Updating document: {file_id}, loai_phieu: {loai_phieu}")
# #
# #         # Get collections
# #         specific_collection = get_or_create_collection(loai_phieu)
# #         common_collection = get_or_create_collection(COMMON_COLLECTION)
# #
# #         # Delete existing document from all collections
# #         collection_names = [name for name in chroma_client.list_collections()]
# #         print(f"Collections to check for deletion: {collection_names}")
# #         deleted_collections = []
# #         for collection_name in collection_names:
# #             try:
# #                 collection = chroma_client.get_collection(name=collection_name)
# #                 collection.delete(ids=[file_id])
# #                 deleted_collections.append(collection_name)
# #                 print(f"Deleted {file_id} from {collection_name}")
# #             except Exception as e:
# #                 print(f"Failed to delete {file_id} from {collection_name}: {str(e)}")
# #                 continue
# #         print(f"Successfully deleted from collections: {deleted_collections}")
# #
# #         # Create new document
# #         document = Document(
# #             text=content,
# #             metadata=metadata,
# #             id_=new_file_id
# #         )
# #         print(f"New document created for PUT: {new_file_id}")
# #
# #         if USE_LLAMA_INDEX:
# #             # Store in specific collection
# #             vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
# #             storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
# #             index_specific = VectorStoreIndex.from_documents(
# #                 [document], storage_context=storage_context_specific, embed_model=embed_model
# #             )
# #             print(f"Stored updated document in {loai_phieu} collection via LlamaIndex")
# #
# #             # Store in common collection
# #             vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
# #             storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
# #             index_common = VectorStoreIndex.from_documents(
# #                 [document], storage_context=storage_context_common, embed_model=embed_model
# #             )
# #             print(f"Stored updated document in {COMMON_COLLECTION} collection via LlamaIndex")
# #         else:
# #             # Store directly in ChromaDB
# #             specific_collection.add(
# #                 ids=[new_file_id],
# #                 documents=[content],
# #                 metadatas=[metadata],
# #                 embeddings=embed_model.embed_query(content) if embed_model else None
# #             )
# #             print(f"Stored updated document directly in {loai_phieu} collection")
# #
# #             common_collection.add(
# #                 ids=[new_file_id],
# #                 documents=[content],
# #                 metadatas=[metadata],
# #                 embeddings=embed_model.embed_query(content) if embed_model else None
# #             )
# #             print(f"Stored updated document directly in {COMMON_COLLECTION} collection")
# #
# #         # Verify storage
# #         result = common_collection.get(ids=[new_file_id])
# #         if result["ids"]:
# #             print("Updated document verified in all_files:", result["ids"])
# #         else:
# #             print("Updated document not found in all_files after storage")
# #             return jsonify({"error": "Failed to store updated document in all_files"}), 500
# #
# #         return jsonify({"message": "Document updated successfully", "file_id": new_file_id}), 200
# #
# #     except Exception as e:
# #         print(f"Error in update_document: {str(e)}")
# #         return jsonify({"error": str(e)}), 500
#
# @app.route('/api/documents/<file_id>', methods=['PUT'])
# def update_document(file_id):
#     """Update a document based on file_id."""
#     try:
#         data = request.get_json()
#         print("Received JSON for PUT:", data)
#         if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
#             return jsonify({"error": "Invalid JSON structure"}), 400
#
#         loai_phieu = data["loai_phieu"]
#         metadata = data["metadata"]
#         content = data["content"]
#         new_file_id = metadata.get("file_id", file_id)
#         if new_file_id != file_id:
#             return jsonify({"error": "file_id in metadata must match the provided file_id"}), 400
#         print(f"Updating document: {file_id}, loai_phieu: {loai_phieu}")
#
#         # Get collections
#         specific_collection = get_or_create_collection(loai_phieu)
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#
#         # Delete existing document from specific and common collections
#         try:
#             specific_collection.delete(ids=[file_id])
#             print(f"Deleted {file_id} from {loai_phieu}")
#         except Exception as e:
#             print(f"Failed to delete {file_id} from {loai_phieu}: {str(e)}")
#
#         try:
#             common_collection.delete(ids=[file_id])
#             print(f"Deleted {file_id} from {COMMON_COLLECTION}")
#         except Exception as e:
#             print(f"Failed to delete {file_id} from {COMMON_COLLECTION}: {str(e)}")
#
#         # Create new document
#         document = Document(
#             text=content,
#             metadata=metadata,
#             id_=file_id
#         )
#         print(f"New document created for PUT: {file_id}")
#
#         if USE_LLAMA_INDEX:
#             # Store in specific collection
#             vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
#             storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
#             index_specific = VectorStoreIndex.from_documents(
#                 [document], storage_context=storage_context_specific, embed_model=embed_model
#             )
#             print(f"Stored updated document in {loai_phieu} collection via LlamaIndex")
#
#             # Store in common collection
#             vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
#             storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
#             index_common = VectorStoreIndex.from_documents(
#                 [document], storage_context=storage_context_common, embed_model=embed_model
#             )
#             print(f"Stored updated document in {COMMON_COLLECTION} collection via LlamaIndex")
#         else:
#             # Store directly in ChromaDB
#             specific_collection.add(
#                 ids=[file_id],
#                 documents=[content],
#                 metadatas=[metadata],
#                 embeddings=embed_model.embed_query(content) if embed_model else None
#             )
#             print(f"Stored updated document directly in {loai_phieu} collection")
#
#             common_collection.add(
#                 ids=[file_id],
#                 documents=[content],
#                 metadatas=[metadata],
#                 embeddings=embed_model.embed_query(content) if embed_model else None
#             )
#             print(f"Stored updated document directly in {COMMON_COLLECTION} collection")
#
#         # Verify storage
#         result = common_collection.get(ids=[file_id])
#         if result["ids"]:
#             print("Updated document verified in all_files:", result["ids"])
#         else:
#             print("Updated document not found in all_files after storage")
#             return jsonify({"error": "Failed to store updated document in all_files"}), 500
#
#         return jsonify({"message": "Document updated successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         print(f"Error in update_document: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# # @app.route('/api/documents/<file_id>', methods=['DELETE'])
# # def delete_document(file_id):
# #     """Delete a document based on file_id."""
# #     try:
# #         # Get list of collection names
# #         collection_names = [name for name in chroma_client.list_collections()]
# #         print(f"Collections to check for deletion: {collection_names}")
# #         deleted_collections = []
# #
# #         # Iterate through collection names and delete from each
# #         for collection_name in collection_names:
# #             try:
# #                 collection = chroma_client.get_collection(name=collection_name)
# #                 # Delete by ID
# #                 collection.delete(ids=[file_id])
# #                 deleted_collections.append(collection_name)
# #                 print(f"Deleted {file_id} from {collection_name}")
# #             except Exception as e:
# #                 print(f"Failed to delete {file_id} from {collection_name}: {str(e)}")
# #                 continue
# #
# #         print(f"Successfully deleted from collections: {deleted_collections}")
# #
# #         # Verify deletion
# #         common_collection = get_or_create_collection(COMMON_COLLECTION)
# #         result = common_collection.get(ids=[file_id])
# #         if result["ids"]:
# #             print(f"Warning: Document {file_id} still exists in all_files after deletion")
# #             return jsonify({"error": "Failed to delete document from all_files"}), 500
# #
# #         return jsonify({"message": "Document deleted successfully", "file_id": file_id}), 200
# #
# #     except Exception as e:
# #         print(f"Error in delete_document: {str(e)}")
# #         return jsonify({"error": str(e)}), 500
#
# @app.route('/api/documents/<file_id>', methods=['DELETE'])
# def delete_document(file_id):
#     """Delete a document based on file_id."""
#     try:
#         # Get common collection to retrieve metadata
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get(ids=[file_id])
#         if not result["ids"]:
#             print(f"Document {file_id} not found in all_files")
#             return jsonify({"error": "Document not found"}), 404
#
#         # Get loai_phieu from metadata
#         loai_phieu = result["metadatas"][0].get("loai")
#         if not loai_phieu:
#             print(f"No loai_phieu found in metadata for {file_id}")
#             return jsonify({"error": "Invalid metadata: loai_phieu not found"}), 400
#
#         # Get specific collection
#         specific_collection = get_or_create_collection(loai_phieu)
#
#         # Delete from specific and common collections
#         try:
#             specific_collection.delete(ids=[file_id])
#             print(f"Deleted {file_id} from {loai_phieu}")
#         except Exception as e:
#             print(f"Failed to delete {file_id} from {loai_phieu}: {str(e)}")
#
#         try:
#             common_collection.delete(ids=[file_id])
#             print(f"Deleted {file_id} from {COMMON_COLLECTION}")
#         except Exception as e:
#             print(f"Failed to delete {file_id} from {COMMON_COLLECTION}: {str(e)}")
#
#         # Verify deletion
#         result = common_collection.get(ids=[file_id])
#         if result["ids"]:
#             print(f"Warning: Document {file_id} still exists in all_files after deletion")
#             return jsonify({"error": "Failed to delete document from all_files"}), 500
#
#         return jsonify({"message": "Document deleted successfully", "file_id": file_id}), 200
#
#     except Exception as e:
#         print(f"Error in delete_document: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# # @app.route('/api/documents/<file_id>', methods=['GET'])
# # def get_document(file_id):
# #     """Retrieve a document based on file_id from the common collection."""
# #     try:
# #         common_collection = get_or_create_collection(COMMON_COLLECTION)
# #         result = common_collection.get(ids=[file_id])
# #         print(f"Retrieving document {file_id} from all_files: {result['ids']}")
# #
# #         if not result["ids"]:
# #             print(f"Document {file_id} not found in all_files")
# #             return jsonify({"error": "Document not found"}), 404
# #
# #         document = {
# #             "file_id": result["ids"][0],
# #             "metadata": result["metadatas"][0],
# #             "content": result["documents"][0]
# #         }
# #         print(f"Retrieved document: {document}")
# #
# #         return jsonify(document), 200
# #
# #     except Exception as e:
# #         print(f"Error in get_document: {str(e)}")
# #         return jsonify({"error": str(e)}), 500
#
# @app.route('/api/documents/<file_id>', methods=['GET'])
# def get_document(file_id):
#     """Retrieve a document based on file_id from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get(ids=[file_id])
#         print(f"Retrieving document {file_id} from all_files: {result['ids']}")
#
#         if not result["ids"]:
#             print(f"Document {file_id} not found in all_files")
#             return jsonify({"error": "Document not found"}), 404
#
#         document = {
#             "file_id": result["ids"][0],
#             "metadata": result["metadatas"][0],
#             "content": result["documents"][0]
#         }
#         print(f"Retrieved document: {document}")
#
#         return jsonify(document), 200
#
#     except Exception as e:
#         print(f"Error in get_document: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/api/documents', methods=['GET'])
# def get_documents():
#     """Retrieve all documents from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get()  # Get all documents
#         print(f"Retrieving all documents from all_files: {result['ids']}")
#
#         documents = []
#         for idx in range(len(result["ids"])):
#             documents.append({
#                 "file_id": result["ids"][idx],
#                 "metadata": result["metadatas"][idx],
#                 "content": result["documents"][idx]
#             })
#         print(f"Retrieved {len(documents)} documents")
#
#         return jsonify(documents), 200
#
#     except Exception as e:
#         print(f"Error in get_documents: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# @app.route('/api/documents', methods=['GET'])
# def get_documents():
#     """Retrieve all documents from the common collection."""
#     try:
#         common_collection = get_or_create_collection(COMMON_COLLECTION)
#         result = common_collection.get()  # Get all documents
#         print(f"Retrieving all documents from all_files: {result['ids']}")
#
#         documents = []
#         for idx in range(len(result["ids"])):
#             documents.append({
#                 "file_id": result["ids"][idx],
#                 "metadata": result["metadatas"][idx],
#                 "content": result["documents"][idx]
#             })
#         print(f"Retrieved {len(documents)} documents")
#
#         return jsonify(documents), 200
#
#     except Exception as e:
#         print(f"Error in get_documents: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time

app = Flask(__name__)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Common collection name for all files
COMMON_COLLECTION = "all_files"

# Toggle between LlamaIndex and direct ChromaDB storage
USE_LLAMA_INDEX = True

def get_or_create_collection(collection_name):
    """Get an existing collection or create a new one if it doesn't exist."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
        return collection
    except Exception as e:
        print(f"Error retrieving collection {collection_name}: {str(e)}")
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
        return collection

@app.route('/api/documents', methods=['POST'])
def create_document():
    """Create a new document in ChromaDB collections."""
    try:
        data = request.get_json()
        print("Received JSON:", data)
        if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
            return jsonify({"error": "Invalid JSON structure"}), 400

        loai_phieu = data["loai_phieu"]
        metadata = data["metadata"]
        content = data["content"]
        file_id = metadata.get("file_id")
        print(f"loai_phieu: {loai_phieu}, file_id: {file_id}")

        if not file_id:
            return jsonify({"error": "file_id is required in metadata"}), 400

        # Get or create collections
        specific_collection = get_or_create_collection(loai_phieu)
        common_collection = get_or_create_collection(COMMON_COLLECTION)

        if USE_LLAMA_INDEX:
            # Create a LlamaIndex Document
            document = Document(
                text=content,
                metadata=metadata,
                id_=file_id
            )
            print("Document created:", document.id_)

            # Store in specific collection
            vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
            storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
            index_specific = VectorStoreIndex.from_documents(
                [document], storage_context=storage_context_specific, embed_model=embed_model
            )
            print(f"Stored in {loai_phieu} collection via LlamaIndex")
            # Immediate check
            temp_result = specific_collection.get(ids=[file_id])
            print(f"Immediate check in {loai_phieu}: {temp_result}")

            # Store in common collection
            vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
            storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
            index_common = VectorStoreIndex.from_documents(
                [document], storage_context=storage_context_common, embed_model=embed_model
            )
            print(f"Stored in {COMMON_COLLECTION} collection via LlamaIndex")
            # Immediate check
            temp_result = common_collection.get(ids=[file_id])
            print(f"Immediate check in {COMMON_COLLECTION}: {temp_result}")
        else:
            # Store directly in ChromaDB
            specific_collection.add(
                ids=[file_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=embed_model.embed_query(content) if embed_model else None
            )
            print(f"Stored directly in {loai_phieu} collection")
            temp_result = specific_collection.get(ids=[file_id])
            print(f"Immediate check in {loai_phieu}: {temp_result}")

            common_collection.add(
                ids=[file_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=embed_model.embed_query(content) if embed_model else None
            )
            print(f"Stored directly in {COMMON_COLLECTION} collection")
            temp_result = common_collection.get(ids=[file_id])
            print(f"Immediate check in {COMMON_COLLECTION}: {temp_result}")

        # Verify storage in all_files with retry
        for attempt in range(3):
            time.sleep(1)  # Wait 1 second before checking
            result = common_collection.get(ids=[file_id])
            if result["ids"]:
                print("Document verified in all_files:", result["ids"])
                break
            print(f"Attempt {attempt + 1}: Document not found in all_files")
        else:
            print("Document not found in all_files after retries")
            # Log additional debug info
            specific_result = specific_collection.get(ids=[file_id])
            print(f"Specific collection {loai_phieu} result: {specific_result}")
            return jsonify({
                "error": "Failed to store document in all_files",
                "debug": {
                    "specific_collection_result": specific_result,
                    "file_id": file_id,
                    "loai_phieu": loai_phieu
                }
            }), 500

        return jsonify({"message": "Document created successfully", "file_id": file_id}), 201

    except Exception as e:
        print(f"Error in create_document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<file_id>', methods=['PUT'])
def update_document(file_id):
    """Update a document based on file_id."""
    try:
        data = request.get_json()
        print("Received JSON for PUT:", data)
        if not data or not all(key in data for key in ["loai_phieu", "metadata", "content"]):
            return jsonify({"error": "Invalid JSON structure"}), 400

        loai_phieu = data["loai_phieu"]
        metadata = data["metadata"]
        content = data["content"]
        new_file_id = metadata.get("file_id", file_id)
        if new_file_id != file_id:
            return jsonify({"error": "file_id in metadata must match the provided file_id"}), 400
        print(f"Updating document: {file_id}, loai_phieu: {loai_phieu}")

        # Get collections
        specific_collection = get_or_create_collection(loai_phieu)
        common_collection = get_or_create_collection(COMMON_COLLECTION)

        # Delete existing document from specific and common collections
        try:
            specific_collection.delete(ids=[file_id])
            print(f"Deleted {file_id} from {loai_phieu}")
        except Exception as e:
            print(f"Failed to delete {file_id} from {loai_phieu}: {str(e)}")

        try:
            common_collection.delete(ids=[file_id])
            print(f"Deleted {file_id} from {COMMON_COLLECTION}")
        except Exception as e:
            print(f"Failed to delete {file_id} from {COMMON_COLLECTION}: {str(e)}")

        # Create new document
        document = Document(
            text=content,
            metadata=metadata,
            id_=file_id
        )
        print(f"New document created for PUT: {file_id}")

        if USE_LLAMA_INDEX:
            # Store in specific collection
            vector_store_specific = ChromaVectorStore(chroma_collection=specific_collection)
            storage_context_specific = StorageContext.from_defaults(vector_store=vector_store_specific)
            index_specific = VectorStoreIndex.from_documents(
                [document], storage_context=storage_context_specific, embed_model=embed_model
            )
            print(f"Stored updated document in {loai_phieu} collection via LlamaIndex")
            temp_result = specific_collection.get(ids=[file_id])
            print(f"Immediate check in {loai_phieu}: {temp_result}")

            # Store in common collection
            vector_store_common = ChromaVectorStore(chroma_collection=common_collection)
            storage_context_common = StorageContext.from_defaults(vector_store=vector_store_common)
            index_common = VectorStoreIndex.from_documents(
                [document], storage_context=storage_context_common, embed_model=embed_model
            )
            print(f"Stored updated document in {COMMON_COLLECTION} collection via LlamaIndex")
            temp_result = common_collection.get(ids=[file_id])
            print(f"Immediate check in {COMMON_COLLECTION}: {temp_result}")
        else:
            # Store directly in ChromaDB
            specific_collection.add(
                ids=[file_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=embed_model.embed_query(content)
            )
            print(f"Stored updated document directly in {loai_phieu} collection")
            temp_result = specific_collection.get(ids=[file_id])
            print(f"Immediate check in {loai_phieu}: {temp_result}")

            common_collection.add(
                ids=[file_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=embed_model.embed_query(content) if embed_model else None
            )
            print(f"Stored updated document directly in {COMMON_COLLECTION} collection")
            temp_result = common_collection.get(ids=[file_id])
            print(f"Immediate check in {COMMON_COLLECTION}: {temp_result}")

        # Verify storage in all_files with retry
        for attempt in range(3):
            time.sleep(1)  # Wait 1 second before checking
            result = common_collection.get(ids=[file_id])
            if result["ids"]:
                print("Updated document verified in all_files:", result["ids"])
                break
            print(f"Attempt {attempt + 1}: Updated document not found in all_files")
        else:
            print("Updated document not found in all_files after retries")
            # Log additional debug info
            specific_result = specific_collection.get(ids=[file_id])
            print(f"Specific collection {loai_phieu} result: {specific_result}")
            return jsonify({
                "error": "Failed to store updated document in all_files",
                "debug": {
                    "specific_collection_result": specific_result,
                    "file_id": file_id,
                    "loai_phieu": loai_phieu
                }
            }), 500

        return jsonify({"message": "Document updated successfully", "file_id": file_id}), 200

    except Exception as e:
        print(f"Error in update_document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<file_id>', methods=['DELETE'])
def delete_document(file_id):
    """Delete a document based on file_id."""
    try:
        # Get common collection to retrieve metadata
        common_collection = get_or_create_collection(COMMON_COLLECTION)
        result = common_collection.get(ids=[file_id])
        if not result["ids"]:
            print(f"Document {file_id} not found in all_files")
            return jsonify({"error": "Document not found"}), 404

        # Get loai_phieu from metadata
        loai_phieu = result["metadatas"][0].get("loai")
        if not loai_phieu:
            print(f"No loai_phieu found in metadata for {file_id}")
            return jsonify({"error": "Invalid metadata: loai_phieu not found"}), 400

        # Get specific collection
        specific_collection = get_or_create_collection(loai_phieu)

        # Delete from specific and common collections
        try:
            specific_collection.delete(ids=[file_id])
            print(f"Deleted {file_id} from {loai_phieu}")
        except Exception as e:
            print(f"Failed to delete {file_id} from {loai_phieu}: {str(e)}")

        try:
            common_collection.delete(ids=[file_id])
            print(f"Deleted {file_id} from {COMMON_COLLECTION}")
        except Exception as e:
            print(f"Failed to delete {file_id} from {COMMON_COLLECTION}: {str(e)}")

        # Verify deletion
        result = common_collection.get(ids=[file_id])
        if result["ids"]:
            print(f"Warning: Document {file_id} still exists in all_files after deletion")
            return jsonify({"error": "Failed to delete document from all_files"}), 500

        return jsonify({"message": "Document deleted successfully", "file_id": file_id}), 200

    except Exception as e:
        print(f"Error in delete_document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<file_id>', methods=['GET'])
def get_document(file_id):
    """Retrieve a document based on file_id from the common collection."""
    try:
        common_collection = get_or_create_collection(COMMON_COLLECTION)
        result = common_collection.get(ids=[file_id])
        print(f"Retrieving document {file_id} from all_files: {result['ids']}")

        if not result["ids"]:
            print(f"Document {file_id} not found in all_files")
            return jsonify({"error": "Document not found"}), 404

        document = {
            "file_id": result["ids"][0],
            "metadata": result["metadatas"][0],
            "content": result["documents"][0]
        }
        print(f"Retrieved document: {document}")

        return jsonify(document), 200

    except Exception as e:
        print(f"Error in get_document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Retrieve all documents from the common collection."""
    try:
        common_collection = get_or_create_collection(COMMON_COLLECTION)
        result = common_collection.get()  # Get all documents
        print(f"Retrieving all documents from all_files: {result['ids']}")

        documents = []
        for idx in range(len(result["ids"])):
            documents.append({
                "file_id": result["ids"][idx],
                "metadata": result["metadatas"][idx],
                "content": result["documents"][idx]
            })
        print(f"Retrieved {len(documents)} documents")

        return jsonify(documents), 200

    except Exception as e:
        print(f"Error in get_documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)