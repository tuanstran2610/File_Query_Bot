# # from huggingface_hub import snapshot_download
# #
# # snapshot_download(
# #     repo_id="trong269/vit5-vietnamese-text-summarization",
# #     local_dir="vit5_model",  # Folder where the model will be saved
# #     local_dir_use_symlinks=False  # Optional: Avoid symbolic links
# # )
#
#
# # from transformers import pipeline
# # from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
# # from qdrant_client import QdrantClient
# # from langchain_qdrant import Qdrant
# # from langchain.chains import RetrievalQA
# #
# # qdrant_client = QdrantClient(
# #     url="http://localhost:6333",  # default local Qdrant url
# #     prefer_grpc=False
# # )
# # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #
# # vectorstore = Qdrant(
# #     client=qdrant_client,
# #     collection_name="collections",  # replace with your actual Qdrant collection name
# #     embeddings=embedding_model
# # )
# #
# # summarizer = pipeline(
# #     "text2text-generation",
# #     model="trong269/vit5-vietnamese-text-summarization",
# #     max_length=256,  # allow longer summaries
# #     do_sample=True,
# #     temperature=0.7,
# #     top_p=0.9,
# # )
# #
# # llm = HuggingFacePipeline(pipeline=summarizer)
# # qa_chain = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     chain_type="stuff",  # simplest; can try "map_reduce" or "refine" for better summaries
# #     retriever=vectorstore.as_retriever()
# # )
# #
# # print("Chatbot is ready! Type 'exit' to quit.")
# #
# # while True:
# #     query = input("You: ")
# #     if query.lower() == "exit":
# #         break
# #
# #     answer = qa_chain.run(query)
# #     print("Bot:", answer)
#
# # Load model directly
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_qdrant import Qdrant
# # from qdrant_client import QdrantClient
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# #
# # # 1. Thiết lập embedding và Qdrant
# # embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# # qdrant_client = QdrantClient(url="http://localhost:6333")
# #
# # vector_store = Qdrant(
# #     client=qdrant_client,
# #     collection_name="collections",  # ⚠️ Thay bằng tên collection thực tế
# #     embeddings=embedding_model,
# #     content_payload_key="page_content"  # ⚠️ Thay bằng key đúng với payload trong Qdrant
# # )
# #
# # # 2. Load mô hình ViT5
# # model_name = "VietAI/vit5-base"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# #
# # # 3. Vòng lặp để nhập câu hỏi từ người dùng
# # while True:
# #     query = input("\n🟡 Nhập câu hỏi (hoặc gõ 'exit' để thoát): ").strip()
# #     if query.lower() in ['exit', 'quit']:
# #         print("👋 Kết thúc.")
# #         break
# #
# #     # Truy vấn Qdrant
# #     retrieved_docs = vector_store.similarity_search(query, k=5)
# #     context = " ".join([doc.page_content for doc in retrieved_docs if doc.page_content])
# #
# #     if not context:
# #         print("❌ Không tìm thấy nội dung liên quan trong Qdrant.")
# #         continue
# #
# #     # Tóm tắt context
# #     input_text = f"summarize: {context}"
# #     inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
# #     summary_ids = model.generate(
# #         inputs["input_ids"],
# #         max_length=150,
# #         min_length=30,
# #         length_penalty=2.0,
# #         num_beams=4,
# #         early_stopping=True
# #     )
# #     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# #
# #     # In kết quả
# #     print("\n🟩 Câu hỏi:", query)
# #     print("🟦 Tóm tắt:", summary)
#
#
# # import asyncio
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_qdrant import QdrantVectorStore
# # from qdrant_client import QdrantClient
# # from qdrant_client.http.models import Distance, VectorParams
# # from langchain_core.documents import Document
# # from langchain_ollama import OllamaLLM
# #
# #
# # async def setup_qdrant():
# #     try:
# #         # Khởi tạo embedding model
# #         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# #
# #         # Kết nối với Qdrant
# #         qdrant_client = QdrantClient(url="http://localhost:6333", timeout=5)
# #         print("Kết nối với Qdrant thành công!")
# #
# #         # Kiểm tra và tạo collection nếu chưa tồn tại
# #         collection_name = "collections"
# #         collections = qdrant_client.get_collections()
# #         if collection_name not in [c.name for c in collections.collections]:
# #             qdrant_client.create_collection(
# #                 collection_name=collection_name,
# #                 vectors_config=VectorParams(size=384, distance=Distance.COSINE)
# #             )
# #             print(f"Đã tạo collection: {collection_name}")
# #
# #         # Khởi tạo QdrantVectorStore
# #         vector_store = QdrantVectorStore(
# #             client=qdrant_client,
# #             collection_name=collection_name,
# #             embedding=embedding_model,
# #             content_payload_key="text"
# #         )
# #
# #         # Thêm dữ liệu mẫu về Chúa Nhẫn
# #         # sample_docs = [
# #         #     Document(
# #         #         page_content="Chúa Nhẫn (The Lord of the Rings) là bộ ba tiểu thuyết giả tưởng của J.R.R. Tolkien, kể về hành trình của Frodo Baggins, một hobbit, để phá hủy Chiếc Nhẫn Tối Thượng, một vũ khí nguy hiểm của Chúa Tể Hắc Ám Sauron. Câu chuyện bắt đầu ở Shire, nơi Frodo nhận chiếc nhẫn từ chú mình, Bilbo Baggins.",
# #         #         metadata={"id": 1}
# #         #     ),
# #         #     Document(
# #         #         page_content="Frodo cùng với các bạn hobbit Sam, Merry, và Pippin được phù thủy Gandalf hướng dẫn, bắt đầu chuyến đi đến Rivendell. Tại đây, một Hội Nhẫn được thành lập, bao gồm Frodo, Sam, Merry, Pippin, Gandalf, Aragorn, Legolas, Gimli, và Boromir, với nhiệm vụ đưa nhẫn đến Mordor để phá hủy nó trong ngọn núi lửa Mount Doom.",
# #         #         metadata={"id": 2}
# #         #     ),
# #         #     Document(
# #         #         page_content="Hội Nhẫn đối mặt với nhiều nguy hiểm: bị Nazgûl truy đuổi, chiến đấu với lũ orc ở Moria, và sự phản bội của Boromir. Sau khi Gandalf ngã xuống ở Moria, Hội Nhẫn tan rã. Frodo và Sam tiếp tục hành trình một mình, được dẫn đường bởi Gollum, một sinh vật bị ám ảnh bởi chiếc nhẫn.",
# #         #         metadata={"id": 3}
# #         #     ),
# #         #     Document(
# #         #         page_content="Trong khi đó, Aragorn, Legolas, và Gimli giúp vương quốc Rohan Roswell và Gondor chống lại quân đội của Sauron. Sam và Frodo đối mặt với nhiều thử thách ở Mordor, bao gồm sự cám dỗ của chiếc nhẫn và sự phản bội của Gollum. Cuối cùng, Frodo phá hủy chiếc nhẫn tại Mount Doom, đánh bại Sauron.",
# #         #         metadata={"id": 4}
# #         #     ),
# #         #     Document(
# #         #         page_content="Tom Bombadil là một nhân vật bí ẩn giúp Frodo và các hobbit thoát khỏi nguy hiểm ở Old Forest. Ông không bị ảnh hưởng bởi sức mạnh của chiếc nhẫn. Aragorn, một chiến binh dũng cảm, là người thừa kế ngai vàng Gondor và dẫn dắt các dân tộc tự do chống lại Sauron.",
# #         #         metadata={"id": 5}
# #         #     )
# #         # ]
# #         # vector_store.add_documents(sample_docs)
# #         # print("Đã thêm dữ liệu mẫu về Chúa Nhẫn vào Qdrant.")
# #
# #         return vector_store, qdrant_client
# #
# #     except Exception as e:
# #         print(f"Lỗi khi thiết lập Qdrant: {str(e)}")
# #         return None, None
# #
# #
# # async def answer_question_with_llama(vector_store, query):
# #     try:
# #         # Khởi tạo LLM
# #         llm = OllamaLLM(model="llama3.1:8b")
# #
# #         # Tìm kiếm tương đồng
# #         docs = vector_store.similarity_search(query, k=5)
# #         context = " ".join([doc.page_content for doc in docs if doc.page_content])
# #         if not context:
# #             return "Không tìm thấy nội dung liên quan trong Qdrant. Vui lòng cung cấp thêm thông tin hoặc thử câu hỏi khác."
# #
# #         # Prompt cải tiến để trả lời chi tiết
# #         prompt = (
# #             f"Bạn là một trợ lý thông minh, am hiểu về 'Chúa Nhẫn' của J.R.R. Tolkien. Dựa trên nội dung sau, hãy cung cấp câu trả lời chi tiết, đầy đủ, và có cấu trúc rõ ràng (bao gồm các ý chính, diễn biến quan trọng, và các nhân vật liên quan). Nếu câu hỏi yêu cầu tóm tắt chi tiết, hãy liệt kê các sự kiện theo thứ tự thời gian. Trả lời bằng tiếng Việt, sử dụng ngôn ngữ tự nhiên và dễ hiểu.\n\n"
# #             f"Nội dung:\n{context}\n\n"
# #             f"Câu hỏi: {query}\n\n"
# #             f"Trả lời:"
# #         )
# #
# #         # Gọi LLM
# #         answer = await asyncio.to_thread(llm.invoke, prompt)
# #         return answer
# #
# #     except Exception as e:
# #         return f"Lỗi khi trả lời câu hỏi: {str(e)}"
# #
# #
# # async def main():
# #     # Thiết lập Qdrant và vector store
# #     vector_store, qdrant_client = await setup_qdrant()
# #     if not vector_store:
# #         print("Không thể tiếp tục do lỗi thiết lập Qdrant.")
# #         return
# #
# #     # Vòng lặp truy vấn
# #     while True:
# #         query = input("\nNhập câu hỏi (hoặc 'exit' để thoát): ").strip()
# #         if query.lower() in ['exit', 'quit']:
# #             break
# #
# #         result = await answer_question_with_llama(vector_store, query)
# #         print("\n🟦 Trả lời từ LLaMA:\n", result)
# #
# #     # Đóng kết nối Qdrant
# #     if qdrant_client:
# #         qdrant_client.close()
# #
# #
# # if __name__ == "__main__":
# #     asyncio.run(main())
#
#
# import asyncio
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from langchain_core.documents import Document
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import torch
# import sentencepiece  # Ensure SentencePiece is imported to verify installation
#
# async def setup_qdrant():
#     try:
#         # Initialize embedding model
#         embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
#         # Connect to Qdrant
#         qdrant_client = QdrantClient(url="http://localhost:6333", timeout=5)
#         print("Connected to Qdrant successfully!")
#
#         # Check and create collection if it doesn't exist
#         collection_name = "collections"
#         collections = qdrant_client.get_collections()
#         if collection_name not in [c.name for c in collections.collections]:
#             qdrant_client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=VectorParams(size=384, distance=Distance.COSINE)
#             )
#             print(f"Created collection: {collection_name}")
#
#             # Add sample Harry Potter documents for testing
#             vector_store = QdrantVectorStore(
#                 client=qdrant_client,
#                 collection_name=collection_name,
#                 embedding=embedding_model,
#                 content_payload_key="text"
#             )
#
#         else:
#             vector_store = QdrantVectorStore(
#                 client=qdrant_client,
#                 collection_name=collection_name,
#                 embedding=embedding_model,
#                 content_payload_key="text"
#             )
#
#         return vector_store, qdrant_client
#
#     except Exception as e:
#         print(f"Error setting up Qdrant: {str(e)}")
#         return None, None
#
# async def summarize_with_vit5(vector_store, query):
#     try:
#         # Initialize ViT5
#         model_name = "VietAI/vit5-base"
#         try:
#             tokenizer = T5Tokenizer.from_pretrained(model_name)
#             model = T5ForConditionalGeneration.from_pretrained(model_name)
#         except Exception as e:
#             return f"Error loading ViT5 model or tokenizer: {str(e)}. Ensure SentencePiece is installed (`pip install sentencepiece`)."
#
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         print(f"ViT5 model loaded on {device}.")
#
#         # Perform similarity search in Qdrant
#         docs = vector_store.similarity_search(query, k=5)
#         context = " ".join([doc.page_content for doc in docs if doc.page_content])
#         if not context:
#             return "No relevant content found in Qdrant. Please add documents or try a different query."
#
#         # Prompt for summarization
#         prompt = (
#             f"Tóm tắt các diễn biến chính từ nội dung sau thành một đoạn văn ngắn gọn, rõ ràng, theo thứ tự thời gian, sử dụng ngôn ngữ tiếng Việt tự nhiên và dễ hiểu. Tập trung vào các sự kiện quan trọng và nhân vật chính liên quan.\n\n"
#             f"Nội dung: {context}\n\n"
#             f"Tóm tắt:"
#         )
#
#         # Prepare input for ViT5
#         inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
#         inputs = inputs.to(device)
#
#         # Generate summary
#         summary_ids = model.generate(
#             inputs["input_ids"],
#             max_length=150,
#             min_length=50,
#             length_penalty=1.0,
#             num_beams=4,
#             early_stopping=True
#         )
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#
#         return summary
#
#     except Exception as e:
#         return f"Error during ViT5 summarization: {str(e)}"
#
# async def main():
#     # Set up Qdrant and vector store
#     vector_store, qdrant_client = await setup_qdrant()
#     if not vector_store:
#         print("Cannot proceed due to Qdrant setup error.")
#         return
#
#     # Query loop
#     while True:
#         query = input("\nEnter your query (or 'exit' to quit): ").strip()
#         if query.lower() in ['exit', 'quit']:
#             break
#
#         print("\nProcessing query...")
#         result = await summarize_with_vit5(vector_store, query)
#         print("\n🟦 Summary from ViT5:\n", result)
#
#     # Close Qdrant connection
#     if qdrant_client:
#         qdrant_client.close()
#
# if __name__ == "__main__":
#     asyncio.run(main())


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model và tokenizer một lần
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large-vietnews-summarization")
device = "cuda" if model.device.type == "cuda" else "cpu"
model.to(device)

# Vòng lặp nhập input
print("Nhập đoạn văn bản tiếng Việt để tóm tắt (gõ 'exit' để thoát):\n")
while True:
    sentence = input(">> Nhập văn bản: ").strip()
    if sentence.lower() == "exit":
        print("Đã thoát.")
        break
    if not sentence:
        print("❌ Vui lòng không để trống.")
        continue

    # Chuẩn bị đầu vào cho model
    text = "vietnews: " + sentence + " </s>"
    encoding = tokenizer(text, return_tensors="pt").to(device)

    # Sinh tóm tắt
    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_length=256,
        early_stopping=True
    )

    # Hiển thị kết quả
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("\n📝 Tóm tắt:")
    print(summary)
    print("-" * 80)


# import torch
# print(torch.cuda.is_available())
