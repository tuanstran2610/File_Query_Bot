from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient

# 1. Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# 2. Load dense embedding model
dense_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Reconnect to existing Qdrant collection with dense retrieval only
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test_json",
    embedding=dense_embedding,
    retrieval_mode=RetrievalMode.DENSE,
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4. LLM setup
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key="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"
)
#
# 5. Prompt
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an assistant. Use the following context to answer the user's question.
# If the answer is not in the context, say "I don't know".
#
# Context:
# {context}
#
# Question:
# {question}
# """
# )
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Bạn là một trợ lý AI. Sử dụng ngữ cảnh dưới đây để trả lời câu hỏi của người dùng bằng tiếng Việt.
Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy trả lời: "Tôi không biết".

Ngữ cảnh:
{context}

Câu hỏi:
{question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. Input loop
print("Ask your question (type 'exit' to quit):")
while True:
    query = input(">> ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    try:
        answer = chain.invoke(query)
        print("Answer:\n", answer.content, "\n")
    except Exception as e:
        print("Error:", e)





#
# from langchain_qdrant import QdrantVectorStore, RetrievalMode
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from qdrant_client import QdrantClient
#
# # 1. Kết nối tới Qdrant
# client = QdrantClient(host="localhost", port=6333)
#
# # 2. Tải mô hình nhúng (embedding) dense
# dense_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# # 3. Kết nối lại với bộ sưu tập Qdrant hiện có, chỉ dùng tìm kiếm dense
# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="test_json_2",
#     embedding=dense_embedding,
#     retrieval_mode=RetrievalMode.DENSE,
# )
#
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#
# # 4. Thiết lập LLM
# llm = ChatGroq(
#     model="llama3-70b-8192",
#     api_key="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"
# )
#
# # 5. Tạo prompt bằng tiếng Việt
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# Bạn là một trợ lý AI. Sử dụng ngữ cảnh dưới đây để trả lời câu hỏi của người dùng bằng tiếng Việt.
# Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy trả lời: "Tôi không biết".
#
# Ngữ cảnh:
# {context}
#
# Câu hỏi:
# {question}
# """
# )
#
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
#
# # Tạo chuỗi xử lý
# chain = (
#     {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
#     | prompt
#     | llm
# )
#
# # 6. Vòng lặp nhận câu hỏi
# print("Hãy đặt câu hỏi của bạn (gõ 'exit' hoặc 'thoát' để thoát):")
# while True:
#     query = input(">> ")
#     if query.lower() in ["exit", "thoát"]:
#         print("Tạm biệt!")
#         break
#     try:
#         # Gọi chuỗi xử lý và lấy nội dung trả lời
#         answer = chain.invoke(query)
#         print("Trả lời:\n", answer.content, "\n")
#     except Exception as e:
#         print("❌ Lỗi:", e)