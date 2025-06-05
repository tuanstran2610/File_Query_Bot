import os
from typing import List, Dict, Any
from langchain.llms import LlamaCpp  # Giả định sử dụng Llama 3.1
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
import langchain
from datetime import datetime
import re
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kích hoạt caching
langchain.llm_cache = InMemoryCache()

# Khởi tạo embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Khởi tạo Chroma DB client (giả định đã có API hoặc client)
# Thay thế bằng cấu hình thực tế của bạn
chroma_client = Chroma(
    collection_name="company_documents",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"  # Thư mục lưu trữ Chroma DB
)

# Khởi tạo LLM (Llama 3.1)
llm = LlamaCpp(
    model_path="/path/to/llama-3.1/model",  # Thay bằng đường dẫn thực tế
    n_ctx=8192,  # Context length
    temperature=0.3,  # Giảm temperature để tăng độ chính xác
    max_tokens=1000,
    verbose=False
)

# Prompt template để phân loại intent từ user input
INTENT_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
    Phân loại ý định của người dùng từ câu sau: "{user_input}"
    Trả về định dạng JSON:
    {
      "intent": "search|tóm tắt|tổng hợp",
      "filters": {
        "ma_bieu_mau": "str|None",
        "ngay_kiem_tra": "YYYY-MM-DD|None",
        "loai_khu_vuc": "str|None",
        "vi_tri": "str|None",
        "nguoi_nhap": "str|None"
      },
      "query": "str"
    }
    Ví dụ: Nếu input là "Tìm báo cáo kiểm tra an toàn tại khu vực A5 vào tháng 4/2025",
    trả về:
    {
      "intent": "search",
      "filters": {
        "ma_bieu_mau": null,
        "ngay_kiem_tra": "2025-04",
        "loai_khu_vuc": null,
        "vi_tri": "Khu vực A5",
        "nguoi_nhap": null
      },
      "query": "báo cáo kiểm tra an toàn"
    }
    """
)

# Prompt template để tổng hợp/tóm tắt kết quả
RESPONSE_PROMPT = PromptTemplate(
    input_variables=["user_input", "documents", "intent"],
    template="""
    Bạn là trợ lý AI hỗ trợ doanh nghiệp. Dựa trên yêu cầu của người dùng: "{user_input}",
    và các tài liệu sau: {documents},
    hãy thực hiện hành động theo ý định: {intent}.
    - Nếu intent là "search": Trả về thông tin chi tiết từ các tài liệu phù hợp.
    - Nếu intent là "tóm tắt": Tóm tắt nội dung các tài liệu trong tối đa 100 từ.
    - Nếu intent là "tổng hợp": Tổng hợp thông tin từ các tài liệu thành một báo cáo ngắn gọn.
    Trả về câu trả lời bằng ngôn ngữ tự nhiên, thân thiện, sử dụng tiếng Việt.
    """
)


# Hàm trích xuất intent và filters từ user input
def extract_intent_and_filters(user_input: str) -> Dict[str, Any]:
    intent_chain = LLMChain(llm=llm, prompt=INTENT_PROMPT)
    try:
        result = intent_chain.run(user_input=user_input)
        return eval(result)  # Chuyển đổi string JSON thành dict
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất intent: {e}")
        return {
            "intent": "search",
            "filters": {},
            "query": user_input
        }


# Hàm query Chroma DB
def query_chroma_db(user_input: str) -> str:
    """
    Hàm query Chroma DB dựa trên user input, trả về câu trả lời ngôn ngữ tự nhiên.

    Args:
        user_input (str): Câu hỏi hoặc yêu cầu từ người dùng.

    Returns:
        str: Câu trả lời ngôn ngữ tự nhiên.
    """
    logger.info(f"Nhận user input: {user_input}")

    # Bước 1: Trích xuất intent và filters
    intent_data = extract_intent_and_filters(user_input)
    intent = intent_data["intent"]
    filters = intent_data["filters"]
    query = intent_data["query"]

    # Bước 2: Xây dựng bộ lọc metadata
    metadata_filter = {}
    for key, value in filters.items():
        if value and value != "None":
            if key == "ngay_kiem_tra" and len(value) == 7:  # YYYY-MM
                # Lọc theo tháng
                metadata_filter[key] = {"$regex": f"^{value}"}
            else:
                metadata_filter[key] = value

    # Bước 3: Query Chroma DB
    try:
        # Hybrid search: Kết hợp metadata filter và semantic search
        if metadata_filter:
            # Lọc trước bằng metadata
            docs = chroma_client.search(
                query=query,
                search_type="similarity",
                filter=metadata_filter,
                k=10  # Lấy tối đa 10 tài liệu
            )
        else:
            # Chỉ tìm kiếm ngữ nghĩa nếu không có filter
            docs = chroma_client.search(
                query=query,
                search_type="similarity",
                k=10
            )

        # Bước 4: Lọc và xếp hạng tài liệu
        if not docs:
            return "Không tìm thấy tài liệu phù hợp với yêu cầu của bạn."

        # Xếp hạng tài liệu dựa trên độ tương đồng
        ranked_docs = sorted(docs, key=lambda x: x.metadata.get("score", 0), reverse=True)

        # Giới hạn số lượng tài liệu gửi cho LLM
        max_docs = 5
        selected_docs = ranked_docs[:max_docs]

        # Chuẩn bị nội dung tài liệu
        documents_content = "\n".join([
            f"Tài liệu {i + 1}: {doc.page_content}\nMetadata: {doc.metadata}"
            for i, doc in enumerate(selected_docs)
        ])

        # Bước 5: Gửi cho LLM để sinh câu trả lời
        response_chain = LLMChain(llm=llm, prompt=RESPONSE_PROMPT)
        with get_openai_callback() as cb:
            response = response_chain.run(
                user_input=user_input,
                documents=documents_content,
                intent=intent
            )
            logger.info(f"Thời gian xử lý LLM: {cb.total_time}")

        return response.strip()

    except Exception as e:
        logger.error(f"Lỗi khi query Chroma DB: {e}")
        return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."


# Ví dụ sử dụng
if __name__ == "__main__":
    user_input = "Tìm các báo cáo kiểm tra an toàn tại khu vực A5 vào tháng 4/2025"
    response = query_chroma_db(user_input)
    print(response)