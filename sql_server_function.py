import os
from typing import Dict, Any
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
import langchain
from sqlalchemy import create_engine, text
import pandas as pd
import logging
import sqlparse
import re

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kích hoạt caching
langchain.llm_cache = InMemoryCache()

# Connection string từ code của bạn
CONNECTION_STRING = (
    "mssql+pyodbc://@ACER\\MSSQLSERVER01/Llama3.1TestNoForeignKey"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

# Schema của các bảng
SCHEMA_CONTEXT = """
Bảng customers:
- customer_id (INTEGER, khóa chính)
- customer_name (VARCHAR(50), không null)
- email (VARCHAR(50), có thể null)
- phone (VARCHAR(15), có thể null)
- country (VARCHAR(50), có thể null)

Bảng products:
- product_id (INTEGER, khóa chính)
- product_name (VARCHAR(50), không null)
- price (DECIMAL(10,2), không null)
- category (VARCHAR(50), có thể null)

Bảng orders:
- order_id (INTEGER, khóa chính)
- customer_id (INTEGER, liên kết với customers.customer_id, không null)
- order_date (DATE, không null)
- total_amount (DECIMAL(10,2), có thể null)

Bảng order_details:
- order_detail_id (INTEGER, khóa chính)
- order_id (INTEGER, liên kết với orders.order_id, không null)
- product_id (INTEGER, liên kết với products.product_id, không null)
- quantity (INTEGER, không null, lớn hơn 0)
- unit_price (DECIMAL(10,2), không null)
"""

# Khởi tạo SQL engine
engine = create_engine(CONNECTION_STRING)

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
      "intent": "search|tổng hợp",
      "tables": ["customers", "orders", "order_details", "products"],
      "filters": {
        "customer_name": "str|None",
        "country": "str|None",
        "order_date": "YYYY-MM-DD|YYYY-MM|None",
        "product_name": "str|None",
        "category": "str|None"
      },
      "query": "str"
    }
    Ví dụ: Nếu input là "Tìm khách hàng đặt đơn hàng trong tháng 1/2024",
    trả về:
    {
      "intent": "search",
      "tables": ["customers", "orders"],
      "filters": {
        "customer_name": null,
        "country": null,
        "order_date": "2024-01",
        "product_name": null,
        "category": null
      },
      "query": "khách hàng đặt đơn hàng"
    }
    """
)

# Prompt template để sinh câu lệnh SQL
SQL_PROMPT = PromptTemplate(
    input_variables=["user_input", "schema_context"],
    template="""
    Bạn là chuyên gia SQL Server. Dựa trên câu hỏi của người dùng và schema cơ sở dữ liệu,
    tạo một câu lệnh SQL chính xác.

    Schema: {schema_context}
    Câu hỏi: {user_input}

    **Quy tắc**:
    1. Chỉ sử dụng các bảng và cột trong schema: customers, orders, order_details, products.
    2. Sử dụng LEFT JOIN để đảm bảo trả về dữ liệu ngay cả khi không có bản ghi khớp.
    3. Sử dụng alias: c (customers), o (orders), od (order_details), p (products).
    4. Đặt alias cho cột trả về: customer_name AS 'Tên khách hàng', product_name AS 'Tên sản phẩm', unit_price AS 'Đơn giá', quantity AS 'Số lượng'.
    5. Nếu lọc theo ngày (order_date), hỗ trợ định dạng YYYY-MM-DD hoặc YYYY-MM.
    6. Giới hạn kết quả với TOP 10 nếu không yêu cầu cụ thể.
    7. Trả về CHỈ câu lệnh SQL, từ "SELECT" đến ";".
    8. Sắp xếp theo c.customer_name, p.product_name nếu không yêu cầu khác.
    9. Xử lý giá trị NULL bằng COALESCE trong phép tính (ví dụ: COALESCE(od.unit_price * od.quantity, 0)).
    """
)

# Prompt template để trả về câu trả lời ngôn ngữ tự nhiên
RESPONSE_PROMPT = PromptTemplate(
    input_variables=["user_input", "results", "intent"],
    template="""
    Bạn là trợ lý AI hỗ trợ doanh nghiệp. Dựa trên yêu cầu: "{user_input}",
    và kết quả truy vấn: {results},
    trả lời bằng tiếng Việt, ngôn ngữ tự nhiên, theo ý định: {intent}.
    - Nếu intent là "search": Trả về thông tin chi tiết từ kết quả (tên khách hàng, sản phẩm, số lượng, v.v.).
    - Nếu intent là "tổng hợp": Tóm tắt hoặc tổng hợp kết quả thành báo cáo ngắn gọn (tối đa 100 từ).
    - Nếu không có dữ liệu: Thông báo "Không tìm thấy dữ liệu phù hợp."
    """
)


# Hàm trích xuất intent và filters
def extract_intent_and_filters(user_input: str) -> Dict[str, Any]:
    intent_chain = LLMChain(llm=llm, prompt=INTENT_PROMPT)
    try:
        result = intent_chain.run(user_input=user_input)
        return eval(result)  # Chuyển đổi string JSON thành dict
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất intent: {e}")
        return {
            "intent": "search",
            "tables": ["customers", "orders", "order_details", "products"],
            "filters": {},
            "query": user_input
        }


# Hàm kiểm tra cú pháp SQL
def validate_sql_query(sql_query: str) -> bool:
    try:
        parsed = sqlparse.parse(sql_query)
        return len(parsed) > 0 and parsed[0].get_type() == "SELECT"
    except Exception:
        return False


# Hàm query SQL Server
def query_sql_server(user_input: str) -> str:
    """
    Hàm query SQL Server dựa trên user input, trả về câu trả lời ngôn ngữ tự nhiên.

    Args:
        user_input (str): Câu hỏi hoặc yêu cầu từ người dùng.

    Returns:
        str: Câu trả lời ngôn ngữ tự nhiên.
    """
    logger.info(f"Nhận user input: {user_input}")

    # Bước 1: Trích xuất intent và filters
    intent_data = extract_intent_and_filters(user_input)
    intent = intent_data["intent"]
    tables = intent_data["tables"]
    filters = intent_data["filters"]
    query = intent_data["query"]

    # Bước 2: Sinh câu lệnh SQL
    sql_chain = LLMChain(llm=llm, prompt=SQL_PROMPT)
    try:
        sql_query = sql_chain.run(
            user_input=user_input,
            schema_context=SCHEMA_CONTEXT
        ).strip()

        # Kiểm tra cú pháp SQL
        if not validate_sql_query(sql_query):
            logger.error("Câu lệnh SQL không hợp lệ")
            return "Lỗi: Không thể sinh câu lệnh SQL hợp lệ."

        # Bước 3: Thực thi truy vấn
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        # Bước 4: Xử lý kết quả
        if df.empty:
            return "Không tìm thấy dữ liệu phù hợp với yêu cầu của bạn."

        # Giới hạn số dòng gửi cho LLM
        max_rows = 10
        results_str = df.head(max_rows).to_string(index=False)

        # Bước 5: Sinh câu trả lời ngôn ngữ tự nhiên
        response_chain = LLMChain(llm=llm, prompt=RESPONSE_PROMPT)
        with get_openai_callback() as cb:
            response = response_chain.run(
                user_input=user_input,
                results=results_str,
                intent=intent
            )
            logger.info(f"Thời gian xử lý LLM: {cb.total_time}")

        return response.strip()

    except Exception as e:
        logger.error(f"Lỗi khi query SQL Server: {e}")
        return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."


# Ví dụ sử dụng
if __name__ == "__main__":
    user_input = "Tìm khách hàng đặt đơn hàng trong tháng 1/2024"
    response = query_sql_server(user_input)
    print(response)