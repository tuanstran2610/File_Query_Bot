import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_react_agent, AgentExecutor
import chromadb
import uuid

# Khởi tạo mô hình cục bộ với Ollama
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# --- Thiết lập SQL Server ---
connection_string = (
    "mssql+pyodbc://@ACER\\MSSQLSERVER01/Llama3.1TestNoForeignKey"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)
db = SQLDatabase.from_uri(
    connection_string,
    include_tables=["customers", "orders", "order_details", "products"],
    sample_rows_in_table_info=2
)

# Thông tin chi tiết về bảng
table_details = {
    "customers": (
        "Lưu trữ thông tin khách hàng, bao gồm: "
        "customer_id (khóa chính, kiểu INTEGER, định danh duy nhất), "
        "customer_name (tên khách hàng, kiểu VARCHAR(50), không null), "
        "email (VARCHAR(50), có thể null), "
        "phone (VARCHAR(15), có thể null), "
        "country (quốc gia, VARCHAR(50), có thể null). "
        "Bảng này liên kết với bảng orders thông qua customer_id."
    ),
    "products": (
        "Lưu trữ thông tin sản phẩm, bao gồm: "
        "product_id (khóa chính, kiểu INTEGER, định danh duy nhất), "
        "product_name (tên sản phẩm, kiểu VARCHAR(50), không null), "
        "price (giá mặc định, kiểu DECIMAL(10,2), không null), "
        "category (danh mục sản phẩm, kiểu VARCHAR(50), có thể null)."
    ),
    "orders": (
        "Lưu trữ thông tin đơn hàng, bao gồm: "
        "order_id (khóa chính, kiểu INTEGER, định danh duy nhất), "
        "customer_id (kiểu INTEGER, liên kết logic với customers(customer_id), không null), "
        "order_date (ngày đặt hàng, kiểu DATE, không null), "
        "total_amount (tổng giá trị đơn hàng, kiểu DECIMAL(10,2), có thể null)."
    ),
    "order_details": (
        "Lưu trữ chi tiết các sản phẩm trong mỗi đơn hàng, bao gồm: "
        "order_detail_id (khóa chính, kiểu INTEGER, định danh duy nhất), "
        "order_id (kiểu INTEGER, liên kết logic với orders(order_id), không null), "
        "product_id (kiểu INTEGER, liên kết logic với products(product_id), không null), "
        "quantity (số lượng sản phẩm, kiểu INTEGER, không null, lớn hơn 0), "
        "unit_price (đơn giá tại thời điểm đặt hàng, kiểu DECIMAL(10,2), không null)."
    )
}

# --- Thiết lập VectorDB (ChromaDB) ---
def setup_chromadb():
    client = chromadb.Client()
    collection = client.create_collection("test_collection") if "test_collection" not in [c.name for c in client.list_collections()] else client.get_collection("test_collection")
    documents = ["Báo cáo dự án: Doanh số Q1 tăng 20%.", "Sổ tay nhân viên: Chính sách nghỉ phép cập nhật năm 2023."]
    metadatas = [{"source": "sales"}, {"source": "hr"}]
    ids = [str(uuid.uuid4()) for _ in documents]
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return Chroma(collection_name="test_collection", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), client=client)

vector_store = setup_chromadb()

# --- Xử lý tác vụ kinh doanh ---
def generate_report(query: str) -> str:
    return f"Báo cáo được tạo: {query}\nTóm tắt: Dữ liệu đã được xử lý thành công."

def send_warning(query: str) -> str:
    return f"Cảnh báo đã gửi: {query}\nThông báo đã được gửi đến người dùng liên quan."

# --- Công cụ LangChain ---
tools = [
    Tool(
        name="SQL_Query",
        func=lambda q: db.run(q),
        description="Thực thi truy vấn SQL trên cơ sở dữ liệu SQL Server. Đầu vào phải là truy vấn SQL hợp lệ."
    ),
    Tool(
        name="VectorDB_Search",
        func=lambda q: vector_store.similarity_search(q, k=1)[0].page_content,
        description="Tìm kiếm văn bản liên quan trong VectorDB dựa trên truy vấn."
    ),
    Tool(
        name="Generate_Report",
        func=generate_report,
        description="Tạo báo cáo kinh doanh dựa trên truy vấn đầu vào."
    ),
    Tool(
        name="Send_Warning",
        func=send_warning,
        description="Gửi cảnh báo đến người dùng dựa trên truy vấn đầu vào."
    )
]

# --- Prompt cho phân tích tác vụ ---
task_analysis_prompt = PromptTemplate(
    template="""
    Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là phân tích yêu cầu người dùng và quyết định hành động:
    1. Thực thi truy vấn SQL trên cơ sở dữ liệu SQL Server.
    2. Tìm kiếm văn bản trong VectorDB.
    3. Thực hiện tác vụ kinh doanh (tạo báo cáo hoặc gửi cảnh báo).

    Thông tin cơ sở dữ liệu:
    {table_details}

    Yêu cầu người dùng: {input}

    **Hướng dẫn**:
    - Nếu yêu cầu liên quan đến thông tin khách hàng, đơn hàng, hoặc sản phẩm, sử dụng công cụ SQL_Query.
    - Nếu yêu cầu tìm kiếm tài liệu hoặc báo cáo, sử dụng VectorDB_Search.
    - Nếu yêu cầu tạo báo cáo hoặc gửi cảnh báo, sử dụng Generate_Report hoặc Send_Warning.
    - Nếu không rõ, yêu cầu làm rõ.

    Trả về tên công cụ cần sử dụng (SQL_Query, VectorDB_Search, Generate_Report, Send_Warning).
    """
)

# --- Prompt tạo truy vấn SQL ---
text_to_sql_prompt = PromptTemplate(
    template="""
    Bạn là một chuyên gia SQL. Dựa trên câu hỏi của người dùng và ngữ cảnh cơ sở dữ liệu,
    hãy tạo một câu lệnh SQL chính xác bằng cú pháp SQL Server.

    Ngữ cảnh: {context_str}
    Câu hỏi của người dùng: {query_str}

    **Quy tắc**:
    1. Chỉ sử dụng các bảng và cột có trong ngữ cảnh: customers, orders, order_details, products.
    2. Để lấy thông tin khách hàng, join bảng customers với orders qua customer_id, sử dụng c.customer_name làm tên khách hàng.
    3. Để lấy tên sản phẩm, join bảng order_details với products qua product_id, sử dụng p.product_name.
    4. Để lấy đơn giá và số lượng, sử dụng order_details.unit_price và order_details.quantity.
    5. Chỉ lọc theo ngày (orders.order_date) nếu người dùng yêu cầu cụ thể (ví dụ: "đơn hàng trong năm 2024" hoặc "đơn hàng tháng 1"). Không tự động thêm bộ lọc thời gian nếu người dùng không yêu cầu.
    6. Nếu cần tổng hợp (SUM, COUNT, AVG), sử dụng GROUP BY và HAVING.
    7. Giới hạn kết quả với TOP 10 nếu không yêu cầu cụ thể số lượng, hoặc theo yêu cầu người dùng.
    8. Trả về CHỈ câu lệnh SQL thuần túy, không bao gồm bất kỳ giải thích, lưu ý, ví dụ, hoặc văn bản bổ sung nào.
       - Chỉ trả về câu lệnh SQL từ "SELECT" đến dấu chấm phẩy cuối cùng (;), không thêm dòng mới hoặc văn bản sau dấu chấm phẩy.
    9. Sử dụng alias (c, o, od, p) để ngắn gọn.
    10. Tên bảng phải đúng chính xác: customers, orders, order_details, products (chữ thường).
    11. Đảm bảo các cột tồn tại: customer_id, customer_name, order_id, product_id, product_name, unit_price, quantity, order_date.
    12. Luôn sử dụng LEFT JOIN thay vì JOIN để đảm bảo trả về dữ liệu ngay cả khi không có bản ghi khớp.
    13. Đặt alias cho các cột trả về như sau: customer_name AS 'Tên khách hàng', product_name AS 'Tên sản phẩm', unit_price AS 'Đơn giá', quantity AS 'Số lượng'.
    14. Khi sử dụng ORDER BY:
        - Nếu sắp xếp theo bí danh cột (alias), sử dụng tên bí danh trong dấu ngoặc vuông, ví dụ: ORDER BY [Tên khách hàng] DESC.
        - Không bao giờ đặt bí danh hoặc tên cột trong dấu nháy đơn ('') hoặc nháy kép ("") trong ORDER BY.
        - Nếu sắp xếp theo cột tổng hợp (như SUM, AVG), lặp lại biểu thức hoặc dùng vị trí cột (ví dụ: ORDER BY 3 DESC).
    15. Nếu người dùng yêu cầu tổng giá trị của "mỗi đơn hàng" lớn hơn một ngưỡng (ví dụ: mỗi đơn hàng > 100), sử dụng CTE hoặc subquery để lọc đơn hàng thỏa mãn trước khi nhóm theo khách hàng.
    16. Xử lý giá trị NULL trong các phép tính (như unit_price * quantity) bằng COALESCE, ví dụ: COALESCE(unit_price * quantity, 0).
    17. Sắp xếp kết quả theo customer_name và product_name (ORDER BY c.customer_name, p.product_name) nếu người dùng không yêu cầu cách sắp xếp khác.
    18. Khi lọc theo quốc gia, kiểm tra các giá trị tương tự (ví dụ: 'USA', 'US', 'United States') bằng NOT IN hoặc thêm điều kiện IS NULL nếu cần.

    Câu lệnh SQL:
    """
)

# --- Prompt trả lời tự nhiên ---
response_synthesis_prompt = PromptTemplate(
    template="""
    Bạn là một trợ lý AI. Dựa trên câu hỏi của người dùng, câu lệnh SQL, và kết quả truy vấn,
    hãy trả lời bằng tiếng Việt dưới dạng ngôn ngữ tự nhiên, dễ hiểu.

    Câu hỏi: {query_str}
    SQL: {sql_query}
    Ngữ cảnh: {context_str}

    **Hướng dẫn**:
    1. Tóm tắt thông tin từ kết quả truy vấn một cách tự nhiên, bao gồm tất cả các cột có trong kết quả (ví dụ: tên khách hàng, mã đơn hàng, số lượng, đơn giá, v.v.).
    2. Trả lời ngắn gọn, tự nhiên, như một câu trả lời cho người dùng thông thường.
    3. Nếu không có dữ liệu, thông báo rõ ràng (ví dụ: 'Hiện tại không có khách hàng nào thỏa mãn điều kiện.').
    4. Không lặp lại câu lệnh SQL hoặc bảng kết quả thô.

    Câu trả lời:
    """
)

# --- Prompt chính cho agent ---
agent_prompt = PromptTemplate(
    template="""
    Bạn là một trợ lý AI thông minh, hỗ trợ người dùng bằng tiếng Việt. Nhiệm vụ của bạn là phân tích yêu cầu người dùng và quyết định hành động: truy vấn SQL, tìm kiếm VectorDB, hoặc thực hiện tác vụ kinh doanh.

    **Thông tin cơ sở dữ liệu**:
    {table_details}

    **Yêu cầu người dùng**: {input}

    **Công cụ**: {tools}
    **Tên công cụ**: {tool_names}

    **Hướng dẫn**:
    1. Nếu yêu cầu liên quan đến thông tin khách hàng, đơn hàng, hoặc sản phẩm, sử dụng công cụ SQL_Query.
    2. Nếu yêu cầu tìm kiếm tài liệu hoặc báo cáo, sử dụng VectorDB_Search.
    3. Nếu yêu cầu tạo báo cáo hoặc gửi cảnh báo, sử dụng Generate_Report hoặc Send_Warning.
    4. Trả về đầu ra theo định dạng JSON sau:
       ```json
       {{
         "action": "tên_công_cụ",
         "action_input": "đầu vào cho công cụ"
       }}
       ```
       - `action`: Một trong các công cụ: SQL_Query, VectorDB_Search, Generate_Report, Send_Warning.
       - `action_input`: Đầu vào cho công cụ (ví dụ: truy vấn SQL hoặc câu hỏi tìm kiếm).
    5. Nếu không rõ yêu cầu, trả về:
       ```json
       {{
         "action": "clarify",
         "action_input": "Vui lòng làm rõ yêu cầu của bạn."
       }}
       ```

    **Suy luận và hành động**: {agent_scratchpad}
    """
)

# --- Tạo agent ---
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Hàm xử lý yêu cầu người dùng ---
def handle_user_request(query: str):
    context_str = "\n".join([f"{k}: {v}" for k, v in table_details.items()])
    response = agent_executor.invoke({"input": query, "table_details": context_str})
    if response["output"].startswith("SQL_Query"):
        # Tạo truy vấn SQL
        sql_query = llm.invoke(text_to_sql_prompt.format(context_str=context_str, query_str=query)).content
        # Thực thi truy vấn
        try:
            sql_result = db.run(sql_query)
            # Tạo câu trả lời tự nhiên
            response = llm.invoke(response_synthesis_prompt.format(query_str=query, sql_query=sql_query, context_str=context_str)).content
            return response
        except Exception as e:
            return f"Lỗi khi thực thi truy vấn SQL: {str(e)}"
    return response["output"]

# --- Kiểm tra hệ thống ---
if __name__ == "__main__":
    queries = [
        "Liệt kê tên khách hàng và sản phẩm họ đã mua trong năm 2024",
        "Tìm thông tin về báo cáo doanh số",
        "Tạo báo cáo doanh số Q2",
        "Gửi cảnh báo về hàng tồn kho thấp"
    ]
    for query in queries:
        print(f"\nYêu cầu: {query}")
        print(f"Trả lời: {handle_user_request(query)}")