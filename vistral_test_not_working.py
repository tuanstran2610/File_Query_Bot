# from llama_cpp import Llama
#
# # Khởi tạo model với tùy chọn không thêm bos_token và tắt verbose
# llm = Llama(
#     model_path="ggml-vistral-7B-chat-q8.gguf",
#     n_gpu_layers=35,
#     n_ctx=4096,
#     add_bos_token=False,
#     verbose=False  # Tắt logging hiệu suất
# )
#
# # System prompt
# system_prompt = "Bạn là một trợ lý Tiếng Việt nhiệt tình và trung thực."
#
# # Vòng lặp chat
# while True:
#     # Nhập câu hỏi từ người dùng
#     user_input = input("Bạn: ")
#
#     # Thoát nếu người dùng nhập "exit" hoặc "quit"
#     if user_input.lower() in ["exit", "quit"]:
#         print("Tạm biệt!")
#         break
#
#     # Tạo prompt theo chat template, bỏ <s> thủ công
#     prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input.strip()} [/INST]"
#
#     # Sinh phản hồi từ model
#     output = llm(prompt, max_tokens=512, stop=["</s>"], echo=False)
#
#     # In phản hồi
#     response = output['choices'][0]['text'].strip()
#     print(f"Bot: {response}\n")
#
#
from sqlalchemy import create_engine, inspect
from llama_index.core import SQLDatabase, VectorStoreIndex
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import re

# ✅ Cài đặt embedding model
from llama_index.core import Settings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v1")

# ✅ Khởi tạo mô hình ggml-vistral-7B-chat-q8.gguf với llama_cpp
llm = LlamaCpp(
    model_path="ggml-vistral-7B-chat-q8.gguf",
    n_gpu_layers=35,
    n_ctx=4096,
    temperature=0.0,
    max_tokens=512,
    verbose=False
)

# ✅ Kết nối SQL Server
connection_string = (
    "mssql+pyodbc://@ACER\\MSSQLSERVER01/Llama3.1TestNoForeignKey"
    "?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)
engine = create_engine(connection_string)

# ✅ Kiểm tra các bảng trong cơ sở dữ liệu
inspector = inspect(engine)
tables = inspector.get_table_names()
print("Các bảng trong cơ sở dữ liệu:", tables)

# ✅ Tạo SQLDatabase
sql_database = SQLDatabase(engine, include_tables=[
    "customers", "orders", "order_details", "products"
])

# ✅ Định nghĩa ngữ cảnh và schema cho các bảng
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
        "category (danh mục sản phẩm, kiểu VARCHAR(50), có thể null). "
        "Bảng này liên kết với bảng order_details thông qua product_id."
    ),
    "orders": (
        "Lưu trữ thông tin đơn hàng, bao gồm: "
        "order_id (khóa chính, kiểu INTEGER, định danh duy nhất), "
        "customer_id (kiểu INTEGER, liên kết logic với customers(customer_id)), "
        "order_date (ngày đặt hàng, kiểu DATE, không null), "
        "total_amount (tổng giá trị đơn hàng, kiểu DECIMAL(10,2), có thể null). "
        "Bảng này liên kết với bảng order_details thông qua order_id."
    ),
    "order_details": (
        "Lưu trữ chi tiết các sản phẩm trong mỗi đơn hàng, bao gồm: "
        "order_detail_id (khóa chính, kiểu INTEGER, định danh duy nhất), "
        "order_id (kiểu INTEGER, liên kết logic với orders(order_id)), "
        "product_id (kiểu INTEGER, liên kết logic với products(product_id)), "
        "quantity (số lượng sản phẩm, kiểu INTEGER, không null), "
        "unit_price (đơn giá, kiểu DECIMAL(10,2), không null)."
    )
}

# ✅ Tạo ánh xạ và schema
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(table_name=table_name, context_str=context_str)
    for table_name, context_str in table_details.items()
]

# ✅ Tạo ObjectIndex
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex
)

# ✅ Prompt để sinh câu lệnh SQL
text_to_sql_prompt = PromptTemplate(
    input_variables=["context_str", "query_str"],
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
5. Chỉ lọc theo ngày (orders.order_date) nếu người dùng yêu cầu cụ thể.
6. Nếu cần tổng hợp (SUM, COUNT, AVG), sử dụng GROUP BY và HAVING.
7. Giới hạn kết quả với TOP 10 nếu không yêu cầu cụ thể số lượng.
8. Trả về CHỈ câu lệnh SQL thuần túy, từ "SELECT" đến dấu chấm phẩy cuối cùng (;).
9. Sử dụng alias (c, o, od, p) để ngắn gọn.
10. Tên bảng phải đúng chính xác: customers, orders, order_details, products.
11. Đảm bảo các cột tồn tại: customer_id, customer_name, order_id, product_id, product_name, unit_price, quantity, order_date.
12. Sử dụng LEFT JOIN để đảm bảo trả về dữ liệu ngay cả khi không có bản ghi khớp.
13. Đặt alias cho các cột trả về: customer_name AS 'Tên khách hàng', product_name AS 'Tên sản phẩm', unit_price AS 'Đơn giá', quantity AS 'Số lượng'.
14. Sắp xếp kết quả theo customer_name và product_name nếu không yêu cầu khác.

Câu lệnh SQL:
"""
)

# ✅ Prompt để diễn giải kết quả
response_synthesis_prompt = PromptTemplate(
    input_variables=["query_str", "sql_query", "context_str"],
    template="""
Bạn là một trợ lý AI. Dựa trên câu hỏi của người dùng, câu lệnh SQL, và ngữ cảnh,
hãy trả lời bằng tiếng Việt dưới dạng ngôn ngữ tự nhiên, dễ hiểu.

Câu hỏi: {query_str}
SQL: {sql_query}
Ngữ cảnh: {context_str}

**Hướng dẫn**:
1. Tóm tắt thông tin từ kết quả truy vấn (các cột: order_id, customer_name, product_name, unit_price, quantity, order_date).
2. Trả lời ngắn gọn, tự nhiên, như một câu trả lời cho người dùng thông thường.
3. Nếu không có dữ liệu, thông báo rõ ràng (ví dụ: 'Hiện tại không có đơn hàng nào được tìm thấy.').
4. Không lặp lại câu lệnh SQL hoặc bảng kết quả thô.

Câu trả lời:
"""
)

# ✅ Tạo LLMChain cho text-to-SQL
sql_chain = LLMChain(llm=llm, prompt=text_to_sql_prompt)

# ✅ Hàm truy vấn SQL
query_cache = {}

def sql_query_bot(question):
    sql_query = None
    try:
        # Kiểm tra cache
        if question in query_cache:
            result, sql_query = query_cache[question]
            return result, sql_query

        # Lấy ngữ cảnh từ các bảng liên quan
        context_str = "\n".join([details for _, details in table_details.items()])

        # Sinh câu lệnh SQL
        sql_query = sql_chain.run(context_str=context_str, query_str=question)

        # Xử lý câu lệnh SQL: Chỉ lấy phần từ SELECT đến dấu chấm phẩy
        select_pos = sql_query.upper().find("SELECT")
        if select_pos == -1:
            return "Lỗi: Câu lệnh SQL không hợp lệ (không bắt đầu bằng SELECT).", sql_query
        sql_query = sql_query[select_pos:]
        end_pos = sql_query.find("\n")
        if end_pos != -1:
            sql_query = sql_query[:end_pos].strip()
        sql_query = sql_query.strip(";").strip() + ";"

        # Chạy câu lệnh SQL
        df = pd.read_sql(sql_query, engine)
        result = df if not df.empty else "Không có dữ liệu trả về."
        query_cache[question] = (result, sql_query)
        return result, sql_query
    except Exception as e:
        return f"Failed to query from SQL Server: {str(e)}", sql_query

# ✅ Hàm bot_query để trả về kết quả
def bot_query(user_input):
    result, sql_query = sql_query_bot(user_input)
    response = {"status": "success", "response2_sql": "", "sql_query": ""}

    if isinstance(result, pd.DataFrame):
        response["response2_sql"] = result.to_string(index=False)
    else:
        response["response2_sql"] = str(result)

    response["sql_query"] = sql_query if sql_query else "Không có câu lệnh SQL."
    return response

# ✅ Chạy thử
question = "Hiển thị danh sách khách hàng và đơn hàng của họ"
print(bot_query(question))