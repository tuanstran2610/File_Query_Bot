# main.py
import time
import pyodbc
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from tools.sql_server_query_tool import query_data
from tools.chromadb_file_query_tool import query_file
from tools.business_task_tool import task_handling

# Khởi tạo LLM
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.0,
    base_url="http://localhost:11434",
    timeout=60
)

# System prompt đơn giản
system_prompt = """
Bạn là một trợ lý AI. Sử dụng công cụ phù hợp dựa trên yêu cầu của người dùng. Chỉ định dạng kết quả từ công cụ và không tự thêm dữ liệu hoặc kết luận nếu không cần thiết.
"""

# Tạo prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Bind prompt với LLM
llm_with_prompt = llm.bind(prompt=prompt)

# Định nghĩa các công cụ
tools = [
    Tool(
        name="SQL Query",
        func=query_data,
        description="Truy vấn dữ liệu từ SQL Server. Sử dụng cho yêu cầu danh sách khách hàng, đơn hàng, doanh thu, hoặc bất kỳ dữ liệu nào từ cơ sở dữ liệu."
    ),
    Tool(
        name="File Query",
        func=query_file,
        description="Tóm tắt hoặc phân tích tài liệu từ ChromaDB. Sử dụng cho yêu cầu tóm tắt văn bản, ví dụ: design patterns."
    ),
    Tool(
        name="Business Task",
        func=task_handling,
        description="Xử lý tác vụ doanh nghiệp như tạo báo cáo hoặc đơn từ."
    )
]

# Bind tools vào LLM
llm_with_tools = llm_with_prompt.bind_tools(tools)

# Khởi tạo Agent với LangGraph
agent = create_react_agent(llm_with_tools, tools)

# Vòng lặp tương tác
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    try:
        start_time = time.time()
        print(f"Processing input: {user_input}")
        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"Agent response: {response}")
        # Lấy nội dung từ tin nhắn cuối cùng (thường là AIMessage)
        last_message = response["messages"][-1]
        if hasattr(last_message, "content"):
            print(last_message.content)
        else:
            print("No content available in response.")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

# sql_server_query_tool.py
def query_data(query):
    print(f"SQL Query Tool called with query: {query}")
    try:
        # Kết nối SQL Server (thay bằng thông tin thực tế)
        conn = pyodbc.connect(
            "DRIVER={SQL Server};"
            "SERVER=your_server_name;"
            "DATABASE=your_database_name;"
            "Trusted_Connection=yes;"
        )
        cursor = conn.cursor()
        sql = "SELECT customer_id, name, email FROM customers"
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        # Định dạng kết quả thành bảng
        result = "Danh sách khách hàng:\n\n| customer_id | name         | email            |\n|-------------|--------------|------------------|\n"
        for row in rows:
            result += f"| {row[0]:<11} | {row[1]:<12} | {row[2]:<16} |\n"
        return result
    except Exception as e:
        return f"Error querying SQL Server: {str(e)}"

# chromadb_file_query_tool.py
def query_file(query):
    print(f"File Query Tool called with query: {query}")
    # Giả lập tóm tắt (thay bằng truy vấn ChromaDB thực tế)
    return """
Summary: **Design Patterns**

Design Patterns là các mẫu thiết kế giải quyết vấn đề chung trong lập trình, giúp tái sử dụng và áp dụng vào nhiều tình huống.

**Các loại chính:**
1. **Creational Patterns**: Liên quan đến tạo đối tượng.
   - Singleton: Chỉ tạo 1 instance.
   - Factory: Tạo instance dựa trên tham số.
2. **Structural Patterns**: Liên quan đến cấu trúc lớp/đối tượng.
   - Adapter: Chuyển đổi interface.
   - Bridge: Tách abstraction và implementation.
3. **Behavioral Patterns**: Liên quan đến hành vi.
   - Observer: Gửi thông tin đến nhiều đối tượng.
   - Strategy: Chọn thuật toán phù hợp.

**Lợi ích:**
- Giảm code trùng lặp.
- Tăng tính linh hoạt và bảo trì.
"""

# business_task_tool.py
def task_handling(query):
    print(f"Business Task Tool called with query: {query}")
    return f"Task handled: {query}"