import json
import openai
import instructor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize client for local Functionary server
client = openai.OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)
client = instructor.patch(client)

# Backend implementations (mock for demonstration)
def sql_query_func(query):
    logger.info(f"Executing SQL query: {query}")
    return [{"Name": "John Doe", "Orders": "Order #123"}, {"Name": "Jane Smith", "Orders": "Order #456"}]

def file_query_func(query):
    logger.info(f"Querying ChromaDB: {query}")
    return [{"document": "Customer order trends for 2025.", "metadata": {"category": "orders"}, "distance": 0.1}]

def business_task_func(task, content):
    logger.info(f"Handling business task: {task}, content: {content}")
    if task == "letter":
        return f"Dear Valued Partner,\n\n{content}\n\nSincerely,\nYour Company"
    elif task == "report":
        return f"Report: {content}\n\nDepartment: Sales"
    elif task == "memo":
        return f"To: All Staff\nSubject: {content}\n\nDetails: Order Update"
    return {"error": "Invalid task type"}

def chat_with_llm_fc(mess_input):
    messages = [
        {
            "role": "system",
            "content": "Base on the information returned by function calling to answer the question."
        },
        {
            "role": "user",
            "content": mess_input
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "sql_query_func",
                "description": "Execute a SQL query against a SQL Server database and return the results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute, e.g., 'SELECT * FROM Customers JOIN Orders ON Customers.ID = Orders.CustomerID'"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_query_func",
                "description": "Query a ChromaDB vector database to retrieve relevant documents or data based on user input",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user-provided query string to search for, e.g., 'customer order trends'"
                        }
                    },
                    "required должно быть": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "business_task_func",
                "description": "Handle various business tasks such as creating reports, drafting letters, or generating memos based on user specifications",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The type of business task to perform, e.g., 'report', 'letter', 'memo'",
                            "enum": ["report", "letter", "memo"]
                        },
                        "content": {
                            "type": "string",
                            "description": "The main content or description of the task, e.g., 'Generate a customer order report' or 'Draft a letter to a customer'"
                        }
                    },
                    "required": ["task", "content"]
                }
            }
        }
    ]

    # Call model 1st time
    try:
        logger.info("Sending first model request")
        response = client.chat.completions.create(
            model="functionary-7b-v2.q8_0.gguf",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
    except Exception as e:
        logger.error(f"First model call failed: {str(e)}")
        raise

    # Get response message
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    function_list = {
        "sql_query_func": sql_query_func,
        "file_query_func": file_query_func,
        "business_task_func": business_task_func
    }

    # Check and call corresponding functions
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = function_list.get(function_name)
            if not function_to_call:
                logger.warning(f"Function {function_name} not found")
                continue
            function_args = json.loads(tool_call.function.arguments)

            try:
                if function_name == "sql_query_func":
                    function_response = function_to_call(
                        query=function_args.get("query")
                    )
                elif function_name == "file_query_func":
                    function_response = function_to_call(
                        query=function_args.get("query")
                    )
                elif function_name == "business_task_func":
                    function_response = function_to_call(
                        task=function_args.get("task"),
                        content=function_args.get("content")
                    )
            except Exception as e:
                logger.error(f"Function {function_name} failed: {str(e)}")
                function_response = {"error": str(e)}

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                }
            )

    # Call LLM 2nd time with function response
    try:
        logger.info("Sending second model request")
        second_response = client.chat.completions.create(
            model="functionary-7b-v2.q8_0.gguf",
            messages=messages,
            temperature=0.1
        )
    except Exception as e:
        logger.error(f"Second model call failed: {str(e)}")
        raise

    return second_response.choices[0].message.content

# Interactive chat loop
while True:
    message_input = input("You: ")
    if message_input.lower() in ["exit", "quit"]:
        break
    try:
        bot_message = chat_with_llm_fc(message_input)
        print("#" * 10, " Bot :", bot_message)
    except Exception as e:
        print(f"Error: {str(e)}")