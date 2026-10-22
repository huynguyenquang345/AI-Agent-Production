import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Annotated, TypedDict
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

print("Đang khởi tạo hệ thống Knowledge Base...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Xác định đường dẫn tới file sample.pdf (nằm ở thư mục gốc)
pdf_path = os.path.join("..", "sample.pdf")
if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="task_knowledge_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Đã khởi tạo xong Vector DB.")
else:
    print("Cảnh báo: Không tìm thấy file sample.pdf ở thư mục gốc!")
    retriever = None

# Định nghĩa Tool quản lý công việc (Task Management Tools)
@tool
def get_company_sop(query: str) -> str:
    """Tra cứu quy trình, SOP, tài liệu hướng dẫn hoặc thông tin chi tiết về hồ sơ ứng viên."""
    if not retriever:
        return "Chưa có tài liệu quy trình."
    docs = retriever.invoke(query)
    return "\n---\n".join([d.page_content for d in docs])

tools = [get_company_sop]

# Bind tool trực tiếp vào LLM để mô hình nhận diện và tự động gọi
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)

# Định nghĩa State hệ thống
class TaskAgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Node xử lý chính của Trợ lý Quản lý Công việc
def assistant_node(state: TaskAgentState):
    system_prompt = SystemMessage(
        content=(
            "Bạn là Trợ lý Quản lý Công việc thông minh của doanh nghiệp. "
            "Nhiệm vụ của bạn là hỗ trợ nhân sự tra cứu quy trình SOP, nắm bắt tiến độ công việc "
            "và phân rã yêu cầu thành các bước thực hiện cụ thể.\n"
            "BẮT BUỘC: Khi người dùng hỏi bất kỳ thông tin gì liên quan đến tài liệu, hồ sơ hoặc quy trình, "
            "bạn phải gọi công cụ 'get_company_sop' ngay lập tức để lấy dữ liệu chính xác trước khi trả lời."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Xây dựng Đồ thị LangGraph tích hợp ToolNode và MemorySaver
builder = StateGraph(TaskAgentState)

# Thêm các node vào đồ thị
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

# Thiết lập luồng điều hướng (Edges)
builder.add_edge(START, "assistant")
# Sử dụng tools_condition để tự động rẽ nhánh sang 'tools' nếu LLM gọi tool
builder.add_conditional_edges("assistant", tools_condition)
# Sau khi chạy xong tool, quay lại assistant để tổng hợp câu trả lời
builder.add_edge("tools", "assistant")

# Sử dụng MemorySaver để lưu trữ trạng thái hội thoại qua checkpoint
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    print("\n[DAY 4] Trợ lý Quản lý Công việc đã sẵn sàng (Hỗ trợ Memory, Tool-Calling & Local Llama). Gõ 'exit' để thoát.\n" + "-"*60)
    
    # Cấu hình phiên làm việc cố định qua thread_id
    config = {"configurable": {"thread_id": "manager_session_01"}}
    
    while True:
        user_input = input("\nQuản lý / Nhân sự: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Tạm biệt!")
            break
        if not user_input.strip():
            continue
            
        input_message = HumanMessage(content=user_input)
        
        # Chạy đồ thị với checkpointer
        for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
            latest_message = event["messages"][-1]
            
        # Chỉ in ra nội dung tin nhắn cuối cùng nếu đó là phản hồi từ AI (AIMessage) và có nội dung
        if hasattr(latest_message, "content") and latest_message.content:
            print(f"\nAI Task Assistant: {latest_message.content}")