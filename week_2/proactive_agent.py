import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sqlite3
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
from langgraph.checkpoint.sqlite import SqliteSaver

# Import APScheduler để chạy tác vụ nền chủ động (Proactive)
from apscheduler.schedulers.background import BackgroundScheduler

print("🔄 Đang khởi tạo hệ thống Knowledge Base & Vector DB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Xác định đường dẫn tới file sample.pdf ở thư mục gốc
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
    print("✅ Đã khởi tạo xong Vector DB từ tài liệu.")
else:
    print("⚠️ Cảnh báo: Không tìm thấy file sample.pdf ở thư mục gốc!")
    retriever = None

# --- Định nghĩa các Tools nghiệp vụ ---
@tool
def get_company_sop(query: str) -> str:
    """Tra cứu quy trình, SOP, tài liệu hướng dẫn công việc hoặc thông tin chi tiết từ tài liệu nộiبو."""
    if not retriever:
        return "Chưa có tài liệu quy trình."
    docs = retriever.invoke(query)
    return "\n---\n".join([d.page_content for d in docs])

@tool
def create_task_item(task_name: str, deadline: str, assign_to: str) -> str:
    """Tạo một công việc mới trong hệ thống quản lý task của doanh nghiệp."""
    return f"Đã ghi nhận tạo task mới: '{task_name}' | Hạn chót: {deadline} | Người thực hiện: {assign_to}."

tools = [get_company_sop, create_task_item]

# Khởi tạo LLM cục bộ với Ollama và bind tools
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)

# Định nghĩa State hệ thống
class TaskAgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Node xử lý chính của Trợ lý Quản lý Công việc
def assistant_node(state: TaskAgentState):
    system_prompt = SystemMessage(
        content=(
            "Bạn là Trợ lý Quản lý Công việc thông minh của doanh nghiệp. "
            "Nhiệm vụ của bạn là hỗ trợ nhân sự tra cứu quy trình SOP, lập kế hoạch, "
            "phân rã công việc và gọi các công cụ quản lý task khi cần thiết.\n"
            "QUY TẮC:\n"
            "1. Khi người dùng hỏi thông tin về tài liệu hoặc quy trình, bắt buộc gọi công cụ 'get_company_sop'.\n"
            "2. Khi người dùng yêu cầu tạo công việc hoặc giao task mới, bắt buộc gọi công cụ 'create_task_item'."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Xây dựng Đồ thị LangGraph tích hợp ToolNode và Checkpointer SQLite
builder = StateGraph(TaskAgentState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Sử dụng SqliteSaver để lưu trữ trạng thái hội thoại cố định xuống file
db_conn = sqlite3.connect("task_assistant_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(db_conn)
graph = builder.compile(checkpointer=memory)


# --- Tác vụ nền (Background Job) chạy tự động định kỳ ---
def background_deadline_checker():
    """Hàm này chạy ngầm để kiểm tra deadline hoặc nhắc nhở tự động (Proactive Agent)."""
    print("\n⏰ [BACKGROUND SYSTEM] Đang quét hệ thống để kiểm tra tiến độ task và deadline quá hạn...")
    # (Ở đây bạn có thể cấu hình gọi graph hoặc query CSDL task để đưa ra cảnh báo chủ động)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 [WEEK 2 - DAY 5] TRỢ LÝ QUẢN LÝ CÔNG VIỆC CHỦ ĐỘNG (Proactive & APScheduler)")
    print("="*70 + "\n")
    
    # Khởi động Background Scheduler
    scheduler = BackgroundScheduler()
    # Thiết lập lịch chạy ngầm kiểm tra định kỳ mỗi 60 giây (để demo)
    scheduler.add_job(background_deadline_checker, 'interval', seconds=60)
    scheduler.start()
    print("🟢 Đã khởi chạy tiến trình nền (Background Scheduler) thành công!\n")
    
    # Cấu hình phiên làm việc cố định qua thread_id
    config = {"configurable": {"thread_id": "manager_production_01"}}
    
    try:
        while True:
            user_input = input("\nQuản lý / Nhân sự: ")
            if user_input.lower().strip() in ["exit", "quit"]:
                print("Tạm biệt! Đang tắt hệ thống...")
                scheduler.shutdown()
                break
            if not user_input.strip():
                continue
                
            input_message = HumanMessage(content=user_input)
            
            # Chạy đồ thị với checkpointer
            for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
                latest_message = event["messages"][-1]
                
            if hasattr(latest_message, "content") and latest_message.content:
                print(f"\nAI Task Assistant: {latest_message.content}")
                
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("Đã dừng hệ thống an toàn.")