import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sqlite3
from typing import Annotated, TypedDict
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

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
    print("✅ Đã khởi tạo xong Vector DB.")
else:
    print("⚠️ Cảnh báo: Không tìm thấy file sample.pdf ở thư mục gốc!")
    retriever = None

# --- Định nghĩa các Tools nghiệp vụ ---
@tool
def get_company_sop(query: str) -> str:
    """Tra cứu quy trình, SOP hoặc tài liệu hướng dẫn công việc nội bộ."""
    if not retriever:
        return "Chưa có tài liệu quy trình."
    docs = retriever.invoke(query)
    return "\n---\n".join([d.page_content for d in docs])

@tool
def create_task_item(task_name: str, deadline: str, assign_to: str) -> str:
    """Tạo một công việc mới trong hệ thống quản lý task của doanh nghiệp."""
    return f"Đã thực hiện tạo task thành công: '{task_name}' | Hạn chót: {deadline} | Người thực hiện: {assign_to}."

tools = [get_company_sop, create_task_item]

# Khởi tạo LLM và bind tools
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)

# Định nghĩa State hệ thống
class TaskAgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Node xử lý chính
def assistant_node(state: TaskAgentState):
    system_prompt = SystemMessage(
        content=(
            "Bạn là Trợ lý Quản lý Công việc thông minh. "
            "QUY TẮC:\n"
            "1. Khi tra cứu tài liệu/quy trình, gọi 'get_company_sop'.\n"
            "2. Khi được yêu cầu tạo task mới, bắt buộc gọi công cụ 'create_task_item'."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Xây dựng Đồ thị LangGraph
builder = StateGraph(TaskAgentState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Sử dụng SQLite Persistent Memory
db_conn = sqlite3.connect("task_assistant_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(db_conn)

# --- ĐIỂM KHÁC BIỆT QUAN TRỌNG (HUMAN-IN-THE-LOOP) ---
# Biên dịch đồ thị với interrupt_before=["tools"]. 
# Hệ thống sẽ TỰ ĐỘNG DỪNG LẠI trước khi thực thi bất kỳ Tool nào để chờ con người Sign-off.
graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🛡️ [DAY 5] HUMAN-IN-THE-LOOP & GATE SIGN-OFF AGENT ĐÃ SẴN SÀNG")
    print("💡 Mọi hành động gọi tool (đặc biệt là tạo task) sẽ bị chặn lại để chờ bạn phê duyệt.")
    print("="*70 + "\n")
    
    config = {"configurable": {"thread_id": "manager_production_01"}}
    
    while True:
        # Kiểm tra xem đồ thị có đang bị tạm dừng (interrupt) ở node nào không
        current_state = graph.get_state(config)
        
        if current_state.next:
            # Hệ thống đang bị treo ở cổng kiểm soát (Gate) trước khi chạy node tiếp theo
            print("\n🚨 [GATE ALERT] AI đã soạn thảo xong hành động và đang chờ bạn phê duyệt (Sign-off)!")
            
            # Lấy thông tin tool call mà AI muốn chạy từ tin nhắn cuối cùng
            last_message = current_state.values["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    print(f"   - Công cụ định gọi: {tc['name']}")
                    print(f"   - Tham số chi tiết: {tc['args']}")
            
            choice = input("\n👉 Bạn có đồng ý phê duyệt (Sign-off) hành động này không? (y/n): ").strip().lower()
            
            if choice == 'y':
                print("✅ Đã phê duyệt! Hệ thống tiếp tục thực thi hành động...")
                # Truyền None vào graph.stream để resume (tiếp tục chạy từ điểm bị dừng)
                for event in graph.stream(None, config, stream_mode="values"):
                    latest_message = event["messages"][-1]
                if hasattr(latest_message, "content") and latest_message.content:
                    print(f"\nAI Task Assistant: {latest_message.content}")
            else:
                print("❌ Đã từ chối hành động (Reject). Hủy bỏ yêu cầu.")
                # Gửi thông điệp hủy bỏ vào state để AI biết và phản hồi lại người dùng
                reject_msg = ToolMessage(content="Người dùng đã từ chối thực hiện hành động này.", tool_call_id=last_message.tool_calls[0]['id'])
                graph.update_state(config, {"messages": [reject_msg]})
            continue

        # Vòng lặp nhận input bình thường nếu không có interrupt nào đang chờ
        user_input = input("\nQuản lý / Nhân sự: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Tạm biệt!")
            break
        if not user_input.strip():
            continue
            
        input_message = HumanMessage(content=user_input)
        
        # Chạy đồ thị cho đến khi gặp Gate (interrupt_before) hoặc hoàn thành
        for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
            latest_message = event["messages"][-1]
            
        # Nếu đồ thị dừng lại ở interrupt thì vòng lặp while True tiếp theo sẽ tự bắt được state.next
        if not graph.get_state(config).next:
            if hasattr(latest_message, "content") and latest_message.content:
                print(f"\nAI Task Assistant: {latest_message.content}")