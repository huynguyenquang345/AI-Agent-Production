import os
from typing import Annotated, TypedDict
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Khởi tạo vector store từ file sample.pdf trong week_1
print("Đang khởi tạo hệ thống RAG cho Multi-Agent...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

pdf_path = os.path.join("..", "week_1", "sample.pdf")
if os.path.exists(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="multi_agent_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Đã khởi tạo xong Vector DB.")
else:
    print("Cảnh báo: Không tìm thấy file sample.pdf.")
    retriever = None


# Hàm tra cứu CV trực tiếp
def get_cv_context(query: str) -> str:
    if not retriever:
        return "Không có dữ liệu CV."
    docs = retriever.invoke(query)
    return "\n---\n".join([d.page_content for d in docs])


# Định nghĩa State chung
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(model="qwen2.5:3b", temperature=0)


# 1. HR Screener Agent Node
def hr_agent_node(state: MultiAgentState):
    user_query = state["messages"][-1].content
    # Tra cứu dữ liệu RAG cho HR
    context = get_cv_context(user_query)
    
    system_prompt = SystemMessage(
        content=(
            "Bạn là Chuyên viên Nhân sự (HR Screener). "
            "Dưới đây là dữ liệu trích xuất từ CV của ứng viên:\n"
            f"{context}\n\n"
            "Dựa trên dữ liệu trên, hãy trả lời ngắn gọn, chính xác câu hỏi của người dùng."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# 2. Tech Interviewer Agent Node
def tech_agent_node(state: MultiAgentState):
    user_query = state["messages"][-1].content
    # Tra cứu dữ liệu RAG cho Kỹ thuật
    context = get_cv_context(user_query)
    
    system_prompt = SystemMessage(
        content=(
            "Bạn là Kỹ sư trưởng (Tech Lead). "
            "Dưới đây là dữ liệu trích xuất từ CV của ứng viên:\n"
            f"{context}\n\n"
            "Dựa trên dữ liệu trên, hãy phân tích kỹ năng lập trình, dự án và kinh nghiệm kỹ thuật."
        )
    )
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# 3. Router Điều hướng
def router_node(state: MultiAgentState):
    last_message = state["messages"][-1].content.lower()
    tech_keywords = ["kỹ thuật", "c++", "code", "dự án", "thuật toán", "oop", "ngôn ngữ", "kỹ năng"]
    if any(keyword in last_message for keyword in tech_keywords):
        return "tech_agent"
    return "hr_agent"


# Xây dựng LangGraph Đồ thị
builder = StateGraph(MultiAgentState)

builder.add_node("hr_agent", hr_agent_node)
builder.add_node("tech_agent", tech_agent_node)

builder.add_conditional_edges(
    START,
    router_node,
    {
        "hr_agent": "hr_agent",
        "tech_agent": "tech_agent"
    }
)

builder.add_edge("hr_agent", END)
builder.add_edge("tech_agent", END)

graph = builder.compile()


if __name__ == "__main__":
    print("\nHệ thống Multi-Agent đã sẵn sàng. Gõ 'exit' để thoát.\n" + "-"*50)
    
    history = []
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Tạm biệt!")
            break
        if not user_input.strip():
            continue
            
        history.append(HumanMessage(content=user_input))
        
        result = graph.invoke({"messages": history})
        history = result["messages"]
        
        print(f"\nAI Agent: {history[-1].content}")