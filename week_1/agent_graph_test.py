import os
from typing import Annotated, TypedDict
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Khởi tạo vector store và load tài liệu RAG
print("Đang khởi tạo hệ thống RAG...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists("sample.pdf"):
    loader = PyPDFLoader("sample.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="agent_rag_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Đã khởi tạo xong Vector DB.")
else:
    print("Cảnh báo: Không tìm thấy file sample.pdf.")
    retriever = None


# Định nghĩa tool tra cứu CV
@tool
def search_candidate_resume(query: str) -> str:
    """Tra cứu thông tin, kỹ năng, dự án hoặc kinh nghiệm của ứng viên trong CV."""
    if not retriever:
        return "Chưa có tài liệu CV để tra cứu."
    
    docs = retriever.invoke(query)
    context = "\n---\n".join([d.page_content for d in docs])
    return f"Kết quả trích xuất từ CV:\n{context}"

tools = [search_candidate_resume]


# Định nghĩa State cho LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# Khởi tạo LLM và gắn tool
llm = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)

def chatbot_node(state: AgentState):
    system_instruction = SystemMessage(
        content=(
            "Bạn là trợ lý AI tuyển dụng. Hãy trả lời thắc mắc dựa trên hồ sơ/CV của ứng viên. "
            "Khi có câu hỏi liên quan đến kinh nghiệm, kỹ năng hay dự án, hãy gọi công cụ search_candidate_resume."
        )
    )
    messages = [system_instruction] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Xây dựng đồ thị StateGraph
builder = StateGraph(AgentState)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile()


# Vòng lặp giao tiếp với người dùng
if __name__ == "__main__":
    print("\nAgent đã sẵn sàng. Gõ 'exit' hoặc 'quit' để thoát.\n" + "-"*50)
    
    messages_history = []
    
    while True:
        user_input = input("\nBạn: ")
        if user_input.lower().strip() in ["exit", "quit"]:
            print("Tạm biệt!")
            break
            
        if not user_input.strip():
            continue
            
        messages_history.append(HumanMessage(content=user_input))
        
        events = graph.stream({"messages": messages_history}, stream_mode="values")
        
        for event in events:
            messages_history = event["messages"]
            
        final_response = messages_history[-1].content
        print(f"\nAI: {final_response}")    