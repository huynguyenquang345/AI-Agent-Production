import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore  # Import mới từ gói langchain-qdrant
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class LocalAdvancedRAGPipeline:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
        print("🔄 Đang khởi tạo mô hình Local (Ollama)...")
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = ChatOllama(model="qwen2.5:3b", temperature=0)
        
        self.vectorstore = None
        self.rag_chain = None

    def process_and_index_documents(self):
        """Load PDF, cắt Chunking và lưu trữ vào Qdrant Vector DB."""
        print(f"📖 Đang đọc tài liệu từ: {self.pdf_path}...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print("✂️ Đang tiến hành Chunking văn bản...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Đã tạo ra {len(chunks)} chunks.")

        print("🧠 Đang tạo Embeddings cục bộ và lưu vào Qdrant Vector DB...")
        # Sử dụng QdrantVectorStore chuẩn mới thay cho Qdrant cũ
        self.vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            location=":memory:",  # Lưu tạm trên RAM
            collection_name="local_tech_knowledge",
        )
        print("✅ Đã hoàn tất lưu trữ Vector DB nội bộ!")

    def build_rag_chain(self):
        """Xây dựng chuỗi truy vấn RAG."""
        if not self.vectorstore:
            raise ValueError("Chưa khởi tạo Vector Store!")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        system_prompt = (
            "Bạn là một chuyên gia hỗ trợ kỹ thuật cao cấp.\n"
            "Hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn dựa TRỰC TIẾP vào ngữ cảnh (context) dưới đây.\n"
            "Nếu thông tin không nằm trong ngữ cảnh, hãy trả lời: 'Tôi không tìm thấy thông tin này trong tài liệu được cung cấp.'\n\n"
            "Ngữ cảnh:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("⛓️ Local RAG Chain đã sẵn sàng!")

    def query(self, user_question: str) -> dict:
        """Gửi câu hỏi tới hệ thống RAG."""
        if not self.rag_chain:
            raise ValueError("Chưa xây dựng RAG Chain!")
        
        response = self.rag_chain.invoke({"input": user_question})
        return {
            "answer": response["answer"],
            "source_documents": response["context"]
        }


if __name__ == "__main__":
    PDF_FILE = "sample.pdf"

    if not os.path.exists(PDF_FILE):
        print(f"⚠️ Vui lòng bỏ 1 file PDF tên là '{PDF_FILE}' vào thư mục `week_1` để chạy!")
    else:
        # Khởi tạo & Chạy Pipeline Local
        rag = LocalAdvancedRAGPipeline(pdf_path=PDF_FILE)
        rag.process_and_index_documents()
        rag.build_rag_chain()
        
        # Đặt câu hỏi thử nghiệm
        question = "Tóm tắt các nội dung chính hoặc kinh nghiệm trong tài liệu này?"
        print(f"\n❓ Câu hỏi: {question}\n" + "-"*50)
        
        result = rag.query(question)
        
        print(f"\n🤖 Trả lời từ AI (Local Ollama):\n{result['answer']}\n")
        print("="*50)
        print("📌 Đoạn trích dẫn được dùng làm bằng chứng:")
        for idx, doc in enumerate(result["source_documents"], 1):
            print(f"\n--- Chunk {idx} (Trang {doc.metadata.get('page', 'N/A')}) ---")
            print(doc.page_content[:200] + "...")