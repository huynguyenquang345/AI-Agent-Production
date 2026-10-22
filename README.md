
# AI Task Management Assistant (Local Llama & Multi-Agent)

Hệ thống Trợ lý Quản lý Công việc tự động hóa toàn diện dựa trên kiến trúc Multi-Agent, Advanced RAG và Local LLM (Ollama / Llama / Qwen), chạy hoàn toàn độc lập không phụ thuộc vào các API trả phí bên ngoài.

## 📅 Lộ Trình Phát Triển 2 Tháng

### Tháng 1: Xây Dựng Nền Tảng Knowledge & Dữ Liệu (Data & Knowledge Engine)

- **Tuần 1-2:** Số hóa tài liệu quy trình, SOP công việc (PDF, Word, Markdown) sử dụng PyPDF / Unstructured, ứng dụng Recursive/Semantic Chunking và gắn Metadata quản lý theo phòng ban.
- **Tuần 3-4:** Xây dựng Vector Database (Qdrant), cấu hình Hybrid Search kết hợp Sparse (BM25) và Dense Vector Search cùng thuật toán Reranking để tối ưu hóa truy xuất ngữ cảnh.

### Tháng 2: Xây Dựng Kiến Trúc Agent & Lập Lịch Tự Động (Agent Core & Operations)

- **Tuần 5-6:** Phát triển kiến trúc Agent chuẩn (Orchestrator - Planner - Executor) dùng **LangGraph** kết hợp mô hình cục bộ (`qwen2.5:3b`, `nomic-embed-text`) để phân loại ý định, lập kế hoạch và thao tác CSDL công việc.
- **Tuần 7:** Tích hợp tính năng chủ động (Proactive Agent) quét công việc quá hạn định kỳ và gửi thông báo tự động.
- **Tuần 8:** Tối ưu hóa hiệu suất, đánh giá chất lượng RAG và đóng gói ứng dụng hoàn chỉnh.

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

- **Framework Agent:** LangGraph (Orchestration & State Management)
- **Vector Database:** Qdrant (In-memory / Persistent)
- **LLM & Embeddings (Local):** Ollama (`qwen2.5:3b` cho Agent suy luận, `nomic-embed-text` cho embedding)
- **Environment:** Python 3.11/3.12, Windows, PowerShell, VS Code

## 🚀 Hướng Dẫn Cài Đặt & Chạy Nhanh (Quick Start)



**Cài đặt các gói thư viện cần thiết:**

**Clone repository về máy (Hỗ trợ ổ E hoặc ổ C):**

```bash
git clone <URL_REPOSITORY>
cd AI-Agent-Production
```


**Bash**

```
pip install langchain langchain-ollama langchain-qdrant langgraph pydantic pypdf
```
