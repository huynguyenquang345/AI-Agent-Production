
## 1. Tổng Quan & Mục Tiêu

* **Mục tiêu:** Chuyển đổi từ mô hình RAG độc lập (Day 1) sang một **AI Agent** tự vận hành, có trạng thái (stateful) với khả năng gọi công cụ (tool calling) linh hoạt và giao tiếp đa lượt qua framework **LangGraph**.
* **Kiến trúc mục tiêu:** Đồ thị trạng thái (`StateGraph`) kết nối giữa nút xử lý suy luận LLM (`chatbot`) và công cụ tra cứu dữ liệu (`tools`) sử dụng Qdrant Vector DB local.

## 2. Kiến Thức Cốt Lõi

### A. AI Agent Khác Gì RAG Đơn Thuần?

* **Standard RAG:** Luồng xử lý cố định 1 chiều (Query $\rightarrow$ Retrieval $\rightarrow$ Generation).
* **AI Agent:** Hệ thống động có **khả năng suy luận (Reasoning)**. Agent tự quyết định trả lời trực tiếp hoặc gọi các công cụ bên ngoài, lặp lại các bước cho đến khi giải quyết xong yêu cầu của người dùng.

### B. Thành Phần Cốt Lõi Trong LangGraph

* **`StateGraph`:** Framework điều phối luồng chạy của Agent dưới dạng đồ thị có trạng thái.
* **`State` (`AgentState`):** Cấu trúc dữ liệu bộ nhớ chung, truyền lịch sử tin nhắn qua các bước xử lý nhờ cơ chế `add_messages`.
* **`Nodes`:** Các nút thực thi tác vụ cụ thể:
  * **`chatbot` Node:** Nhận lịch sử hội thoại, định hình system prompt và gọi LLM suy luận.
  * **`ToolNode`:** Khởi chạy các công cụ (tools) do LLM yêu cầu gọi.
* **`Edges`:** Cơ chế rẽ nhánh điều kiện (`tools_condition`) tự động điều hướng: chuyển qua gọi Tool nếu LLM phát lệnh `tool_calls`, hoặc kết thúc luồng nếu LLM đã hoàn thành câu trả lời.

## 3. Tổng Kết Mã Nguồn Thực Hành (`agent_graph.py`)

### Tech Stack Sử Dụng

* **Ngôn ngữ:** Python 3.11 / 3.12
* **Framework:** LangGraph, LangChain
* **LLM Engine:** Local Ollama (`qwen2.5:3b`)
* **Vector Store & Embeddings:** Local Qdrant (In-Memory), `nomic-embed-text`
