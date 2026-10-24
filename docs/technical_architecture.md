
# Tài liệu Kiến trúc Kỹ thuật & Luồng Xử lý Dữ liệu

## 1. Lựa chọn Trường phái Kỹ thuật

Dự án áp dụng kết hợp **Context Engineering (Advanced RAG)** và **Agentic Engineering (LangGraph Multi-Agent)**:

- **Context Engineering:** Xây dựng luồng xử lý dữ liệu đầu vào khắt khe từ file quy trình (PDF) $\rightarrow$ Chunking (800 token, overlap 150) $\rightarrow$ Embedding cục bộ (`nomic-embed-text`) $\rightarrow$ Lưu trữ Vector Store (Qdrant) $\rightarrow$ Truy xuất Hybrid/Similarity có kiểm soát ngữ cảnh.
- **Agentic Engineering:** Thay vì dùng 1 prompt đơn lẻ ôm đồm mọi việc, hệ thống sử dụng kiến trúc **LangGraph (StateGraph)** với các Agent chuyên biệt (HR Screener, Tech Interviewer, Task Planner) phối hợp thông qua cơ chế Router và Tool-calling có kiểm soát.

## 2. Tuân thủ Nguyên tắc "Maker - Checker" & Kiểm soát AI (L0 - L4)

Dựa theo khung quy trình phát triển phần mềm có AI tham gia:

- **AI chỉ đóng vai trò Responsible (R):** AI chịu trách nhiệm tạo draft, sinh test case, tra cứu tài liệu (Mức L2 - L3).
- **Con người luôn là Accountable (A):** Mọi quyết định phê duyệt (Sign-off các Gate), thay đổi scope, ưu tiên đều do con người thực hiện. AI không được tự ý quyết định.
