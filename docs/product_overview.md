
# Tài liệu Thiết kế Sản phẩm: AI Task Management Assistant

## 1. Tổng quan dự án

Ứng dụng **Trợ lý Quản lý Công việc** được thiết kế riêng cho các doanh nghiệp muốn tự động hóa quy trình vận hành nội bộ, tra cứu tài liệu SOP và theo dõi tiến độ công việc. Hệ thống vận hành hoàn toàn bằng mã nguồn mở và các mô hình ngôn ngữ cục bộ (**Ollama / Llama / Qwen**) để bảo mật dữ liệu tuyệt đối và tối ưu chi phí (0 đồng phí API).

## 2. Kiến trúc Hệ thống (System Architecture)

- **Knowledge Base & RAG Engine:** Tài liệu quy trình và hướng dẫn công việc được phân đoạn (Chunking) và lưu trữ trong Vector Database (Qdrant). Khi người dùng cần tìm hiểu nghiệp vụ, hệ thống thực hiện truy xuất ngữ cảnh chính xác cao.
- **Orchestrator & Multi-Agent Core (LangGraph):** Điều phối các yêu cầu từ nhân sự hoặc quản lý, tự động phân tích nhiệm vụ và gọi công cụ (Tools) phù hợp.
- **Memory Persistence & State Management:** Sử dụng `MemorySaver` của LangGraph để ghi nhớ lịch sử tương tác xuyên suốt các phiên làm việc.

## 3. Các tính năng cốt lõi theo Lộ trình

1. **Tra cứu quy trình thông minh (SOP RAG):** Giải đáp nhanh chóng các thắc mắc về quy định, biểu mẫu và công việc qua tài liệu nội bộ.
2. **Quản lý & Theo dõi trạng thái task:** Hỗ trợ tạo, cập nhật và kiểm tra danh sách công việc.
3. **Mô hình Local 100%:** Hoạt động ổn định trên phần cứng cá nhân thông qua Ollama mà không cần kết nối API trả phí của OpenAI hay Gemini.
