
# 📊 Phân tích Nghiệp vụ & Định hướng Sản phẩm: Task Management Assistant

## 1. Bối cảnh & Vấn đề thực tế (Pain Points)

- **Quản lý thủ công kém hiệu quả:** Các nhà phát triển (đặc biệt là đội ngũ Fresher/Junior tại các doanh nghiệp phần mềm) thường gặp khó khăn trong việc theo dõi task, cập nhật trạng thái tiến độ và tra cứu tài liệu kỹ thuật rải rác.
- **Thiếu kiểm soát an toàn dữ liệu:** Các trợ lý AI thông thường tự ý thực thi thay đổi dữ liệu mà không có cơ chế kiểm duyệt (Human-in-the-Loop), gây rủi ro cao khi vận hành thực tế.

## 2. Các Kịch Bản Sử Dụng Cốt Lõi (Use Cases)

- **Use Case 1 (Context Engineering):** Tự động nạp, số hóa và chunking tài liệu kỹ thuật nội bộ (như tài liệu quy trình C++ Fresher) vào Qdrant Vector Store để Agent truy xuất chính xác ngữ cảnh.
- **Use Case 2 (Agentic Decision Making):** Agent phân tích yêu cầu ngôn ngữ tự nhiên thành các bước lập lịch công việc chi tiết.
- **Use Case 3 (Governance & HITL):** Áp dụng mô hình phê duyệt (Maker-Checker) ngắt luồng trước khi thực hiện ghi dữ liệu nhạy cảm vào hệ thống.
