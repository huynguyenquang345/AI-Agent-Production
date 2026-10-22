
* **Kiến trúc Multi-Agent Collaboration:** Chuyển đổi từ mô hình một Agent đơn lẻ sang hệ thống nhiều Agent chuyên môn hóa (chia nhiệm vụ rõ ràng thay vì để một Agent ôm đồm mọi việc).
* **Mô hình Phân rã Tác vụ & Định tuyến (Routing / Supervisor):**
  * Xây dựng bộ định tuyến (`router_node`) để tự động phân tích ý định của người dùng từ câu hỏi đầu vào.
  * Điều hướng câu hỏi về đúng chuyên gia phụ trách:**HR Screener** (chuyên xử lý thông tin chung, học vấn, kinh nghiệm) hoặc **Tech Interviewer** (chuyên phân tích kỹ năng lập trình, C++, OOP, thuật toán).
* **Tích hợp RAG chuyên biệt:** Cấu hình để các sub-agent có khả năng tra cứu trực tiếp dữ liệu từ file CV (`sample.pdf`) thông qua hệ thống RAG đã xây dựng từ tuần trước.
