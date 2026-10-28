
# 📐 Kiến trúc Kỹ thuật Hệ thống (Technical Architecture)

Hệ thống được thiết kế theo mô hình phân tầng rõ rệt:

1. **Prompt Engineering Layer:** Định hình vai trò và quy chuẩn phản hồi chuẩn xác cho LLM thông qua các system prompt chặt chẽ.
2. **Context Engineering Layer (`src/context_engineering/`):** Chịu trách nhiệm nạp file động, chuyển đổi tài liệu thô thành các vector embeddings và lưu trữ tại Qdrant Vector Database, giúp giải quyết triệt để bài toán ảo giác (hallucination).
3. **Agentic Engineering Layer (`src/agentic_core/`):** Xây dựng đồ thị trạng thái tự trị sử dụng **LangGraph**, quản lý chu trình Suy luận (Reasoning) -> Lên kế hoạch (Planning) -> Thực thi (Execution).
4. **Governance & Monitoring (`src/governance/`, `src/monitoring/`):** Tích hợp kiểm soát gián đoạn (Human-in-the-Loop) và bộ lập lịch chủ động (`APScheduler`) để giám sát tiến độ ngầm.
