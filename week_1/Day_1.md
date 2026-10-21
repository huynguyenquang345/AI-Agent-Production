
1. Vector Embeddings & Vector Database

- **Vector Embedding là gì?**Vector Embedding là một lớp toán học chuyển đổi văn bản (từ, câu, đoạn) thành một chuỗi số đại diện trong không gian nhiều chiều. Ví dụ, một vector có 1536 chiều với OpenAI text-embedding-3-small.
- **Khoảng cách Ngữ nghĩa (Semantic Distance):**Các câu có ý nghĩa tương tự nhau sẽ nằm gần nhau trong không gian vector. Chúng ta đo khoảng cách này bằng Cosine Similarity hoặc Dot Product.
- **Vector Database (Qdrant, ChromaDB, Pinecone):**
  Đây là các cơ sở dữ liệu tối ưu cho việc tìm kiếm vector hàng láng giềng gần nhất (ANN - Approximate Nearest Neighbors) cực nhanh thông qua các thuật toán chỉ mục như HNSW (Hierarchical Navigable Small World).


2. Chiến lược Cắt khúc văn bản (Chunking Strategy)

   - **Tại sao phải Chunking?**
     LLM có giới hạn Context Window (ngบริบท) và chi phí xử lý/độ trễ tăng theo độ dài input. Chunking giúp chia nhỏ tài liệu thành các đoạn thông tin súc tích.
   - **RecursiveCharacterTextSplitter:**
     Thuật toán chia văn bản ưu tiên giữ nguyên cấu trúc đoạn văn (tách theo thứ tự ưu tiên các ký tự: ["\n\n", "\n", " ", ""]).
   - **Chunk Overlap (Đoạn chồng lấp):**
     Giữ lại một phần ký tự của chunk trước sang chunk sau (ví dụ: chunk_size=1000, chunk_overlap=200) để tránh bị mất ngữ cảnh ở các câu nằm ngay đường ranh giới cắt.
3. Kỹ thuật RAG Nâng cao (Advanced Retrieval Augmented Generation Concepts)

   - **Naive RAG**: Nhận câu hỏi -> Embedding -> Top-K Search -> Đưa vào Prompt -> Gọi LLM. (Yếu điểm: Dễ lấy nhầm chunk rác, nhạy cảm với câu hỏi mơ hồ).
   - **Hybrid Search (Search kết hợp)**: Kết hợp cả Dense Search (tìm kiếm vector theo ý nghĩa) + Sparse Search (BM25 - tìm kiếm theo từ khóa chính xác như Google).
   - **Re-ranking (Xếp hạng lại):** Sau khi retriever lấy ra Top 20 chunks gần nhất, một mô hình Cross-Encoder (như Cohere Rerank) sẽ chấm điểm chính xác lại từng chunk so với câu hỏi để chọn ra Top 3-5 chunks chất lượng nhất đưa vào Prompt.
