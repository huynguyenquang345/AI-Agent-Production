# 🤖 Technical Knowledge AI Agent (Production-Ready)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-orange.svg)](https://www.langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://www.docker.com/)
[![Status](<https://img.shields.io/badge/Status-Live%20Production-brightgreen.svg>)]()

> Hệ thống Trợ lý AI tra cứu tài liệu kỹ thuật tự động hóa quy trình (Agentic Workflow) sử dụng RAG nâng cao, đóng gói Microservices và đã vận hành thực tế trên Cloud


# AI Agent Production System

Hệ thống Multi-Agent AI kết hợp Advanced RAG phục vụ cho việc tự động hóa tuyển dụng, phân tích hồ sơ ứng viên và hỗ trợ đánh giá kỹ thuật. Dự án được tối ưu hóa chạy hoàn toàn bằng local models để đảm bảo tính bảo mật và tối ưu chi phí.

## 🚀 Công nghệ sử dụng (Tech Stack)

- **Language:** Python 3.11 / 3.12
- **Orchestration:** LangGraph (StateGraph, Multi-Agent Collaboration, Tool Calling)
- **RAG Framework:** LangChain, Qdrant Vector Store (In-memory / Persistent)
- **Local LLM & Embeddings:** Ollama (`qwen2.5:3b` cho suy luận agent, `nomic-embed-text` cho embedding)
- **Environment:** Windows, PowerShell, VS Code

## 📂 Cấu trúc Thư mục Dự án

- `week_1/`: Xây dựng pipeline Advanced RAG cơ bản và Agent đơn lẻ (Day 1 & Day 2).
- `week_2/`: Xây dựng hệ thống Multi-Agent phân chia vai trò (HR Screener & Tech Interviewer) với Router điều hướng (Day 3 & Day 4).
- `docs/`: Tài liệu thiết kế và giới thiệu chi tiết sản phẩm.

## 🛠️ Hướng Dẫn Cài Đặt & Chạy (Quick Start)

1. **Clone repository:**
   ```bash
   git clone <URL_REPOSITORY_CUA_BAN>
   AI-Agent-Production
   ```
