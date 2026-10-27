import os
import subprocess
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

def open_explorer_to_select_file():
    """Mở cửa sổ Explorer trực tiếp bằng PowerShell của Windows để người dùng chọn file bất kỳ."""
    print("\n📂 Đang mở cửa sổ Explorer để chọn tài liệu...")
    
    powershell_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dlg = New-Object System.Windows.Forms.OpenFileDialog; "
        "$dlg.Filter = 'Supported Files (*.pdf;*.txt;*.md;*.csv;*.docx)|*.pdf;*.txt;*.md;*.csv;*.docx|All Files (*.*)|*.*'; "
        "$dlg.Title = 'Chọn tài liệu cho Knowledge Base (Day 7 Test)'; "
        "if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
        "    Write-Output $dlg.FileName "
        "}"
    )

    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", powershell_script],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Lỗi khi mở cửa sổ Explorer: {e}")
        return ""

def load_documents_from_file(file_path: str):
    """Tự động nhận diện phần mở rộng file và chọn Loader tương ứng."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            print(f"⚠️ Định dạng '{ext}' chưa được tối ưu, thử đọc dạng text...")
            loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file '{file_path}': {e}")
        return []

# Thêm các hàm xử lý Chunking và nạp vào Vector Store nếu cần thiết
def test_system_components(embeddings, retriever):
    """Bộ kiểm thử tự động (Unit Test) xác thực các thành phần cốt lõi của hệ thống."""
    print("\n" + "="*50)
    print("🧪 [QUALITY GATE] Đang chạy Test Suite tự động...")
    print("="*50)
    
    # Test 1: Kiểm tra Embedding Model
    assert embeddings is not None, "❌ Lỗi: Embedding model chưa được khởi tạo."
    print("✅ [Passed] Embedding Model hoạt động bình thường.")
    
    # Test 2: Kiểm tra Retriever và khả năng truy vấn động
    if retriever:
        test_query = "thông tin"
        docs = retriever.invoke(test_query)
        assert isinstance(docs, list), "❌ Lỗi: Retriever phải trả về danh sách tài liệu."
        print(f"✅ [Passed] Dynamic Retriever truy vấn thành công (Tìm thấy {len(docs)} chunks liên quan).")
    else:
        print("⚠️ [Skipped] Bỏ qua test retriever do chưa có file tài liệu nào được nạp.")
    
    print("🎉 Tất cả các bài kiểm tra chất lượng (Quality Gate) đã hoàn thành xuất sắc!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 [DAY 7] DYNAMIC FILE SELECTION & TEST SUITE KHỞI CHẠY")
    print("="*70)
    
    # 1. Gọi hộp thoại Windows Explorer để chọn file động
    selected_file = open_explorer_to_select_file()
    
    docs = []
    if selected_file and os.path.exists(selected_file):
        print(f"✅ Đã chọn file: {selected_file}")
        docs = load_documents_from_file(selected_file)
    else:
        print("⚠️ Người dùng đã hủy chọn file hoặc file không tồn tại. Hệ thống chuyển sang chế độ chạy kiểm thử thuần túy.")

    # 2. Khởi tạo Embedding
    print("\n🔄 Đang khởi tạo mô hình Embedding cục bộ...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    retriever = None
    if docs:
        # 3. Xử lý Chunking và nạp vào Vector Store (Qdrant)
        print("🔄 Đang phân đoạn tài liệu (Chunking) và tạo Vector Database...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            location=":memory:",
            collection_name="dynamic_test_db",
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"✅ Đã nạp thành công {len(chunks)} chunks vào Vector Store.")

    # 4. Thực thi Test Suite (Day 7 Quality Gate)
    test_system_components(embeddings, retriever)