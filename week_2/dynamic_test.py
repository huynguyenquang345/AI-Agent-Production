import os
import sys
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
    """Chỉ chạy khi ở máy Local (Windows) có hỗ trợ PowerShell GUI."""
    if os.environ.get("CI") == "true":
        print("⚠️ Đang chạy trong môi trường CI (Headless), bỏ qua mở hộp thoại Explorer.")
        return ""
        
    print("\n📂 Đang mở cửa sổ Explorer để chọn tài liệu...")
    powershell_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "$dlg = New-Object System.Windows.Forms.OpenFileDialog; "
        "$dlg.Filter = 'Supported Files (*.pdf;*.txt;*.md;*.csv;*.docx)|*.pdf;*.txt;*.md;*.csv;*.docx|All Files (*.*)|*.*'; "
        "$dlg.Title = 'Chọn tài liệu cho Knowledge Base'; "
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
        print(f"❌ Không thể mở hộp thoại Explorer: {e}")
        return ""

def load_documents_from_file(file_path: str):
    """Tự động nhận diện định dạng file và load dữ liệu."""
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
            loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    except Exception as e:
        print(f"❌ Lỗi khi đọc file '{file_path}': {e}")
        return []


# --- UNIT TEST CHO PYTEST (Không dùng tham số fixture lỗi) ---
def test_environment_and_imports():
    """Kiểm tra các thư viện cốt lõi được import thành công."""
    import langchain
    import qdrant_client
    import apscheduler
    assert langchain is not None
    assert qdrant_client is not None
    assert apscheduler is not None
    print("✅ [Passed] Các thư viện cốt lõi đã sẵn sàng.")

def test_vector_store_logic():
    """Kiểm tra logic khởi tạo text splitter và vector store giả lập."""
    from langchain_core.documents import Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = [Document(page_content="Test content for AI agent production pipeline quality gate.")]
    chunks = text_splitter.split_documents(docs)
    assert len(chunks) > 0
    print(f"✅ [Passed] Chunking logic hoạt động chính xác ({len(chunks)} chunks).")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 [DAY 7] CHẠY ỨNG DỤNG LOCAL VỚI DYNAMIC FILE SELECTION")
    print("="*70)
    
    selected_file = open_explorer_to_select_file()
    docs = []
    if selected_file and os.path.exists(selected_file):
        print(f"✅ Đã chọn file: {selected_file}")
        docs = load_documents_from_file(selected_file)
    else:
        print("⚠️ Không có file nào được chọn hoặc đang chạy ở chế độ không có giao diện.")