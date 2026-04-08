import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf.errors import DependencyError

# Alle PDFs im Ordner 'data/' finden
pdf_files = glob(os.path.join("data", "*.pdf"))
print(f"Found {len(pdf_files)} PDFs:", pdf_files)

all_chunks = []

for pdf_path in pdf_files:
    # Nur PDFs mit 'english' im Dateinamen laden
    if "english" not in os.path.basename(pdf_path).lower():
        print(f"Skipping non-English PDF: {pdf_path}")
        continue

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Text in Chunks splitten (optimiert für RAG)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        docs_chunks = text_splitter.split_documents(documents)
        all_chunks.extend(docs_chunks)

        print(f"Loaded and split: {pdf_path} ({len(docs_chunks)} chunks)")

    except DependencyError as e:
        print(f"Skipped encrypted PDF: {pdf_path} ({e})")
    except Exception as e:
        print(f"Error loading {pdf_path}: {e}")

print(f"\nTotal chunks from all English PDFs: {len(all_chunks)}")

if all_chunks:
    print("First chunk preview:")
    print(all_chunks[0].page_content[:500], "...")