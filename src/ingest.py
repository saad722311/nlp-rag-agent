"""
ingest.py
Parses PDFs → splits into chunks → embeds with nomic-embed-text → stores in ChromaDB.
Run this once to build the vector store, then re-run whenever you add new papers.
"""

import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns list of {text, page, source}."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:  # skip blank pages
            pages.append({
                "text": text,
                "page": page_num,
                "source": os.path.basename(pdf_path),
            })
    doc.close()
    return pages


def load_all_pdfs() -> list[dict]:
    """Load all PDFs from the papers directory."""
    all_pages = []
    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PAPERS_DIR}. Run arxiv_fetcher.py first.")

    for filename in pdf_files:
        path = os.path.join(PAPERS_DIR, filename)
        print(f"  [reading] {filename}")
        pages = extract_text_from_pdf(path)
        all_pages.extend(pages)
        print(f"           {len(pages)} pages extracted")

    return all_pages


def chunk_pages(pages: list[dict]) -> tuple[list[str], list[dict]]:
    """Split page texts into smaller chunks. Returns (texts, metadatas)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    texts, metadatas = [], []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "source": page["source"],
                "page": page["page"],
            })

    return texts, metadatas


def build_vector_store(texts: list[str], metadatas: list[dict]):
    """Embed chunks with nomic-embed-text and persist to ChromaDB."""
    print(f"\n  [embedding] {len(texts)} chunks with nomic-embed-text...")
    print("  (this may take a few minutes on first run)\n")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR,
    )

    print(f"  [saved] Vector store persisted to {CHROMA_DIR}")
    return vectorstore


def run_ingestion():
    """Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB."""
    print("=" * 50)
    print("  NLP RAG Agent — Ingestion Pipeline")
    print("=" * 50)

    print("\n[1/3] Loading PDFs...")
    pages = load_all_pdfs()
    print(f"      Total pages loaded: {len(pages)}")

    print("\n[2/3] Chunking text...")
    texts, metadatas = chunk_pages(pages)
    print(f"      Total chunks created: {len(texts)}")

    print("\n[3/3] Embedding and storing in ChromaDB...")
    build_vector_store(texts, metadatas)

    print("\nIngestion complete!")
    print(f"  Papers processed : {len(set(m['source'] for m in metadatas))}")
    print(f"  Total chunks     : {len(texts)}")
    print(f"  Vector store     : {CHROMA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    run_ingestion()