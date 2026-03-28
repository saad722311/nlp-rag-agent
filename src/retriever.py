"""
retriever.py
Loads the persisted ChromaDB vector store and retrieves the top-K
most relevant chunks for a given query.
"""

import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
TOP_K = 5  # number of chunks to retrieve per query


def load_retriever():
    """Load the persisted ChromaDB vector store and return a retriever."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. Run ingest.py first."
        )

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    return retriever