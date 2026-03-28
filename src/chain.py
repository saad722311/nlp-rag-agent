"""
chain.py
Builds the LangChain RAG chain:
  user query → ChromaDB retrieval → prompt with context → Ollama LLM → answer + sources
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.retriever import load_retriever

LLM_MODEL = "qwen3:8b"

PROMPT_TEMPLATE = """You are an expert AI assistant specializing in NLP research.
Answer the question using ONLY the context provided below from NLP research papers.
Be precise and cite which paper your answer comes from where possible.
If the context does not contain enough information to answer, say "I don't have enough information in the loaded papers to answer this."

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    """Combine retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_chain():
    """Build and return the RAG chain."""
    retriever = load_retriever()
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str) -> dict:
    """
    Ask a question and return the answer plus the source chunks used.
    Returns: {"answer": str, "sources": list[dict]}
    """
    chain, retriever = build_chain()

    # Retrieve source docs separately so we can display them
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "excerpt": doc.page_content[:200] + "...",
        }
        for doc in source_docs
    ]

    return {"answer": answer, "sources": sources}