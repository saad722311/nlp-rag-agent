# NLP Research Paper Q&A — RAG Agent

> Ask questions about NLP research papers and get cited, grounded answers powered by local LLMs.

---

## What This Is

A **Retrieval-Augmented Generation (RAG)** chatbot built to answer questions about NLP research papers. Instead of relying on an LLM's training data alone, this agent retrieves relevant passages from actual papers (Transformers, BERT, GPT-2, Word2Vec, ELMo, and more) before generating an answer.

Everything runs **100% locally** — no API keys, no cloud costs.

---

## Tech Stack

- **LLM:** [Ollama](https://ollama.ai) — `qwen3:8b` (runs locally)
- **Embeddings:** `nomic-embed-text` via Ollama
- **Vector DB:** [ChromaDB](https://www.trychroma.com/)
- **Framework:** [LangChain](https://langchain.com/)
- **UI:** [Streamlit](https://streamlit.io/)
- **PDF Parsing:** PyMuPDF
- **Paper Fetching:** ArXiv API

---

## Demo

> Coming after Iteration 3

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nlp-rag-agent.git
cd nlp-rag-agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull required Ollama models
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 5. Run the app
streamlit run app.py
```

---

## Project Iterations

| Iteration | Description | Status |
|---|---|---|
| 1 | PDF ingestion pipeline → ChromaDB | In Progress |
| 2 | RAG chain + CLI Q&A | Not Started |
| 3 | Streamlit UI + ArXiv fetcher | Not Started |
| 4 | Evaluation with RAGAS | Not Started |
| 5 | Polish + README + Demo | Not Started |

---

## Seed Papers

- Attention Is All You Need — Vaswani et al. (2017)
- BERT — Devlin et al. (2019)
- Word2Vec — Mikolov et al. (2013)
- ELMo — Peters et al. (2018)

---

*Built as a portfolio project to demonstrate RAG pipeline design, LangChain, local LLM integration, and vector database usage.*
