# NLP Research Paper Q&A — RAG Agent

## Project Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about NLP research papers.
Built as a CV/portfolio project to showcase AI Engineering expertise.

**GitHub Repo:** https://github.com/saad722311/nlp-rag-agent
**Owner:** Muhammad Saad
**Purpose:** Portfolio project — demonstrates RAG pipeline, LangChain, local LLMs, vector DBs

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Ollama — `qwen3:8b` (locally installed) |
| Embeddings | Ollama — `nomic-embed-text` |
| Vector DB | ChromaDB (local, persisted to disk) |
| RAG Framework | LangChain + langchain-community + langchain-ollama |
| PDF Parsing | PyMuPDF (fitz) |
| Paper Fetching | `arxiv` Python library (ArXiv API) |
| UI | Streamlit |
| Evaluation | RAGAS |
| Language | Python 3.x |

---

## Project Structure

```
RAG-chatbot/
├── data/
│   └── papers/            ← downloaded NLP PDFs
├── src/
│   ├── ingest.py          ← PDF → chunks → ChromaDB
│   ├── retriever.py       ← query ChromaDB
│   ├── chain.py           ← LangChain RAG chain
│   ├── arxiv_fetcher.py   ← download papers by ID/topic
│   └── evaluate.py        ← RAGAS metrics
├── chroma_db/             ← persisted vector store (gitignored)
├── app.py                 ← Streamlit UI entry point
├── requirements.txt
├── .gitignore
├── CLAUDE.md              ← this file
└── README.md
```

---

## Seed NLP Papers (ArXiv IDs)

| Paper | Authors | Year | ArXiv ID |
|---|---|---|---|
| Attention Is All You Need (Transformers) | Vaswani et al. | 2017 | 1706.03762 |
| BERT | Devlin et al. | 2019 | 1810.04805 |
| GPT-2 | Radford et al. | 2019 | (OpenAI blog, manual download) |
| Word2Vec | Mikolov et al. | 2013 | 1301.3781 |
| ELMo | Peters et al. | 2018 | 1802.05365 |

---

## Build Iterations

### Iteration 1 — Data Ingestion Pipeline
**Status:** Complete
**Goal:** Download NLP PDFs → parse → chunk → embed → store in ChromaDB
**Files:** `src/ingest.py`, `src/arxiv_fetcher.py`, `requirements.txt`, `.gitignore`
**Key decisions:**
- Chunk size: 500 tokens, 50 token overlap
- Embedding model: nomic-embed-text via Ollama
- ChromaDB persisted to `./chroma_db/`

---

### Iteration 2 — RAG Chain + CLI
**Status:** Not started
**Goal:** Query ChromaDB → build prompt → LLM answer → show sources
**Files:** `src/retriever.py`, `src/chain.py`
**Key decisions:**
- Top-K retrieval: 5 chunks
- LLM: qwen3:8b via Ollama
- Prompt instructs model to answer ONLY from provided context

---

### Iteration 3 — Streamlit UI
**Status:** Not started
**Goal:** Full chat UI with paper upload, ArXiv fetcher, source citations
**Files:** `app.py`
**Key decisions:**
- Show source paper + page number with every answer
- Allow uploading custom PDFs
- Allow fetching by ArXiv ID directly in UI

---

### Iteration 4 — Evaluation
**Status:** Not started
**Goal:** Measure RAG quality with RAGAS metrics
**Files:** `src/evaluate.py`
**Metrics:** Faithfulness, Answer Relevance, Context Recall

---

### Iteration 5 — Polish & CV-Ready
**Status:** Not started
**Goal:** Clean README, demo GIF, deploy (optional HuggingFace Spaces or local Docker)
**Files:** `README.md`, `demo.gif`

---

## Current Status

**Active Iteration:** Iteration 2 — RAG Chain + CLI
**Last GitHub Push:** Iteration 1
**Next Action:** Build retriever.py + chain.py, test Q&A in terminal

---

## How to Run (updated per iteration)

```bash
# Activate venv
source .venv/bin/activate

# Step 1: Download papers
python src/arxiv_fetcher.py

# Step 2: Ingest into ChromaDB
python src/ingest.py
```

---

## Important Notes for Future Claude Sessions

- The user is building this step-by-step, one iteration at a time
- After EACH iteration is complete: update this file's "Current Status" section and push to GitHub
- Ollama is running locally — do NOT suggest OpenAI API or paid services unless asked
- Available Ollama models: `deepseek-r1:8b`, `qwen3:8b`, `nomic-embed-text` (to be pulled)
- The user wants to push to GitHub after every iteration
- Keep code clean and well-commented — this is a portfolio project
- GitHub repo name: `nlp-rag-agent`
