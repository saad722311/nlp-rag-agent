"""
arxiv_fetcher.py
Downloads NLP papers from ArXiv by ID and saves them to data/papers/.
"""

import arxiv
import os

# Landmark NLP papers to seed the vector store
SEED_PAPERS = [
    {"id": "1706.03762", "name": "attention_is_all_you_need"},   # Transformers
    {"id": "1810.04805", "name": "bert"},                         # BERT
    {"id": "1301.3781",  "name": "word2vec"},                     # Word2Vec
    {"id": "1802.05365", "name": "elmo"},                         # ELMo
]

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "papers")


def fetch_paper(arxiv_id: str, filename: str) -> str:
    """Download a single paper by ArXiv ID. Returns the saved file path."""
    os.makedirs(PAPERS_DIR, exist_ok=True)
    filepath = os.path.join(PAPERS_DIR, f"{filename}.pdf")

    if os.path.exists(filepath):
        print(f"  [skip] {filename}.pdf already exists")
        return filepath

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))

    print(f"  [downloading] {paper.title}")
    paper.download_pdf(dirpath=PAPERS_DIR, filename=f"{filename}.pdf")
    print(f"  [saved] {filename}.pdf")
    return filepath


def fetch_all_seed_papers():
    """Download all seed NLP papers."""
    print("Fetching seed NLP papers from ArXiv...\n")
    paths = []
    for p in SEED_PAPERS:
        path = fetch_paper(p["id"], p["name"])
        paths.append(path)
    print(f"\nDone. {len(paths)} papers ready in data/papers/")
    return paths


if __name__ == "__main__":
    fetch_all_seed_papers()