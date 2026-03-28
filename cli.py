"""
cli.py
Interactive command-line interface for the NLP RAG Agent.
Run: python cli.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.chain import ask

BANNER = """
╔══════════════════════════════════════════════════════╗
║         NLP Research Paper Q&A — RAG Agent          ║
║  Papers: Transformers, BERT, Word2Vec, ELMo         ║
║  Type 'exit' or 'quit' to stop                      ║
╚══════════════════════════════════════════════════════╝
"""


def print_answer(result: dict):
    print("\n" + "─" * 54)
    print("ANSWER:")
    print(result["answer"])
    print("\nSOURCES:")
    for i, src in enumerate(result["sources"], 1):
        print(f"  [{i}] {src['source']}  (page {src['page']})")
        print(f"      \"{src['excerpt']}\"")
    print("─" * 54 + "\n")


def main():
    print(BANNER)
    print("Loading RAG chain (first query may take a moment)...\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        print("\nThinking...\n")
        result = ask(question)
        print_answer(result)


if __name__ == "__main__":
    main()