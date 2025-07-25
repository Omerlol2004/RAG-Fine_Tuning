"""
Script to query the ArXiv index built with build_arxiv_index.py.
"""

import numpy as np
import faiss
import pathlib
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.rag_engine import RAGEngine

# Path to artifacts
ART_DIR = pathlib.Path("artifacts")

def query_direct(query_text, top_k=5):
    """Query using direct FAISS and SentenceTransformer approach."""
    # Load the abstracts and index
    try:
        abstracts = np.load(ART_DIR/"abstracts.npy")
        index = faiss.read_index(str(ART_DIR/"arxiv.faiss"))
    except FileNotFoundError:
        print("Error: Could not find index files. Please run build_arxiv_index.py first.")
        return

    # Load the model and encode the query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query_text], normalize_embeddings=True)

    # Search
    scores, indices = index.search(query_vec.astype("float32"), top_k)

    # Print results
    print(f"Results for query: '{query_text}'\n")

    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"[{i+1}] Score: {score:.4f}")
        print(f"Abstract: {abstracts[idx][:200]}...\n")


def query_rag_lite(query_text):
    """Query using the RAG-Lite engine."""
    try:
        # Load the saved RAG engine
        engine = RAGEngine.load(str(ART_DIR/"arxiv_rag"))
    except Exception as e:
        print(f"Error loading RAG engine: {e}")
        print("Please run build_arxiv_index.py first.")
        return

    # Generate response
    result = engine.generate(query_text)

    # Print results
    print(f"Answer for query: '{query_text}'\n")
    print(result["answer"])

    print("\nSources:")
    for i, doc in enumerate(result["context"][:5]):
        print(f"[{i+1}] Score: {doc['score']:.4f}")
        if 'metadata' in doc and doc['metadata']:
            category = doc['metadata'].get('category', 'unknown')
            print(f"    Category: {category}")
        print(f"    Text: {doc['text'][:150]}...\n")


def main():
    if not ART_DIR.exists():
        print("Error: Artifacts directory not found. Please run build_arxiv_index.py first.")
        return

    print("ArXiv Query Demo")
    print("================\n")

    # Interactive query loop
    while True:
        print("Enter your query (or 'exit' to quit):")
        query = input("> ")

        if query.lower() in ("exit", "quit"):
            break

        print("\n1) Direct FAISS Query Results:")
        query_direct(query)

        print("\n2) RAG-Lite Generated Answer:")
        query_rag_lite(query)

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()
