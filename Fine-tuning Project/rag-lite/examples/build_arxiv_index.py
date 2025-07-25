"""
Script to download and index a small ArXiv sample dataset (5000 abstracts).
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pathlib
import tqdm
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.indexer import DocumentIndexer
from rag_lite.rag_engine import RAGEngine

# Create artifacts directory
ART_DIR = pathlib.Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

def main():
    print("Downloading ArXiv dataset (5000 abstracts)...")
    ds = load_dataset("ccdv/arxiv-classification", split="train[:5000]")

    # Extract texts and save them
    texts = ds["text"]
    labels = ds["label"]
    categories = ds["label_text"]

    print(f"Downloaded {len(texts)} abstracts")

    # Save raw texts for later use
    np.save(ART_DIR/"abstracts.npy", np.array(texts))
    print(f"Saved abstracts to {ART_DIR/'abstracts.npy'}")

    # Process data into documents
    documents = []
    for i, (text, label, category) in enumerate(zip(texts, labels, categories)):
        doc = {
            "id": f"arxiv_{i}",
            "text": text,
            "metadata": {
                "label": int(label),
                "category": category
            }
        }
        documents.append(doc)

    print("Creating document index...")

    # Method 1: Using SentenceTransformer and FAISS directly
    print("Method 1: Using SentenceTransformer and FAISS directly")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs.astype("float32"))
    faiss.write_index(index, str(ART_DIR/"arxiv.faiss"))
    print(f"Saved FAISS index to {ART_DIR/'arxiv.faiss'}")

    # Method 2: Using RAG-Lite's built-in classes
    print("\nMethod 2: Using RAG-Lite's built-in classes")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.add_documents(documents)

    # Create RAG engine
    engine = RAGEngine(indexer=indexer)

    # Save the RAG engine
    engine.save(str(ART_DIR/"arxiv_rag"))
    print(f"Saved RAG engine to {ART_DIR/'arxiv_rag'}")

    # Test a simple query
    query = "Tell me about quantum computing"
    print(f"\nTesting query: '{query}'")
    result = engine.generate(query)
    print(f"Answer: {result['answer']}")

    print("\nTop retrieved documents:")
    for i, doc in enumerate(result['context'][:3]):
        print(f"[{i+1}] Score: {doc['score']:.4f}")
        if 'metadata' in doc:
            category = doc['metadata'].get('category', 'unknown')
            print(f"    Category: {category}")
        print(f"    Text: {doc['text'][:150]}...")

if __name__ == "__main__":
    main()
