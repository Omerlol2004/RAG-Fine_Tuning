"""
Simple example demonstrating how to use RAG-Lite as a library.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.document_processor import DocumentProcessor
from rag_lite.indexer import DocumentIndexer
from rag_lite.rag_engine import RAGEngine


def main():
    """Run a simple RAG example."""
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "text": "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre Museum.",
            "metadata": {"source": "geography.txt"}
        },
        {
            "id": "doc2",
            "text": "Tokyo is the capital of Japan. It is the most populous metropolitan area in the world.",
            "metadata": {"source": "geography.txt"}
        },
        {
            "id": "doc3",
            "text": "Python is a popular programming language. It was created by Guido van Rossum.",
            "metadata": {"source": "programming.txt"}
        }
    ]

    # Create an indexer and add documents
    print("Creating document index...")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.add_documents(documents)

    # Create a RAG engine
    print("Initializing RAG engine...")
    engine = RAGEngine(indexer=indexer, model_name="gpt2")

    # Process a query
    query = "What is the capital of France?"
    print(f"\nProcessing query: '{query}'")
    result = engine.generate(query)

    # Display the result
    print("\nResult:")
    print(f"Answer: {result['answer']}\n")

    print("Context documents used:")
    for i, doc in enumerate(result['context']):
        print(f"[{i+1}] Score: {doc['score']:.4f}")
        if 'metadata' in doc:
            print(f"    Source: {doc['metadata'].get('source', 'unknown')}")
        print(f"    Text: {doc['text']}")
        print()

    # Try saving and loading the engine
    save_dir = "example_model"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving model to {save_dir}...")
    engine.save(save_dir)

    print(f"Loading model from {save_dir}...")
    loaded_engine = RAGEngine.load(save_dir)

    # Verify loaded engine works
    print("Testing loaded engine...")
    new_query = "Tell me about Python."
    result = loaded_engine.generate(new_query)
    print(f"Answer to '{new_query}': {result['answer']}")


if __name__ == "__main__":
    main()
