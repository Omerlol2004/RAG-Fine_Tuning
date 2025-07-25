"""
Example demonstrating the Enhanced RAG system with Qwen-7B-Chat model.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.document_processor import DocumentProcessor
from rag_lite.indexer import DocumentIndexer
from rag_lite.enhanced_rag import EnhancedRAG


def create_sample_index():
    """Create a simple document index for demonstration."""
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
        },
        {
            "id": "doc4",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve from experience.",
            "metadata": {"source": "tech.txt"}
        },
        {
            "id": "doc5",
            "text": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to create more accurate responses.",
            "metadata": {"source": "tech.txt"}
        }
    ]

    # Create an indexer and add documents
    print("Creating document index...")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.add_documents(documents)

    return indexer


def main():
    """Run the enhanced RAG example."""
    print("Enhanced RAG Example with Qwen-7B-Chat")
    print("======================================")

    # Create a sample index
    indexer = create_sample_index()

    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. This example will run very slowly on CPU.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            return

    # Initialize the enhanced RAG system
    print("\nInitializing enhanced RAG with Qwen-7B-Chat...")
    print("This may take a few minutes to download and load the model...")

    rag = EnhancedRAG(
        indexer=indexer,
        model_name="qwen",  # Using Qwen-7B-Chat
        temperature=0.7,
        top_k=3
    )

    print("\nEnhanced RAG system initialized!")

    # Interactive query loop
    print("\nEnter your questions (type 'exit' to quit):")

    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("exit", "quit"):
            break

        print("Generating response...")
        result = rag.generate(query)

        print("\nAnswer:")
        print(result["answer"])

        print("\nRetrieved documents:")
        for i, doc in enumerate(result["context"]):
            print(f"[{i+1}] Score: {doc['score']:.4f}")
            if 'metadata' in doc:
                print(f"    Source: {doc['metadata'].get('source', 'unknown')}")
            print(f"    Text: {doc['text']}")


if __name__ == "__main__":
    main()
