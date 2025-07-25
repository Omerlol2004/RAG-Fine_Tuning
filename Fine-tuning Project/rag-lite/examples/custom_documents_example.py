"""
Example demonstrating how to use RAG-Lite with custom document collection.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.document_processor import DocumentProcessor
from rag_lite.indexer import DocumentIndexer
from rag_lite.rag_engine import RAGEngine


def create_sample_docs():
    """Create some sample document files for demonstration."""
    os.makedirs("sample_docs", exist_ok=True)

    # Sample documents about various topics
    docs = {
        "ai.txt": """
Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.
Machine learning is a subset of AI focused on developing systems that learn from data.
Deep learning is a type of machine learning based on artificial neural networks.
Natural Language Processing (NLP) is a field of AI that gives machines the ability to read, understand, and derive meaning from human languages.
        """,

        "planets.txt": """
The solar system consists of the Sun and everything that orbits around it.
There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
Earth is the third planet from the Sun and the only astronomical object known to harbor life.
Mars is often called the Red Planet due to its reddish appearance.
Jupiter is the largest planet in our solar system.
        """,

        "history.txt": """
World War II was a global war that lasted from 1939 to 1945, involving many of the world's nations.
The Renaissance was a period in European history marking the transition from the Middle Ages to modernity.
The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States.
Ancient Egypt was a civilization of ancient North Africa, concentrated along the lower reaches of the Nile River.
        """
    }

    # Write sample documents to files
    for filename, content in docs.items():
        with open(os.path.join("sample_docs", filename), "w", encoding="utf-8") as f:
            f.write(content)

    return os.path.abspath("sample_docs")


def main():
    """Run an example with custom document collection."""
    # Create sample documents
    docs_dir = create_sample_docs()
    print(f"Created sample documents in: {docs_dir}")

    # Create document processor
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # Load documents from directory
    print("Processing documents...")
    documents = processor.load_directory(docs_dir)
    print(f"Processed {len(documents)} document chunks")

    # Create an indexer and add documents
    print("Creating document index...")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.add_documents(documents)

    # Create a RAG engine
    print("Initializing RAG engine...")
    engine = RAGEngine(indexer=indexer, model_name="gpt2")

    # Interactive query loop
    print("\nRAG-Lite Interactive Demo")
    print("Ask questions about AI, planets, or history (type 'exit' to quit)")

    while True:
        query = input("\nQuestion: ")
        if query.lower() in ("exit", "quit"):
            break

        # Process query
        result = engine.generate(query)

        # Display the result
        print("\nAnswer:")
        print(result['answer'])

        print("\nTop retrieved documents:")
        for i, doc in enumerate(result['context'][:3]):  # Show top 3 docs
            print(f"[{i+1}] Score: {doc['score']:.4f}")
            if 'metadata' in doc and doc['metadata']:
                source = doc['metadata'].get('source', 'unknown')
                print(f"    Source: {source}")
            print(f"    Text: {doc['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()
