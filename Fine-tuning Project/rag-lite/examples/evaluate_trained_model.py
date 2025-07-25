"""
Script to evaluate the trained LoRA model on ArXiv queries.
"""

import os
import sys
import json
from pathlib import Path
import random
import torch
import numpy as np

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.document_processor import DocumentProcessor
from rag_lite.indexer import DocumentIndexer
from rag_lite.enhanced_rag import EnhancedRAG


def create_arxiv_index():
    """Create or load an ArXiv document index."""
    # Path to artifacts
    artifacts_dir = Path("artifacts")

    # Check if we have the abstracts file
    if not os.path.exists(artifacts_dir/"abstracts.npy"):
        print("Error: Could not find abstracts.npy in the artifacts directory.")
        print("Please run build_arxiv_index.py first to download the ArXiv dataset.")
        return None

    # Check if we have a pre-built index
    if os.path.exists(artifacts_dir/"arxiv.faiss"):
        print("Found existing ArXiv index, loading...")
        # Load the abstracts
        abstracts = np.load(artifacts_dir/"abstracts.npy", allow_pickle=True)

        # Convert abstracts to documents
        documents = []
        for i, text in enumerate(abstracts):
            doc = {
                "id": f"arxiv_{i}",
                "text": text
            }
            documents.append(doc)

        # Create indexer and load the index
        indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")

        # A simple hack to add documents without re-computing embeddings
        indexer.documents = documents
        indexer.index = faiss.read_index(str(artifacts_dir/"arxiv.faiss"))

        return indexer

    else:
        print("No pre-built index found, creating from abstracts...")
        # Load the abstracts
        abstracts = np.load(artifacts_dir/"abstracts.npy", allow_pickle=True)

        # Convert abstracts to documents
        documents = []
        for i, text in enumerate(abstracts):
            doc = {
                "id": f"arxiv_{i}",
                "text": text
            }
            documents.append(doc)

        # Create indexer and add documents
        indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
        indexer.add_documents(documents)

        return indexer


def generate_evaluation_questions():
    """Generate evaluation questions about scientific topics."""
    eval_questions = [
        "What are the applications of transformer models in computer vision?",
        "How does quantum computing differ from classical computing?",
        "What are the latest advances in natural language processing?",
        "How do neural networks learn representations?",
        "What are the challenges in reinforcement learning?",
        "How can machine learning be applied to climate science?",
        "What is the current state of research in generative AI?",
        "How do graph neural networks work?",
        "What are the ethical considerations in AI research?",
        "How is deep learning applied to medical imaging?"
    ]
    return eval_questions


def main():
    """Evaluate the trained LoRA model on ArXiv queries."""
    print("Evaluating Trained LoRA Model on ArXiv Queries")
    print("=============================================")

    # Define paths
    artifacts_dir = Path("artifacts")
    lora_model_dir = artifacts_dir / "qa_lora_model"

    # Check if the LoRA model exists
    if not lora_model_dir.exists():
        print(f"Error: LoRA model not found at {lora_model_dir}")
        print("Please run train_lora_on_qa.py first to train the model.")
        return

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Inference will be slow on CPU.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            return

    # Create or load the ArXiv index
    print("Setting up document index...")
    indexer = create_arxiv_index()
    if indexer is None:
        return

    # Get evaluation questions
    eval_questions = generate_evaluation_questions()

    # Initialize the RAG systems
    print("\nInitializing RAG systems...")

    # First, initialize a vanilla RAG system without LoRA for comparison
    try:
        print("1. Initializing vanilla EnhancedRAG (without LoRA)...")
        vanilla_rag = EnhancedRAG(
            indexer=indexer,
            model_name="qwen",
            temperature=0.7,
            top_k=5
        )
        have_vanilla = True
    except Exception as e:
        print(f"Error initializing vanilla RAG: {e}")
        print("Continuing with only the LoRA model...")
        have_vanilla = False

    # Then, initialize a RAG system with the trained LoRA adapter
    print("2. Initializing EnhancedRAG with trained LoRA adapter...")
    try:
        lora_rag = EnhancedRAG(
            indexer=indexer,
            model_name="qwen",
            lora_weights=str(lora_model_dir),
            temperature=0.7,
            top_k=5
        )
        print("LoRA model loaded successfully!")
    except Exception as e:
        print(f"Error initializing LoRA RAG: {e}")
        return

    # Evaluate the models
    print("\nStarting evaluation...")
    results = []

    for i, question in enumerate(eval_questions):
        print(f"\nQuestion {i+1}/{len(eval_questions)}: {question}")

        # Get responses
        if have_vanilla:
            print("\nGenerating vanilla response...")
            vanilla_result = vanilla_rag.generate(question)
            print(f"Vanilla response: {vanilla_result['answer'][:100]}...")

        print("\nGenerating LoRA response...")
        lora_result = lora_rag.generate(question)
        print(f"LoRA response: {lora_result['answer'][:100]}...")

        # Record results
        result = {
            "question": question,
            "lora_answer": lora_result["answer"]
        }

        if have_vanilla:
            result["vanilla_answer"] = vanilla_result["answer"]

        results.append(result)

    # Save results
    output_path = artifacts_dir / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete! Results saved to {output_path}")
    print("Compare the vanilla and LoRA responses to see the impact of fine-tuning.")


if __name__ == "__main__":
    main()
