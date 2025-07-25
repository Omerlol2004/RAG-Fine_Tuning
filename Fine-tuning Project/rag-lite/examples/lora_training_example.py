"""
Example demonstrating how to fine-tune a model with LoRA on RAG data.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.document_processor import DocumentProcessor
from rag_lite.indexer import DocumentIndexer
from rag_lite.rag_engine import RAGEngine
from rag_lite.lora_trainer import LoRATrainer


def generate_training_data():
    """Generate sample training data for LoRA fine-tuning."""
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
            "text": "Python is a programming language created by Guido van Rossum. It is known for its readability and versatility.",
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
        },
        {
            "id": "doc6",
            "text": "Climate change refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels.",
            "metadata": {"source": "environment.txt"}
        },
        {
            "id": "doc7",
            "text": "The Internet was developed in the late 1960s as ARPANET, a network connecting research institutions in the United States.",
            "metadata": {"source": "tech_history.txt"}
        },
        {
            "id": "doc8",
            "text": "Quantum computing uses quantum bits or qubits that can represent both 0 and 1 simultaneously, enabling much faster computation for certain problems.",
            "metadata": {"source": "quantum.txt"}
        },
        {
            "id": "doc9",
            "text": "Renewable energy comes from sources that are naturally replenishing, such as sunlight, wind, rain, tides, and geothermal heat.",
            "metadata": {"source": "energy.txt"}
        },
        {
            "id": "doc10",
            "text": "DNA (deoxyribonucleic acid) is the genetic material that carries the instructions for development and functioning of living organisms.",
            "metadata": {"source": "biology.txt"}
        }
    ]

    # Sample queries
    queries = [
        "What is the capital of France?",
        "Tell me about Tokyo.",
        "Who created Python?",
        "Explain machine learning in simple terms.",
        "What is RAG in AI?",
        "What causes climate change?",
        "How was the Internet developed?",
        "Explain quantum computing.",
        "What are renewable energy sources?",
        "What is DNA and what does it do?"
    ]

    # Create an indexer and RAG engine for generating answers
    print("Creating document index...")
    indexer = DocumentIndexer(model_name="all-MiniLM-L6-v2")
    indexer.add_documents(documents)

    engine = RAGEngine(indexer=indexer)

    # Generate RAG outputs for the queries
    print("Generating RAG outputs for training...")
    rag_outputs = []

    for query in queries:
        print(f"Processing query: {query}")
        result = engine.generate(query)
        rag_outputs.append(result)

    # Create training examples using the LoRATrainer utility
    print("Creating training examples...")
    examples = LoRATrainer.generate_synthetic_data(rag_outputs)

    # Save examples to a JSON file for inspection
    os.makedirs("training_data", exist_ok=True)
    with open("training_data/sample_data.json", "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Created {len(examples)} training examples")
    print(f"Examples saved to training_data/sample_data.json")

    return examples


def main():
    """Run the LoRA training example."""
    print("LoRA Training Example")
    print("====================")

    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be extremely slow on CPU.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            return

    # Generate training data
    examples = generate_training_data()

    # Initialize the LoRA trainer
    print("\nInitializing LoRA trainer...")
    print("This will download and load the model, which may take several minutes...")

    trainer = LoRATrainer(
        model_name="qwen",  # Using Qwen-7B-Chat
        output_dir="lora_rag_model",
        lora_r=8,
        lora_alpha=16
    )

    # Prepare dataset
    print("Preparing training dataset...")
    dataset = trainer.prepare_dataset(examples)

    # Split into train and eval
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Train the model
    print("\nStarting LoRA training...")
    print("Note: Training on a CPU will be extremely slow.")
    print("Press Ctrl+C to stop training at any point.")

    try:
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_train_epochs=1,  # For demonstration, use just 1 epoch
            learning_rate=2e-4,
            batch_size=1,  # Small batch size for CPU/limited memory
            gradient_accumulation_steps=8,
            save_steps=10,
            logging_steps=5
        )

        print("\nTraining complete!")
        print(f"LoRA adapter saved to {trainer.output_dir}")
        print("\nYou can now use this adapter with the EnhancedRAG class:")
        print("rag = EnhancedRAG(indexer=indexer, model_name='qwen', lora_weights='lora_rag_model')")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        print("Partial training results may have been saved.")
        print("You can continue training by initializing a new LoRATrainer with the same output_dir.")


if __name__ == "__main__":
    main()
