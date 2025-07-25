"""
Script to train a LoRA adapter on synthetic QA pairs generated from ArXiv abstracts.
"""

import os
import sys
import json
from pathlib import Path
import random
import torch
from datasets import Dataset

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

from rag_lite.lora_trainer import LoRATrainer


def prepare_training_data(qa_file_path):
    """Prepare training data from QA pairs."""
    # Load the QA pairs
    with open(qa_file_path, 'r') as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs")

    # Convert QA pairs to training examples
    training_examples = []

    for pair in qa_pairs:
        # Extract the context and question
        prompt_parts = pair["prompt"].split("\n\nQuestion: ")
        context = prompt_parts[0].replace("Context: ", "")
        question = prompt_parts[1] if len(prompt_parts) > 1 else ""

        # Create training example
        example = {
            "context": context,
            "query": question,
            "answer": pair["answer"]
        }

        training_examples.append(example)

    return training_examples


def main():
    """Train a LoRA adapter on synthetic QA pairs."""
    print("LoRA Training on Synthetic QA Pairs")
    print("===================================")

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be extremely slow on CPU.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            return

    # Define paths
    artifacts_dir = Path("artifacts")
    qa_file_path = artifacts_dir / "train_qa.json"
    output_dir = artifacts_dir / "qa_lora_model"

    # Check if the QA file exists
    if not qa_file_path.exists():
        print(f"Error: QA file not found at {qa_file_path}")
        print("Please run make_synthetic_qa.py first to generate the QA pairs.")
        return

    # Prepare training data
    print("Preparing training data...")
    training_examples = prepare_training_data(qa_file_path)

    # Split into training and validation sets
    random.shuffle(training_examples)
    split_idx = int(len(training_examples) * 0.9)  # 90% for training, 10% for validation
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Initialize LoRA trainer
    print("\nInitializing LoRA trainer...")
    print("This will download and load the model, which may take several minutes...")

    trainer = LoRATrainer(
        model_name="qwen",  # Using Qwen-7B-Chat
        output_dir=str(output_dir),
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = trainer.prepare_dataset(train_examples)
    val_dataset = trainer.prepare_dataset(val_examples)

    # Train the model
    print("\nStarting LoRA training...")
    print("Note: Training on a CPU will be extremely slow.")
    print("Press Ctrl+C to stop training at any point.")

    try:
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            num_train_epochs=3,
            learning_rate=2e-4,
            batch_size=4,  # Adjust based on your GPU memory
            gradient_accumulation_steps=4,
            save_steps=100,
            logging_steps=20
        )

        print("\nTraining complete!")
        print(f"LoRA adapter saved to {output_dir}")
        print("\nYou can now use this adapter with the EnhancedRAG class:")
        print(f"rag = EnhancedRAG(indexer=indexer, model_name='qwen', lora_weights='{output_dir}')")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        print("Partial training results may have been saved.")


if __name__ == "__main__":
    main()
