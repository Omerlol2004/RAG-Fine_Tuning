"""
Script to generate synthetic QA pairs from ArXiv abstracts for fine-tuning.
"""

import json
import random
import numpy as np
import pathlib
import sys
from pathlib import Path
from tqdm import tqdm
import os

# Add the parent directory to the path to import rag_lite
sys.path.append(str(Path(__file__).parent.parent))

# Ensure artifacts directory exists
ART_DIR = pathlib.Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

def main():
    """Generate synthetic QA pairs from ArXiv abstracts."""

    # Check if we have the abstracts file
    if not os.path.exists(ART_DIR/"abstracts.npy"):
        print("Error: Could not find abstracts.npy in the artifacts directory.")
        print("Please run build_arxiv_index.py first to download the ArXiv dataset.")
        return

    # Load the abstracts
    print("Loading ArXiv abstracts...")
    texts = np.load(ART_DIR/"abstracts.npy", allow_pickle=True)

    # Templates for generating questions
    templates = [
        "Write one exam question the paper could answer.",
        "Ask a researcher about the paper's main topic.",
        "What is the key finding of this research?",
        "What methodology is described in this abstract?",
        "What problem does this research address?",
        "What are the implications of this study?",
        "How does this research contribute to the field?",
        "What is novel about this approach?",
        "Summarize the main point of this abstract.",
        "What theoretical framework is used in this research?"
    ]

    # Generate QA pairs
    print("Generating 1,000 synthetic QA pairs...")
    qa = []

    for t in tqdm(texts[:1000]):
        # Select a random template
        q = random.choice(templates)

        # Get a snippet from the abstract
        snippet = " ".join(t.split()[:40])

        # Use the first sentence as a simple answer
        first_sentence = t.split(".")[0] + "."

        # Create the QA pair
        qa.append({
            "prompt": f"Context: {snippet}\n\nQuestion: {q}",
            "answer": first_sentence
        })

    # Save the QA pairs to a JSON file
    output_path = ART_DIR/"train_qa.json"
    output_path.write_text(json.dumps(qa, indent=2))

    print(f"Generated {len(qa)} QA pairs")
    print(f"Saved to {output_path}")

    # Print a sample
    print("\nSample QA pair:")
    sample = random.choice(qa)
    print("Prompt:")
    print(sample["prompt"])
    print("\nAnswer:")
    print(sample["answer"])


if __name__ == "__main__":
    main()
