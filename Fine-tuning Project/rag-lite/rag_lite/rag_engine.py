"""
RAG (Retrieval-Augmented Generation) core functionality.
"""

import os
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .indexer import DocumentIndexer


class RAGEngine:
    """Core RAG engine that combines retrieval and generation."""

    def __init__(self, 
                 indexer: DocumentIndexer,
                 model_name: str = "gpt2",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_k: int = 5):
        """
        Initialize the RAG engine.

        Args:
            indexer: Document indexer for retrieval
            model_name: Name of the language model to use
            max_length: Maximum length of generated text
            temperature: Sampling temperature for generation
            top_k: Number of documents to retrieve
        """
        self.indexer = indexer
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k

        # Initialize the language model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

    def generate(self, query: str, max_new_tokens: int = 256) -> Dict[str, Any]:
        """
        Generate a response given a query.

        Args:
            query: User query
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Dictionary containing response and retrieved context
        """
        # Retrieve relevant documents
        retrieved_docs = self.indexer.search(query, k=self.top_k)

        # Extract text from retrieved documents
        context = "\n\n".join([doc["text"] for doc in retrieved_docs])

        # Create prompt with context
        prompt = f"Context information:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response
        response = self.generator(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]

        # Extract just the answer part (after the prompt)
        answer = response[len(prompt):].strip()

        return {
            "query": query,
            "answer": answer,
            "context": retrieved_docs
        }

    def save(self, directory: str) -> None:
        """
        Save the RAG engine components.

        Args:
            directory: Directory to save the components in
        """
        os.makedirs(directory, exist_ok=True)

        # Save indexer
        indexer_dir = os.path.join(directory, "indexer")
        self.indexer.save(indexer_dir)

        # Save configuration
        import json
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_k": self.top_k
        }

        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, directory: str) -> 'RAGEngine':
        """
        Load a RAG engine from disk.

        Args:
            directory: Directory containing the saved engine

        Returns:
            Loaded RAGEngine instance
        """
        # Load configuration
        import json
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)

        # Load indexer
        indexer_dir = os.path.join(directory, "indexer")
        indexer = DocumentIndexer.load(indexer_dir)

        # Create engine with loaded components
        engine = cls(
            indexer=indexer,
            model_name=config["model_name"],
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_k=config["top_k"]
        )

        return engine
