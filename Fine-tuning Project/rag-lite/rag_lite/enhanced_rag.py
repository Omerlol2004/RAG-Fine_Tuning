"""
Enhanced RAG engine with support for advanced language models and LoRA fine-tuning.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import torch

from .indexer import DocumentIndexer
from .advanced_models import AdvancedLanguageModel


class EnhancedRAG:
    """Enhanced RAG system with advanced LLM support."""

    def __init__(
        self,
        indexer: DocumentIndexer,
        model_name: str = "qwen",
        model_path: Optional[str] = None,
        lora_weights: Optional[str] = None,
        template: Optional[str] = None,
        temperature: float = 0.7,
        top_k: int = 5
    ):
        """
        Initialize the enhanced RAG engine.

        Args:
            indexer: Document indexer for retrieval
            model_name: Name of the language model to use (llama2 or qwen)
            model_path: Custom path to model if not using predefined models
            lora_weights: Path to LoRA weights if using fine-tuned model
            template: Prompt template to use (if None, uses default)
            temperature: Sampling temperature for generation
            top_k: Number of documents to retrieve
        """
        self.indexer = indexer
        self.temperature = temperature
        self.top_k = top_k

        # Load the advanced language model
        self.model = AdvancedLanguageModel(
            model_name=model_name,
            model_path=model_path,
            lora_weights=lora_weights,
            quantize=True
        )

        # Set template based on model or use custom
        if template:
            self.template = template
        elif model_name.lower() == "llama2":
            self.template = "[INST] <<SYS>>\nYou are a helpful assistant. Use the following context to answer the question.\nContext: {context}\n<</SYS>>\n\nQuestion: {query} [/INST]"
        elif model_name.lower() == "qwen":
            self.template = "<|im_start|>system\nYou are a helpful assistant. Use the following context to answer the question.\nContext: {context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        else:
            self.template = "Context information:\n{context}\n\nQuestion: {query}\nAnswer:"

    def generate(self, query: str, max_new_tokens: int = 512) -> Dict[str, Any]:
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
        prompt = self.template.format(context=context, query=query)

        # Generate response
        answer = self.model.generate(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=self.temperature
        )

        return {
            "query": query,
            "answer": answer,
            "context": retrieved_docs
        }

    def save(self, directory: str) -> None:
        """
        Save the RAG components.

        Args:
            directory: Directory to save the components in
        """
        os.makedirs(directory, exist_ok=True)

        # Save indexer
        indexer_dir = os.path.join(directory, "indexer")
        self.indexer.save(indexer_dir)

        # Save configuration
        config = {
            "model_name": self.model.model_name,
            "model_path": self.model.model_path,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "template": self.template
        }

        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, directory: str, lora_weights: Optional[str] = None) -> 'EnhancedRAG':
        """
        Load a RAG engine from disk.

        Args:
            directory: Directory containing the saved engine
            lora_weights: Optional path to LoRA weights to use (overrides saved config)

        Returns:
            Loaded EnhancedRAG instance
        """
        # Load configuration
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)

        # Load indexer
        indexer_dir = os.path.join(directory, "indexer")
        indexer = DocumentIndexer.load(indexer_dir)

        # Create engine with loaded components
        engine = cls(
            indexer=indexer,
            model_name=config["model_name"],
            model_path=config.get("model_path"),
            lora_weights=lora_weights,  # Use provided LoRA weights if any
            template=config.get("template"),
            temperature=config.get("temperature", 0.7),
            top_k=config.get("top_k", 5)
        )

        return engine
