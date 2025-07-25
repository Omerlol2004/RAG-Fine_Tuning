"""
Utilities for training LoRA adapters on RAG data.
"""

import os
import torch
import json
from typing import List, Dict, Any, Optional, Union
from datasets import Dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig

from .advanced_models import AdvancedLanguageModel


class LoRATrainer:
    """Class for training LoRA adapters on RAG data."""

    def __init__(
        self,
        model_name: str = "qwen",
        model_path: Optional[str] = None,
        output_dir: str = "lora-rag-model",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        """
        Initialize the LoRA trainer.

        Args:
            model_name: Name of the base model (llama2 or qwen)
            model_path: Custom path to model if not using predefined models
            output_dir: Directory to save the trained model
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
        """
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Create model
        self.model = AdvancedLanguageModel(
            model_name=model_name,
            model_path=model_path,
            quantize=True
        )

        # Define LoRA config
        self.lora_config = AdvancedLanguageModel.create_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # Prepare model for training
        self.model.prepare_for_training(self.lora_config)

    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format a prompt based on the model type."""
        context = example.get("context", "")
        query = example.get("query", "")
        answer = example.get("answer", "")

        if self.model_name.lower() == "llama2":
            return f"[INST] <<SYS>>\nYou are a helpful assistant. Use the following context to answer the question.\nContext: {context}\n<</SYS>>\n\nQuestion: {query} [/INST] {answer}"
        elif self.model_name.lower() == "qwen":
            return f"<|im_start|>system\nYou are a helpful assistant. Use the following context to answer the question.\nContext: {context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        else:
            return f"Context information:\n{context}\n\nQuestion: {query}\nAnswer: {answer}"

    def prepare_dataset(self, examples: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare a dataset for training from examples.

        Args:
            examples: List of dictionaries with 'context', 'query', and 'answer' keys

        Returns:
            HuggingFace dataset ready for training
        """
        # Format prompts for all examples
        formatted_data = []

        for example in examples:
            formatted_prompt = self._format_prompt(example)
            formatted_data.append({
                "text": formatted_prompt
            })

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.model.tokenizer(examples["text"], truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 100,
        logging_steps: int = 10
    ) -> None:
        """
        Train the LoRA adapter.

        Args:
            train_dataset: Dataset for training
            eval_dataset: Optional dataset for evaluation
            num_train_epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            gradient_accumulation_steps: Number of steps for gradient accumulation
            save_steps: Save checkpoint every this many steps
            logging_steps: Log metrics every this many steps
        """
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),  # Use mixed precision when available
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,  # Only keep the last 3 checkpoints
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="tensorboard"
        )

        # Create trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model.tokenizer,
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(self.output_dir)

        # Save the tokenizer and config
        self.model.tokenizer.save_pretrained(self.output_dir)

        # Save a training config file
        training_config = {
            "model_name": self.model_name,
            "model_path": self.model.model_path,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "training_args": {
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps
            }
        }

        with open(os.path.join(self.output_dir, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2)

        print(f"LoRA adapter trained and saved to {self.output_dir}")

    @staticmethod
    def generate_synthetic_data(
        rag_outputs: List[Dict[str, Any]], 
        n_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate training examples from RAG outputs.

        Args:
            rag_outputs: List of RAG outputs (with query, answer, context)
            n_samples: Number of samples to generate (if None, uses all)

        Returns:
            List of training examples
        """
        examples = []

        for i, output in enumerate(rag_outputs):
            if n_samples is not None and i >= n_samples:
                break

            # Extract relevant information
            query = output["query"]
            answer = output["answer"]

            # Extract context from retrieved documents
            context_docs = output.get("context", [])
            context = "\n\n".join([doc.get("text", "") for doc in context_docs])

            # Create example
            example = {
                "context": context,
                "query": query,
                "answer": answer
            }

            examples.append(example)

        return examples
