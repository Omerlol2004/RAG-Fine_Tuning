"""
Single-file script for LoRA fine-tuning of LLMs on RAG data.

Usage:
    python src/train_lora.py --base llama --epochs 3 --json artifacts/train_qa.json
    python src/train_lora.py --base qwen --epochs 3 --json artifacts/train_qa.json
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    PeftModel, 
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training
)


# Model configurations
MODEL_CONFIGS = {
    "llama": {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "template": "[INST] <<SYS>>\nYou are a helpful assistant. Use the following context to answer the question.\nContext: {context}\n<</SYS>>\n\nQuestion: {query} [/INST] {answer}",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    "qwen": {
        "path": "Qwen/Qwen-7B-Chat",
        "template": "<|im_start|>system\nYou are a helpful assistant. Use the following context to answer the question.\nContext: {context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
}


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on RAG data")

    parser.add_argument(
        "--base",
        type=str,
        default="qwen",
        choices=["llama", "qwen"],
        help="Base model to use (llama or qwen)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON file containing training data"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/lora_out",
        help="Output directory for the trained model"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA attention dimension"
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )

    return parser.parse_args()


def print_trainable_parameters(model):
    """Print information about trainable parameters."""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")


def load_and_prepare_data(json_path, model_name, tokenizer):
    """Load and prepare data for training."""
    # Check if file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Training data file not found: {json_path}")

    # Load the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare formatted examples
    formatted_data = []
    template = MODEL_CONFIGS[model_name]["template"]

    for item in tqdm(data, desc="Formatting examples"):
        # Extract context and question from prompt
        prompt_parts = item["prompt"].split("\n\nQuestion: ")
        context = prompt_parts[0].replace("Context: ", "")
        query = prompt_parts[1] if len(prompt_parts) > 1 else ""
        answer = item["answer"]

        # Format according to the template
        formatted_text = template.format(
            context=context,
            query=query,
            answer=answer
        )

        formatted_data.append({"text": formatted_text})

    # Create dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split into train and validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    return split_dataset["train"], split_dataset["test"]


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Check if output directory exists, create if not
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be extremely slow on CPU.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting...")
            return

    # Get model configuration
    model_config = MODEL_CONFIGS.get(args.base)
    if not model_config:
        print(f"Error: Unknown base model '{args.base}'")
        return

    model_path = model_config["path"]
    print(f"Using model: {model_path}")

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Special handling for models
    if args.base == "qwen":
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=model_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Prepare model for training
    print("Preparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    print_trainable_parameters(model)

    # Load and prepare data
    print("Loading and preparing training data...")
    train_dataset, eval_dataset = load_and_prepare_data(
        args.json, 
        args.base,
        tokenizer
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the model
    print(f"Training complete. Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save configuration
    config_info = {
        "base_model": args.base,
        "model_path": model_path,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
