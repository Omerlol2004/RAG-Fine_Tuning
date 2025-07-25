"""
Advanced language model support with LoRA configuration for RAG-Lite.
"""

import os
import torch
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import (
    PeftModel,
    PeftConfig,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


class AdvancedLanguageModel:
    """Advanced language model with LoRA support for RAG-Lite."""

    SUPPORTED_MODELS = {
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "qwen": "Qwen/Qwen-7B-Chat"
    }

    def __init__(
        self,
        model_name: str = "qwen",
        model_path: Optional[str] = None,
        lora_weights: Optional[str] = None,
        quantize: bool = True,
        device_map: str = "auto"
    ):
        """
        Initialize an advanced language model with optional LoRA weights.

        Args:
            model_name: Name of the model (llama2 or qwen)
            model_path: Custom path to model if not using predefined models
            lora_weights: Path to LoRA weights if using fine-tuned model
            quantize: Whether to use quantization (4-bit) for memory efficiency
            device_map: Device mapping strategy
        """
        self.model_name = model_name
        self.model_path = model_path or self.SUPPORTED_MODELS.get(model_name.lower())

        if not self.model_path:
            raise ValueError(
                f"Unknown model: {model_name}. Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        print(f"Loading {self.model_path}...")

        # Configure quantization if enabled
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Add special handling for specific models
        if model_name.lower() == "qwen":
            self.tokenizer.padding_side = "left"
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add LoRA adapter if weights are provided
        if lora_weights:
            print(f"Loading LoRA weights from {lora_weights}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights
            )

        # Create generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text based on the prompt.

        Args:
            prompt: Input prompt for generation
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
            **kwargs
        }

        # Generate response
        response = self.generator(
            prompt,
            **generation_kwargs
        )[0]["generated_text"]

        # Strip the prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    @staticmethod
    def create_lora_config(
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ) -> LoraConfig:
        """
        Create a LoRA configuration.

        Args:
            r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of modules to apply LoRA to

        Returns:
            LoRA configuration
        """
        # Default target modules by model type if not specified
        if target_modules is None:
            # These are examples - actual target modules may vary by model
            target_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj"
            ]

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )

        return config

    def prepare_for_training(self, lora_config: Optional[LoraConfig] = None) -> None:
        """
        Prepare the model for LoRA fine-tuning.

        Args:
            lora_config: LoRA configuration, will create default if None
        """
        if lora_config is None:
            lora_config = self.create_lora_config()

        # Prepare the model for k-bit training if using quantization
        if hasattr(self.model, "is_quantized") and self.model.is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)

        # Add LoRA adapter
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters info
        self.print_trainable_parameters()

    def print_trainable_parameters(self) -> None:
        """Print information about trainable parameters."""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")
