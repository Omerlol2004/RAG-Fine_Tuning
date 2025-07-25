# RAG-Lite Source Directory

This directory contains scripts for LoRA fine-tuning and deploying a RAG server with the fine-tuned model.

## Scripts

### `train_lora.py`

Fine-tunes a large language model with LoRA on RAG data.

```bash
python src/train_lora.py --base llama --epochs 3 --json artifacts/train_qa.json
```

Options:
- `--base`: Base model to use (`llama` or `qwen`)
- `--epochs`: Number of training epochs (default: 3)
- `--json`: Path to training data JSON file
- `--output`: Output directory for the trained model (default: artifacts/lora_out)
- `--batch-size`: Training batch size (default: 4)
- `--lora-r`: LoRA attention dimension (default: 16)
- `--lora-alpha`: LoRA alpha parameter (default: 16)
- `--lora-dropout`: LoRA dropout probability (default: 0.05)

### `rag_server.py`

FastAPI server implementing RAG with a fine-tuned model.

### `start_server.py`

Helper script to start the RAG server with specific settings.

```bash
python src/start_server.py --model qwen --port 8000
```

Options:
- `--model`: Base model to use (`qwen` or `llama`)
- `--port`: Port to run the server on (default: 8000)
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--no-lora`: Don't use LoRA weights (use only base model)

### `rag_client.py`

Client script to interact with the RAG server.

```bash
# Interactive mode
python src/rag_client.py

# Single query mode
python src/rag_client.py --query "What are neural networks?"
```

Options:
- `--url`: RAG server URL (default: http://localhost:8000)
- `--max-length`: Maximum answer length (default: 512)
- `--temperature`: Generation temperature (default: 0.7)
- `--query`: Query to send (if not specified, enters interactive mode)

## Workflow

1. Generate synthetic QA pairs:
   ```bash
   python examples/make_synthetic_qa.py
   ```

2. Fine-tune the model using LoRA:
   ```bash
   python src/train_lora.py --base qwen --epochs 3 --json artifacts/train_qa.json
   ```

3. Start the RAG server with the fine-tuned model:
   ```bash
   python src/start_server.py --model qwen
   ```

4. Query the server:
   ```bash
   python src/rag_client.py
   ```
