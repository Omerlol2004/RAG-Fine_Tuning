# RAG-Lite: GPU-Optimized RAG System

A lightweight Retrieval-Augmented Generation (RAG) implementation optimized for GPU acceleration with sophisticated prompt engineering and quality controls.

## Overview
RAG-Lite provides a simplified RAG implementation that combines document retrieval with generative AI to produce more accurate, contextually relevant responses based on your own documents.

## Features
- **GPU Acceleration**: Optimized for consumer GPUs (tested on RTX 3050) with quantization support
- **Smart Relevance Detection**: Automatically identifies if retrieved documents are actually relevant
- **Quality Control**: Detects and prevents hallucination and common LLM failure patterns
- **Document indexing and retrieval** using FAISS vector database
- **Semantic search** with sentence transformers
- **Context-aware text generation** with Hugging Face transformers
- **FastAPI-based web service**
- **Command-line interface** for easy interaction
- **Advanced LLM Support**: Integration with powerful models like Phi-1.5, Qwen-7B-Chat and Llama-2-7B-Chat
- **LoRA Fine-tuning**: Ability to fine-tune models on domain-specific data with efficient LoRA adapters

## Getting Started

### Environment Setup
You can set up the required environment using conda:

#### Windows
```bash
# Run the provided setup script
setup_env.bat
```

#### Linux/macOS
```bash
# Make the script executable
chmod +x setup_env.sh

# Run the provided setup script
./setup_env.sh
```

Or manually:
```bash
# Create and activate conda environment
conda create -n raglite python=3.10 -y
conda activate raglite

# Install dependencies
pip install -r requirements.txt
```

### Installation
After setting up the environment, install the package:

```bash
# Install in development mode
pip install -e .
```

## Usage

### Index Documents
```bash
python -m rag_lite.cli index --input /path/to/documents --output /path/to/model_dir
```

### Query Documents
```bash
# Single query
python -m rag_lite.cli query --model-dir /path/to/model_dir --query "Your question here?"

# Interactive mode
python -m rag_lite.cli query --model-dir /path/to/model_dir
```

### Start API Server
```bash
python -m rag_lite.cli serve --model-dir /path/to/model_dir --host 127.0.0.1 --port 8000
```

## API Endpoints

- `POST /documents/`: Add documents to the RAG system
- `POST /query/`: Query the RAG system
- `POST /save/`: Save the current state to disk
- `GET /health/`: Health check endpoint

## Examples

The repository includes several example applications to help you get started:

### Simple Example
A basic example showing how to use RAG-Lite as a Python library:
```bash
python examples/simple_example.py
```

### Custom Documents Example
A more advanced example that processes a custom document collection:
```bash
python examples/custom_documents_example.py
```

### Web Application Example
A Streamlit-based web interface that interacts with the RAG-Lite API:
```bash
# First, start the API server
python -m rag_lite.cli serve --model-dir /path/to/model_dir

# Then, in a new terminal, run the web app
cd examples/web_app_example
pip install -r requirements.txt
streamlit run app.py
```

### ArXiv Dataset Example
An example that downloads and indexes 5,000 papers from ArXiv:
```bash
python examples/build_arxiv_index.py
python examples/query_arxiv_index.py
```

### Enhanced RAG with Advanced LLMs
An example demonstrating the use of Qwen-7B-Chat with RAG:
```bash
python examples/enhanced_rag_example.py
```

### LoRA Fine-tuning Example
An example showing how to fine-tune a large language model on RAG data:
```bash
python examples/lora_training_example.py
```

## Fine-tuning and Deployment

The repository also includes streamlined scripts for LoRA fine-tuning and server deployment:

### Fine-tune with LoRA
Fine-tune a large language model (Qwen or Llama) with LoRA on RAG data:
```bash
# Generate synthetic QA pairs
python examples/make_synthetic_qa.py

# Fine-tune with LoRA
python src/train_lora.py --base qwen --epochs 3 --json artifacts/train_qa.json
```

### RAG Server with Fine-tuned Model
Deploy a FastAPI server using the fine-tuned model:
```bash
# Start the server
python src/start_server.py --model qwen

# Query the server (interactive mode)
python src/rag_client.py

# Query the server (single query)
python src/rag_client.py --query "What are neural networks?"
```

See the README in the `src` directory for more details on these scripts.

Check the README files in each example directory for more details.

## GPU Optimizations

RAG-Lite has been extensively optimized for GPU usage, with key improvements including:

- **Quantization Support**: 4-bit and 8-bit quantization to reduce memory footprint while maintaining quality
- **Batch Processing**: Optimized embedding generation with batched processing
- **Prompt Engineering**: Sophisticated prompts tailored for each model type (Llama, Phi, Qwen, etc.)
- **Response Quality Controls**: Multiple quality filters to prevent common LLM failure modes
- **Relevance Detection**: Automatic detection and handling of low-relevance documents
- **Generation Parameters**: Fine-tuned parameters for optimal quality-speed trade-off

## Project Structure

```
rag-lite/
├── rag_lite/               # Main package
│   ├── __init__.py         # Package initialization
│   ├── indexer.py          # Document indexing and vector storage
│   ├── document_processor.py # Document loading and processing
│   ├── rag_engine.py       # Core RAG functionality
│   ├── api.py              # FastAPI implementation
│   ├── cli.py              # Command-line interface
│   ├── advanced_models.py  # Support for advanced LLMs (Qwen, Llama)
│   ├── enhanced_rag.py     # Enhanced RAG with advanced LLMs
│   └── lora_trainer.py     # Utilities for LoRA fine-tuning
├── examples/               # Example applications
├── src/                    # Streamlined scripts
│   ├── train_lora.py       # Single-file LoRA fine-tuning
│   ├── rag_server.py       # FastAPI server with fine-tuned model
│   ├── rag_client.py       # Client for the RAG server
│   └── start_server.py     # Helper script to start the server
├── setup.py                # Package setup script
└── requirements.txt        # Dependencies
```

## License
This project is open source and available under the MIT License.

