# RAG-Lite Examples

This directory contains example applications that demonstrate how to use RAG-Lite in various scenarios.

## Available Examples

### 1. Simple Example (`simple_example.py`)

A basic example showing how to use RAG-Lite as a Python library with predefined documents.

**Run with:**
```bash
python simple_example.py
```

**Features demonstrated:**
- Creating an indexer and adding documents
- Initializing the RAG engine
- Processing queries
- Saving and loading models

### 2. Custom Documents Example (`custom_documents_example.py`)

A more advanced example that creates and processes a custom document collection.

**Run with:**
```bash
python custom_documents_example.py
```

**Features demonstrated:**
- Creating sample document files
- Processing documents from files
- Interactive querying

### 3. Web Application Example (`web_app_example/`)

A Streamlit-based web interface that interacts with the RAG-Lite API.

**Setup and run:**
```bash
# First, start the RAG-Lite API server
cd ..
python -m rag_lite.cli serve --model-dir /path/to/model_dir

# Then, in a new terminal, run the web app
cd examples/web_app_example
pip install -r requirements.txt
streamlit run app.py
```

**Features demonstrated:**
- Interacting with the RAG-Lite API
- Building a web interface
- Adding documents through a UI
- Querying with adjustable parameters

## Usage Tips

1. Make sure you have installed RAG-Lite before running these examples
2. The examples assume you are running them from the examples directory
3. For the web app example, make sure the API server is running before starting the Streamlit app
