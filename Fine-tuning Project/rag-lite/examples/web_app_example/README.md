# RAG-Lite Web App Example

This example demonstrates how to create a web application that uses the RAG-Lite API.

## Prerequisites

- RAG-Lite installed and set up
- Streamlit and requests libraries

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Start the RAG-Lite API server in a terminal:

```bash
# Navigate to the RAG-Lite root directory
cd ../..

# Start the server with your model directory
python -m rag_lite.cli serve --model-dir /path/to/model_dir
```

3. Run the Streamlit app in another terminal:

```bash
streamlit run app.py
```

The web interface should open automatically in your browser.

## Usage

1. In the left column, add documents to the knowledge base
2. In the right column, ask questions about the added documents
3. Adjust the temperature and max tokens parameters as needed

The app will display the answer along with the source documents used to generate it.
