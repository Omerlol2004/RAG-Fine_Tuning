"""
Example web application using RAG-Lite API.
"""

import os
import requests
import streamlit as st
import json


# API Configuration
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"


def add_documents(text, metadata=None):
    """Add documents to the RAG API."""
    url = f"{API_URL}/documents/"

    # Create a document
    document = {
        "text": text
    }

    if metadata:
        document["metadata"] = metadata

    # Send request
    response = requests.post(url, json=[document])
    return response.json()


def query_rag(query_text, temperature=0.7, max_new_tokens=256):
    """Query the RAG API."""
    url = f"{API_URL}/query/"

    # Create query request
    query = {
        "query": query_text,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens
    }

    # Send request
    response = requests.post(url, json=query)
    return response.json()


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="RAG-Lite Demo", page_icon="🔍", layout="wide")

    st.title("RAG-Lite Web Demo")
    st.markdown("""
    This is a simple web interface for the RAG-Lite system. You can add documents 
    and ask questions about them.

    **Note:** Make sure the RAG-Lite API is running with:
    ```
    python -m rag_lite.cli serve --model-dir /path/to/model_dir
    ```
    """)

    # Check API status
    try:
        response = requests.get(f"{API_URL}/health/")
        if response.status_code == 200:
            st.success("✅ Connected to RAG-Lite API")
        else:
            st.error("❌ API is running but returned an error")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the RAG-Lite API. Make sure it's running.")
        st.stop()

    # Create two columns
    col1, col2 = st.columns(2)

    # Document adding section
    with col1:
        st.header("Add Documents")

        doc_text = st.text_area(
            "Document text", 
            height=200,
            placeholder="Enter document text to add to the knowledge base..."
        )

        source_name = st.text_input("Source name (optional)", "")

        if st.button("Add Document"):
            if doc_text:
                metadata = {"source": source_name} if source_name else None
                with st.spinner("Adding document..."):
                    result = add_documents(doc_text, metadata)
                    st.success(f"Document added! {result}")
            else:
                st.warning("Please enter document text")

    # Query section
    with col2:
        st.header("Ask Questions")

        query_text = st.text_input("Question", "")

        col_temp, col_tokens = st.columns(2)
        with col_temp:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        with col_tokens:
            max_tokens = st.slider("Max tokens", 50, 500, 256, 50)

        if st.button("Ask"):
            if query_text:
                with st.spinner("Generating answer..."):
                    result = query_rag(query_text, temperature, max_tokens)

                    # Display answer
                    st.subheader("Answer")
                    st.write(result["answer"])

                    # Display sources
                    st.subheader("Sources")
                    for i, doc in enumerate(result["context"]):
                        with st.expander(f"Source {i+1} (Score: {doc['score']:.4f})"):
                            if "metadata" in doc and doc["metadata"]:
                                st.write(f"**Source:** {doc['metadata'].get('source', 'unknown')}")
                            st.write(doc["text"])
            else:
                st.warning("Please enter a question")


if __name__ == "__main__":
    main()
