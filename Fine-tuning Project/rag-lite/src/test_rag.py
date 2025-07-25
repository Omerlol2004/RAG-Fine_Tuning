"""
Simple script to test the RAG server using Python requests.
"""

import requests
import json
import sys
import time
from pathlib import Path


def test_health(server_url):
    """Test the server's health endpoint."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        print("✓ Server health check passed!")
        print(f"  Response: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_query(server_url, query="What is quantum computing?"):
    """Send a test query to the server."""
    try:
        # Prepare the payload
        payload = {"question": query}

        print(f"Sending query: {query}")
        print("Waiting for response (this might take a while for the first query)...")

        # Send request
        start_time = time.time()
        response = requests.post(
            f"{server_url}/query", 
            json=payload,
            timeout=180  # Much higher timeout for larger models
        )
        response.raise_for_status()

        elapsed = time.time() - start_time
        result = response.json()

        # Print results
        print(f"✓ Query successful! (took {elapsed:.2f} seconds)")
        print("\nQUERY RESULT:")
        print("-" * 80)
        print(f"Query: {result['query']}")
        print("-" * 80)
        print(f"Answer: {result['answer']}")
        print("-" * 80)
        print("Retrieved Documents:")

        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"[{i+1}] Similarity: {doc['similarity']:.4f}")
            print(f"    {doc['text'][:150]}...")

        return True
    except Exception as e:
        print(f"✗ Query failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error = e.response.json()
                print(f"  Server error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"  Status code: {e.response.status_code}")
        return False


def main():
    """Main test function."""
    server_url = "http://localhost:8001"

    print("=" * 80)
    print(f"RAG Server Test Script - Testing server at {server_url}")
    print("=" * 80)

    # Test server health
    print("\n1. Testing server health...")
    if not test_health(server_url):
        print("\nServer health check failed. Make sure the server is running.")
        print("Run 'uvicorn src.rag_server:app --host 0.0.0.0 --port 8001' in another terminal.")
        return 1

    # Test query
    print("\n2. Testing query endpoint...")
    if not test_query(server_url):
        print("\nQuery test failed. See error details above.")
        return 1

    # Success
    print("\n" + "=" * 80)
    print("All tests passed successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
