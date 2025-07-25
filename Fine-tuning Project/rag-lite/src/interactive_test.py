"""
Interactive script to test the RAG server.
"""

import requests
import json
import sys
import time


def query_rag_server(server_url="http://localhost:8001", question="What is quantum computing?"):
    """Send a query to the RAG server and print the response."""
    try:
        # Prepare the payload
        payload = {"question": question}

        print(f"\nSending query: '{question}'")
        print("Waiting for response (this might take a while)...")

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
        print(f"\n✓ Response received! (took {elapsed:.2f} seconds)")
        print("=" * 80)
        print(f"QUESTION: {result['query']}")
        print("=" * 80)
        print(f"ANSWER: {result['answer']}")
        print("=" * 80)
        print("RETRIEVED DOCUMENTS:")

        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"[{i+1}] Similarity: {doc['similarity']:.4f}")
            print(f"    {doc['text'][:150]}...")

        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error = e.response.json()
                print(f"  Server error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"  Status code: {e.response.status_code}")
        return False


def main():
    """Main function to interactively query the RAG server."""
    print("=" * 80)
    print("RAG Server Interactive Testing")
    print("=" * 80)
    print("Enter questions to test the RAG system. Type 'exit' to quit.")

    while True:
        print()
        question = input("Enter your question: ")

        if question.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break

        if not question.strip():
            print("Please enter a valid question.")
            continue

        query_rag_server(question=question)
        print("\n" + "-" * 80)


if __name__ == "__main__":
    main()
