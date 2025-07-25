"""
Simple client to interact with the RAG server.
"""

import requests
import argparse
import json
from typing import Dict, Any


def query_server(
    query: str,
    server_url: str = "http://localhost:8000",
    max_length: int = 512,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Send a query to the RAG server.

    Args:
        query: The question to ask
        server_url: URL of the RAG server
        max_length: Maximum length of the generated answer
        temperature: Temperature for text generation

    Returns:
        Server response as dictionary
    """
    # Prepare request
    endpoint = f"{server_url}/query"
    payload = {
        "query": query,
        "max_length": max_length,
        "temperature": temperature
    }

    # Send request
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json().get("detail", "Unknown error")
                print(f"Server error: {error_detail}")
            except:
                print(f"Server returned status code {e.response.status_code}")
        return None


def print_response(response: Dict[str, Any]) -> None:
    """
    Print a formatted RAG response.

    Args:
        response: Response from the RAG server
    """
    if not response:
        return

    print("
" + "=" * 80)
    print(f"QUERY: {response['query']}")
    print("=" * 80)

    print("
ANSWER:")
    print(response["answer"])

    print("
RETRIEVED DOCUMENTS:")
    for i, doc in enumerate(response["retrieved_documents"]):
        print(f"[{i+1}] Similarity: {doc['similarity']:.4f}")
        print(f"    {doc['text']}")
        print()

    print("-" * 80)


def main():
    """Main function to interact with the RAG server."""
    parser = argparse.ArgumentParser(description="Client for RAG server")
    parser.add_argument("--url", default="http://localhost:8000", help="RAG server URL")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum answer length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--query", help="Query to send (if not specified, enters interactive mode)")

    args = parser.parse_args()

    # Check if server is available
    try:
        health_check = requests.get(f"{args.url}/health")
        health_check.raise_for_status()
        print(f"Connected to RAG server at {args.url}")
    except requests.exceptions.RequestException:
        print(f"Error: Cannot connect to RAG server at {args.url}")
        print("Make sure the server is running and accessible.")
        return

    # Single query mode
    if args.query:
        response = query_server(
            query=args.query,
            server_url=args.url,
            max_length=args.max_length,
            temperature=args.temperature
        )
        print_response(response)
        return

    # Interactive mode
    print("
RAG Interactive Query Mode")
    print("Enter your questions (type 'exit' to quit)")

    while True:
        try:
            query = input("
Question: ")
            if query.lower() in ("exit", "quit"):
                break

            print("Generating response...")
            response = query_server(
                query=query,
                server_url=args.url,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print_response(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
