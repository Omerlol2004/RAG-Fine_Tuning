"""
Command-line interface for RAG-Lite.
"""

import os
import argparse
import sys
from typing import List, Dict, Any
from pathlib import Path

from .rag_engine import RAGEngine
from .indexer import DocumentIndexer
from .document_processor import DocumentProcessor


def create_parser():
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG-Lite: A lightweight Retrieval-Augmented Generation tool"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    index_parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    index_parser.add_argument("--output", "-o", required=True, help="Output directory for the index")
    index_parser.add_argument("--chunk-size", type=int, default=200, help="Size of document chunks in words")
    index_parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap between chunks in words")
    index_parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model to use")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--model-dir", "-m", required=True, help="Directory with the saved model")
    query_parser.add_argument("--query", "-q", help="Query text (if not provided, enters interactive mode)")
    query_parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Generation temperature")
    query_parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Start the API server")
    server_parser.add_argument("--model-dir", "-m", required=True, help="Directory with the saved model")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    server_parser.add_argument("--port", "-p", type=int, default=8000, help="Port number")

    return parser


def index_command(args):
    """Handle the 'index' command."""
    # Create document processor
    processor = DocumentProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # Create indexer
    indexer = DocumentIndexer(model_name=args.model)

    # Process input (file or directory)
    input_path = Path(args.input)
    if input_path.is_file():
        print(f"Processing file: {input_path}")
        documents = processor.load_file(str(input_path))
    elif input_path.is_dir():
        print(f"Processing directory: {input_path}")
        documents = processor.load_directory(
            str(input_path),
            recursive=args.recursive
        )
    else:
        print(f"Error: Input path {input_path} does not exist.")
        return 1

    print(f"Processed {len(documents)} document chunks")

    # Add documents to index
    print("Adding documents to index...")
    indexer.add_documents(documents)

    # Create RAG engine
    engine = RAGEngine(indexer=indexer)

    # Save the model
    output_dir = Path(args.output)
    print(f"Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    engine.save(str(output_dir))

    print("Indexing complete!")
    return 0


def query_command(args):
    """Handle the 'query' command."""
    # Load the model
    try:
        print(f"Loading model from {args.model_dir}")
        engine = RAGEngine.load(args.model_dir)
        engine.temperature = args.temperature
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Handle single query or interactive mode
    if args.query:
        response = engine.generate(args.query, max_new_tokens=args.max_tokens)
        print_response(response)
    else:
        # Interactive mode
        print("RAG-Lite Interactive Mode")
        print("Enter your questions (type 'exit' to quit)")
        while True:
            try:
                query = input("\n> ")
                if query.lower() in ("exit", "quit"):
                    break

                response = engine.generate(query, max_new_tokens=args.max_tokens)
                print_response(response)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    return 0


def serve_command(args):
    """Handle the 'serve' command."""
    # Set environment variable for the API to find the model
    os.environ["RAG_MODEL_DIR"] = args.model_dir

    # Import and run the API server
    from .api import app
    import uvicorn

    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def print_response(response):
    """Print a formatted response."""
    print("\n" + "=" * 40)
    print(f"ANSWER: {response['answer']}")
    print("=" * 40)
    print("\nSOURCES:")

    for i, doc in enumerate(response['context']):
        print(f"[{i+1}] Score: {doc['score']:.4f}")
        if 'metadata' in doc and doc['metadata']:
            source = doc['metadata'].get('source', 'unknown')
            print(f"    Source: {source}")
        print(f"    Text: {doc['text'][:100]}...")
        print()


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "index":
        return index_command(args)
    elif args.command == "query":
        return query_command(args)
    elif args.command == "serve":
        return serve_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
