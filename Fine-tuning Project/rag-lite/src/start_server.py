"""
Script to start the RAG server with proper settings.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    """Parse arguments and start the RAG server."""
    parser = argparse.ArgumentParser(description="Start the RAG server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument(
        "--model", 
        choices=["qwen", "llama"], 
        default="qwen",
        help="Base model to use (qwen or llama)"
    )
    parser.add_argument(
        "--no-lora", 
        action="store_true",
        help="Don't use LoRA weights"
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ["PORT"] = str(args.port)

    # Create a temporary file to modify the server configuration
    server_file = Path("src/rag_server.py")
    temp_file = Path("src/temp_server.py")

    if not server_file.exists():
        print(f"Error: Server file not found at {server_file}")
        return 1

    # Read the server file
    content = server_file.read_text()

    # Modify the content based on the arguments
    if args.model == "llama":
        content = content.replace(
            'BASE_MODEL = "Qwen/Qwen-7B-Chat"',
            'BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"'
        )

    if args.no_lora:
        content = content.replace(
            'LORA_PATH = "artifacts/lora_out"',
            'LORA_PATH = ""  # No LoRA weights'
        )

    # Write the modified content to the temporary file
    temp_file.write_text(content)

    try:
        # Start the server
        print(f"Starting RAG server with {args.model} model" + (" (no LoRA)" if args.no_lora else ""))
        print(f"Server will be available at http://localhost:{args.port}")

        # Use uvicorn to run the server
        cmd = [
            "uvicorn", 
            "src.temp_server:app", 
            "--host", args.host,
            "--port", str(args.port),
            "--reload"
        ]

        process = subprocess.run(cmd)
        return process.returncode

    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()


if __name__ == "__main__":
    sys.exit(main())
