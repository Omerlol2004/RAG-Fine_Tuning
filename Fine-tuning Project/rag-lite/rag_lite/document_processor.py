"""
Document loading and processing functionality.
"""

import os
import re
from typing import List, Dict, Any, Callable, Optional, Union
import nltk
from bs4 import BeautifulSoup
import pandas as pd


# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class DocumentProcessor:
    """Handles document loading and text processing."""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        """
        Initialize the document processor.

        Args:
            chunk_size: Maximum number of words per chunk
            chunk_overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process raw text into document chunks.

        Args:
            text: Raw text content
            metadata: Optional metadata to attach to each document

        Returns:
            List of document dictionaries
        """
        # Split text into sentences
        sentences = nltk.sent_tokenize(text)

        # Create chunks with overlap
        chunks = []
        current_chunk = []
        current_chunk_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)

            # If adding this sentence would exceed chunk size, 
            # finalize the current chunk and start a new one
            if current_chunk_word_count + word_count > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                doc_id = f"doc_{len(chunks)}"

                doc = {
                    "id": doc_id,
                    "text": chunk_text,
                    "word_count": current_chunk_word_count
                }

                if metadata:
                    doc["metadata"] = metadata.copy()

                chunks.append(doc)

                # Start new chunk with overlap
                overlap_words = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
                current_chunk = overlap_words + [sentence]
                current_chunk_word_count = len(overlap_words) + word_count
            else:
                current_chunk.append(sentence)
                current_chunk_word_count += word_count

        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            doc_id = f"doc_{len(chunks)}"

            doc = {
                "id": doc_id,
                "text": chunk_text,
                "word_count": current_chunk_word_count
            }

            if metadata:
                doc["metadata"] = metadata.copy()

            chunks.append(doc)

        return chunks

    def load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process a document file.

        Args:
            file_path: Path to the file to process

        Returns:
            List of document dictionaries
        """
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        metadata = {
            "source": file_path,
            "file_name": file_name
        }

        # Process based on file type
        if file_ext == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        elif file_ext == ".html" or file_ext == ".htm":
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text(separator=' ')

        elif file_ext == ".pdf":
            # For PDF, you would need additional libraries
            # This is a placeholder - you would need to install PyPDF2 or pdfminer
            raise NotImplementedError("PDF processing requires additional libraries")

        elif file_ext in [".csv", ".xlsx", ".xls"]:
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Convert DataFrame to text
            text = df.to_string()

        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Clean the text
        text = self._clean_text(text)

        return self.load_text(text, metadata)

    def load_directory(self, directory_path: str, recursive: bool = True,
                      file_pattern: str = r'.*\.(txt|html|htm|csv|xlsx|xls)$') -> List[Dict[str, Any]]:
        """
        Load all matching files from a directory.

        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            file_pattern: Regex pattern to match file names

        Returns:
            List of document dictionaries
        """
        pattern = re.compile(file_pattern)
        documents = []

        for root, dirs, files in os.walk(directory_path):
            if not recursive and root != directory_path:
                continue

            for file in files:
                if pattern.match(file):
                    file_path = os.path.join(root, file)
                    try:
                        file_docs = self.load_file(file_path)
                        documents.extend(file_docs)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

        return documents

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        # Remove extra newlines
        text = re.sub(r'\n+', '\n', text)

        # Other cleaning as needed

        return text.strip()
