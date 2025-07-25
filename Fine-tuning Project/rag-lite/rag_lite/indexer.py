"""
Document indexing and vector storage functionality.
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DocumentIndexer:
    """Handles document indexing and vector search operations."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """
        Initialize the document indexer.

        Args:
            model_name: Name of the sentence transformer model to use
            dimension: Dimension of the embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        self.documents = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the index.

        Args:
            documents: List of document dictionaries with at least 'text' and 'id' keys
        """
        texts = [doc['text'] for doc in documents]
        embeddings = self.model.encode(texts)

        # Add to FAISS index
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)

        # Store documents
        self.documents.extend(documents)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of document dictionaries with similarity scores
        """
        # Encode the query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search the index
        distances, indices = self.index.search(query_embedding, k)

        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                doc = self.documents[idx].copy()
                doc['score'] = float(1 - distances[0][i])  # Convert distance to similarity score
                results.append(doc)

        return results

    def save(self, directory: str) -> None:
        """
        Save the index and documents to disk.

        Args:
            directory: Directory to save the index in
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        import pickle
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, directory: str, model_name: str = "all-MiniLM-L6-v2") -> 'DocumentIndexer':
        """
        Load an index from disk.

        Args:
            directory: Directory containing the index
            model_name: Name of the sentence transformer model

        Returns:
            Loaded DocumentIndexer instance
        """
        indexer = cls(model_name=model_name)

        # Load the FAISS index
        indexer.index = faiss.read_index(os.path.join(directory, "index.faiss"))

        # Load the documents
        import pickle
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            indexer.documents = pickle.load(f)

        return indexer
