"""
API interface for the RAG-Lite system.
"""

from fastapi import FastAPI, HTTPException, Depends, Body, Query, Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json

from .rag_engine import RAGEngine
from .indexer import DocumentIndexer
from .document_processor import DocumentProcessor


app = FastAPI(
    title="RAG-Lite API",
    description="API for Retrieval-Augmented Generation",
    version="0.1.0",
)


# Models for request/response
class DocumentInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class QueryInput(BaseModel):
    query: str
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] = None


class RAGResponse(BaseModel):
    query: str
    answer: str
    context: List[Dict[str, Any]]


# Dependency for getting RAG engine instance
def get_rag_engine():
    """Dependency to get a RAG engine instance."""
    model_dir = os.environ.get("RAG_MODEL_DIR", "./models")

    try:
        engine = RAGEngine.load(model_dir)
        return engine
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # If no saved model exists, create a new one with default settings
        indexer = DocumentIndexer()
        engine = RAGEngine(indexer=indexer)
        return engine


@app.post("/documents/", response_model=Dict[str, Any])
async def add_documents(documents: List[DocumentInput], 
                       engine: RAGEngine = Depends(get_rag_engine)):
    """
    Add documents to the RAG engine.
    """
    docs_to_add = []

    for i, doc in enumerate(documents):
        doc_dict = {
            "id": f"api_doc_{i}",
            "text": doc.text
        }

        if doc.metadata:
            doc_dict["metadata"] = doc.metadata

        docs_to_add.append(doc_dict)

    engine.indexer.add_documents(docs_to_add)

    return {"status": "success", "added": len(docs_to_add)}


@app.post("/query/", response_model=RAGResponse)
async def query(query_input: QueryInput, 
              engine: RAGEngine = Depends(get_rag_engine)):
    """
    Query the RAG engine and get a response.
    """
    # Override temperature if provided
    temp = engine.temperature
    if query_input.temperature is not None:
        temp = query_input.temperature

    # Modify the engine temperature temporarily
    original_temp = engine.temperature
    engine.temperature = temp

    try:
        result = engine.generate(
            query=query_input.query,
            max_new_tokens=query_input.max_new_tokens
        )
        return result
    finally:
        # Restore original temperature
        engine.temperature = original_temp


@app.post("/save/")
async def save_model(directory: str = Body(..., embed=True),
                   engine: RAGEngine = Depends(get_rag_engine)):
    """
    Save the current RAG engine state to disk.
    """
    try:
        engine.save(directory)
        return {"status": "success", "saved_to": directory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")


@app.get("/health/")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
