"""
RAG Server with LoRA fine-tuned model support.
"""

import os
import sys
import json
import faiss
import numpy as np
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Import custom prompt generator
try:
    from .rag_prompts import get_rag_prompt
except ImportError:
    # For direct module execution
    from rag_prompts import get_rag_prompt


# Configuration
BASE_MODEL = "facebook/opt-350m"  # Smaller OPT model for faster loading and inference
# Options: "facebook/opt-1.3b", "Qwen/Qwen-1.8B", "facebook/opt-6.7b"
LORA_PATH = "artifacts/lora_out"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "artifacts/arxiv.faiss"
DOCUMENTS_PATH = "artifacts/abstracts.npy"
TOP_K = 4  # Increased number of documents to retrieve for better context


# Initialize app
app = FastAPI(
    title="RAG-Lite Server",
    description="Retrieval-Augmented Generation with LoRA fine-tuned model",
    version="0.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models for request/response
class QueryRequest(BaseModel):
    query: Optional[str] = None
    question: Optional[str] = None  # Alternative field for compatibility
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

    @property
    def user_query(self) -> str:
        """Return the query text, handling both field names."""
        return self.query or self.question or ""


class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]

    class Config:
        schema_extra = {
            "example": {
                "query": "What is quantum computing?",
                "answer": "Quantum computing is a type of computing that uses quantum bits...",
                "retrieved_documents": [
                    {"id": 1, "text": "Sample document text...", "similarity": 0.95}
                ]
            }
        }


# Global variables to hold models and data
embedding_model = None
tokenizer = None
generator = None
index = None
documents = None


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup."""
    global embedding_model, tokenizer, generator, index, documents

    print("Loading models and data...")

    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Check if we have a FAISS index and documents
    has_real_data = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH)

    if has_real_data:
        # Load real index and documents
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
        index = faiss.read_index(FAISS_INDEX_PATH)

        print(f"Loading documents from {DOCUMENTS_PATH}")
        documents = np.load(DOCUMENTS_PATH, allow_pickle=True)
    else:
        # Create mock index and documents for testing
        print("WARNING: Real data not found. Creating mock index and documents for testing.")

        # Create a small sample index
        d = 384  # Dimension for all-MiniLM-L6-v2
        index = faiss.IndexFlatIP(d)

        # Create a few mock documents about various topics
        mock_documents = [
            "Quantum computing is a type of computing that uses quantum phenomena such as superposition and entanglement. Unlike classical computers that use bits as 0 or 1, quantum computers use qubits that can exist in multiple states simultaneously.",
            "Quantum computing leverages quantum mechanics to process information. Quantum bits or qubits allow for much faster processing of certain types of problems compared to classical computing.",
            "Quantum supremacy refers to the point at which a quantum computer can solve problems that classical computers practically cannot. Google claimed to have achieved this milestone in 2019.",
            "Quantum error correction is a major challenge in building practical quantum computers. Quantum states are fragile and susceptible to noise and decoherence.",
            "Quantum algorithms like Shor's algorithm for factoring and Grover's search algorithm demonstrate potential quantum advantage over classical algorithms.",
            "NP-complete problems are a class of computational decision problems for which no efficient solution algorithm has been found. For these problems, the time required to solve them using any currently known algorithm increases very quickly as the size of the problem grows.",
            "The most famous NP-complete problem is the Traveling Salesman Problem (TSP), which asks for the shortest possible route that visits each city exactly once and returns to the origin city. Despite decades of research, no polynomial-time algorithm has been found for solving TSP optimally.",
            "Other classic examples of NP-complete problems include the Boolean satisfiability problem (SAT), the Knapsack problem, and the Graph Coloring problem, all of which have important applications in computer science and operations research.",
            "Machine learning is a field of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data. It enables computers to improve their performance on a task through experience without being explicitly programmed.",
            "Deep learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to analyze various factors of data. It is particularly powerful for tasks like image and speech recognition, natural language processing, and playing complex games."
        ]

        # Create embeddings for mock documents
        mock_embeddings = embedding_model.encode(mock_documents)
        faiss.normalize_L2(mock_embeddings)

        # Add to index
        index.add(mock_embeddings)

        # Store the documents
        documents = np.array(mock_documents)

    # We'll skip quantization for testing purposes
    use_quantization = False  # Set to True when proper bitsandbytes setup is available

    if use_quantization:
        # Configure quantization for LLM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # Load base model and tokenizer
    print(f"Loading base model: {BASE_MODEL}")

    # Check if we need trust_remote_code for this model
    needs_remote_code = "qwen" in BASE_MODEL.lower() or "baichuan" in BASE_MODEL.lower()

    # Load tokenizer with appropriate flags
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=needs_remote_code
    )

    # Special handling for Qwen models
    if "qwen" in BASE_MODEL.lower():
        tokenizer.padding_side = "left"
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

    # Load model (with or without quantization)
    try:
        print("Loading model...")
        if use_quantization:
            # With quantization
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=needs_remote_code
            )
        else:
            # Optimized for CPU operation
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=needs_remote_code,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None  # Explicitly use CPU only
            )
    except Exception as e:
        print(f"Error loading model: {e}")

        # Try with minimal configuration as fallback
        print("Trying with minimal configuration for faster loading...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=needs_remote_code,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            device_map=None,  # Don't use device map in fallback mode
            offload_folder="temp_offload"  # Use disk offloading if needed
        )

    # Load LoRA weights if available
    if os.path.exists(LORA_PATH) and os.path.isdir(LORA_PATH) and os.listdir(LORA_PATH):
        try:
            print(f"Loading LoRA weights from {LORA_PATH}")
            model = PeftModel.from_pretrained(model, LORA_PATH)
            print("LoRA weights loaded successfully!")
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            print("Falling back to base model")
    else:
        print(f"LoRA weights not found at {LORA_PATH} or directory is empty")
        print("Using base model only")

    # Create text generation pipeline with error handling
    try:
        generator = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            device=-1  # Use CPU for better compatibility
        )
        print("Text generation pipeline created successfully!")
    except Exception as e:
        print(f"Warning: Error creating standard pipeline: {e}")
        print("Falling back to basic generation...")
        # Define a simple function to mimic the pipeline API
        def basic_generator(prompt, **kwargs):
            max_tokens = kwargs.get("max_new_tokens", 50)
            try:
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                gen_tokens = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=kwargs.get("do_sample", False),
                    temperature=kwargs.get("temperature", 0.7),
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                return [{"generated_text": generated_text}]
            except Exception as inner_e:
                print(f"Generation error: {inner_e}")
                # Return fallback response
                return [{"generated_text": f"{prompt} [Error: Could not generate text]"}]

        generator = basic_generator

    print("Startup complete, models loaded!")


@app.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    """Process a RAG query."""
    try:
        # Check if models are loaded
        if embedding_model is None or generator is None:
            raise HTTPException(status_code=503, detail="Models are still loading, please try again later")

        # Get query embedding
        user_query = request.user_query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query or question field is required")

        query_embedding = embedding_model.encode([user_query])
        faiss.normalize_L2(query_embedding)

        # Search for similar documents
        distances, indices = index.search(query_embedding, TOP_K)

        # Get retrieved documents
        retrieved_docs = []
        context_text = ""
        relevant_docs_found = False
        min_relevance_threshold = 0.15  # Lower threshold as our embeddings have low similarity scores

        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(documents):
                doc_text = documents[idx]
                similarity = 1 - distances[0][i]
                
                # Check if document is relevant
                if similarity >= min_relevance_threshold:
                    relevant_docs_found = True
                
                # Add to context
                context_text += f"\n\n{doc_text}"

                # Add to retrieved documents list
                retrieved_docs.append({
                    "id": int(idx),
                    "text": doc_text[:300] + "..." if len(doc_text) > 300 else doc_text,
                    "similarity": float(similarity)
                })

        # Check if we found relevant documents and generate appropriate prompt
        if not relevant_docs_found:
            # No relevant documents found - provide a generic response
            answer = f"I don't have enough information to answer this question properly. While I've found some documents, they don't appear to be directly relevant to your question about '{user_query}'."
            
            return RAGResponse(
                query=user_query,
                answer=answer,
                retrieved_documents=retrieved_docs
            )
            
        # Create prompt based on the model
        if "llama" in BASE_MODEL.lower():
            prompt = f"[INST] <<SYS>>\nYou are a helpful AI assistant. Use the context provided to answer the question. Keep your answer clear and concise.\nContext: {context_text}\n<</SYS>>\n\nQuestion: {user_query} [/INST]"
        elif "qwen" in BASE_MODEL.lower():
            prompt = f"<|im_start|>system\nYou are a helpful AI assistant. Use the context provided to answer the question. Keep your answer clear and concise.\nContext: {context_text}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        elif "gpt2" in BASE_MODEL.lower() or "opt" in BASE_MODEL.lower() or "phi" in BASE_MODEL.lower():
            # Improved prompt for phi, gpt2 and opt models
            prompt = f"""You are a helpful AI assistant. Answer ONLY based on the context below. If the information isn't in the context, say "I don't have enough information to answer this question."

CONTEXT:
{context_text}

QUESTION: {user_query}

ANSWER:"""
        else:
            # Generic prompt
            prompt = f"""You are a helpful AI assistant. Use only the information provided in the context to answer the question.

CONTEXT:
{context_text}

QUESTION: {user_query}

ANSWER:"""

        # Generate response
        generation_kwargs = {
            "max_new_tokens": request.max_length or 256,  # Longer responses for better quality on GPU
            "temperature": request.temperature,
            "do_sample": request.temperature > 0,
            "top_p": 0.92,
            "top_k": 50,  # Better diversity with GPU
            "repetition_penalty": 1.15,  # Less aggressive to allow some repetition when needed
            "num_return_sequences": 1,
            "no_repeat_ngram_size": 3,  # Avoid repeating phrases
            "use_cache": True  # Enable KV caching for faster generation
        }

        try:
            # Generate response
            raw = generator(prompt, **generation_kwargs)[0]["generated_text"]

            # Extract answer based on model type
            if "llama" in BASE_MODEL.lower():
                answer = raw.split("[/INST]")[-1].strip()
            elif "qwen" in BASE_MODEL.lower():
                answer = raw.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
            elif "gpt2" in BASE_MODEL.lower() or "opt" in BASE_MODEL.lower() or "phi" in BASE_MODEL.lower():
                # For smaller models, get text after "Answer:"
                if "Answer:" in raw:
                    answer = raw.split("Answer:")[-1].strip()
                else:
                    # If no "Answer:" marker, get text after the last newline in prompt
                    prompt_end = prompt.rfind("\n")
                    if prompt_end != -1 and prompt_end < len(raw):
                        answer = raw[prompt_end+1:].strip()
                    else:
                        # Fallback - get everything after the prompt
                        answer = raw[len(prompt):].strip()
            else:
                # Generic extraction - get everything after the prompt
                answer = raw[len(prompt):].strip()

            # Clean up any weird artifacts
            answer = answer.replace("down is a deference", "").strip()
            
            # Detect low quality answers that repeat the question as a list of questions
            if answer.count('?') > 5 and ("1." in answer or "2." in answer):
                answer = "I apologize, but I'm unable to provide a coherent answer based on the context. The documents I have are about quantum computing, NP-complete problems, and machine learning."

            # If answer is empty or too short, provide a fallback
            if len(answer) < 10:
                answer = "I couldn't generate a proper response based on the context. Please try rephrasing your question."
                
            # Final formatting cleanup
            import re
            # Remove multiple consecutive newlines
            answer = re.sub(r'\n{2,}', '\n\n', answer)
            # Remove repeated sentences at the beginning
            if len(answer) > 50 and "." in answer:
                first_sentence = answer.split('.')[0] + '.'
                if answer.count(first_sentence) > 1:
                    answer = answer.replace(first_sentence, '', 1).strip()
        except Exception as e:
            print(f"Error in text generation: {e}")
            answer = f"Error generating response: {str(e)}"

        return RAGResponse(
            query=user_query,
            answer=answer,
            retrieved_documents=retrieved_docs
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
