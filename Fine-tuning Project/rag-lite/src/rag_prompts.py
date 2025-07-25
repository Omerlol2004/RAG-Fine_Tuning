"""
Improved RAG prompts for different model types
"""

def get_rag_prompt(model_name, context_text, user_query):
    """
    Generate an appropriate RAG prompt based on the model type.

    Args:
        model_name: The name of the language model
        context_text: Retrieved context information
        user_query: The user's question

    Returns:
        str: A formatted prompt appropriate for the model
    """
    model_name = model_name.lower()

    # Llama-style models
    if "llama" in model_name:
        return f"[INST] <<SYS>>\nYou are a helpful AI assistant. Use only the context provided to answer the question. If you don't have relevant information in the context, say 'I don't have enough information to answer this question properly.'\nContext: {context_text}\n<</SYS>>\n\nQuestion: {user_query} [/INST]"

    # Qwen-style models
    elif "qwen" in model_name:
        return f"<|im_start|>system\nYou are a helpful AI assistant. Use only the context provided to answer the question. If you don't have relevant information in the context, say 'I don't have enough information to answer this question properly.'\nContext: {context_text}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"

    # GPT-2, OPT and Phi models
    elif "gpt2" in model_name or "opt" in model_name or "phi" in model_name:
        return f"""You are a helpful AI assistant. Answer ONLY based on the context below. If the information isn't in the context, say "I don't have enough information to answer this question."

CONTEXT:
{context_text}

QUESTION: {user_query}

ANSWER:"""

    # Generic prompt for other models
    else:
        return f"""You are a helpful AI assistant. Answer ONLY based on the context below. If the information isn't in the context, say "I don't have enough information to answer this question."

CONTEXT:
{context_text}

QUESTION: {user_query}

ANSWER:"""
