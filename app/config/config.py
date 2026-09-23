import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DB_FAISS_PATH = "vectorstore/db_faiss"

# =============================================================================
# MODEL COMBINATIONS
# =============================================================================
# Each config now exposes the full set of hyperparameters used in the paper
# (Section 5.1 - Experimental Setup & Reproducibility):
#   llm              : Groq model identifier
#   embeddings       : HuggingFace sentence-transformers model
#   vectorstore      : path to FAISS index directory
#   rerank           : whether to use the cross-encoder reranker
#   reranker_model   : HuggingFace cross-encoder identifier
#   top_n            : number of documents returned AFTER reranking
#   initial_k        : number of documents fetched BEFORE reranking
#   answer_style     : "concise" or "detailed" (controls prompt template)
#   temperature      : LLM sampling temperature
#   max_tokens       : maximum output tokens
# =============================================================================

MODEL_COMBINATIONS = {
    "Setup_Default": {
        # Model 1 in paper: High Speed & Efficiency
        "llm": "meta-llama/llama-4-scout-17b-16e-instruct",
        "embeddings": "nomic-ai/nomic-embed-text-v1.5",
        "vectorstore": "vectorstore/db_faiss_default",
        "rerank": False,
        "reranker_model": None,
        "top_n": 2,
        "initial_k": 2,
        "answer_style": "concise",
        "temperature": 0.3,
        "max_tokens": 500,
    },
    "Setup_Enhanced": {
        # Model 2 in paper: Advanced Reasoning & Agentic Tasks
        "llm": "openai/gpt-oss-20b",
        "embeddings": "sentence-transformers/all-distilroberta-v1",
        "vectorstore": "vectorstore/db_faiss_enhanced",
        "rerank": False,
        "reranker_model": None,
        "top_n": 2,
        "initial_k": 2,
        "answer_style": "concise",
        "temperature": 0.3,
        "max_tokens": 500,
    },
    "Setup_Medical": {
        # Model 3 in paper: High Precision & Multilingual Support (with Reranking)
        "llm": "qwen/qwen3-32b",
        "embeddings": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "vectorstore": "vectorstore/db_faiss_medical",
        "rerank": True,
        "reranker_model": "BAAI/bge-reranker-base",
        "top_n": 2,
        "initial_k": 10,
        "answer_style": "detailed",
        "temperature": 0.3,
        "max_tokens": 800,
    },
    "Setup_Model4": {
        # Model 4 in paper: LLM Doctor Main System
        "llm": "llama-3.1-8b-instant",
        "embeddings": "sentence-transformers/all-mpnet-base-v2",
        "vectorstore": "vectorstore/db_faiss_model4",
        "rerank": False,
        "reranker_model": None,
        "top_n": 2,
        "initial_k": 5,
        "answer_style": "concise",
        "temperature": 0.2,
        "max_tokens": 500,
    },
    # =========================================================================
    # BASELINE CONFIGURATIONS (added for Reviewer #2 / #3)
    # =========================================================================
    "Non_RAG_Baseline": {
        # Standalone Llama-3.1-8B-Instant with NO retrieval.
        # Measures the model's parametric medical knowledge only.
        "llm": "llama-3.1-8b-instant",
        "embeddings": None,
        "vectorstore": None,
        "rerank": False,
        "reranker_model": None,
        "top_n": 0,
        "initial_k": 0,
        "answer_style": "concise",
        "temperature": 0.2,
        "max_tokens": 500,
    },
    "Naive_RAG_Baseline": {
        # FAISS retrieval WITHOUT reranking, same LLM and embeddings as Model 4.
        # Isolates the contribution of the reranker.
        "llm": "llama-3.1-8b-instant",
        "embeddings": "sentence-transformers/all-mpnet-base-v2",
        "vectorstore": "vectorstore/db_faiss_model4",
        "rerank": False,
        "reranker_model": None,
        "top_n": 2,
        "initial_k": 2,
        "answer_style": "concise",
        "temperature": 0.2,
        "max_tokens": 500,
    },
}