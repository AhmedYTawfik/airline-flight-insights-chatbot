"""
Airline Flight Insights Assistant - Backend Package

This package provides the full Graph-RAG pipeline for airline flight analysis:
- Neo4j database connectivity
- Cypher query templates (17+ queries across 5 intents)
- Embedding-based semantic search (MiniLM and MPNet)
- Multiple LLM providers (Gemini, Groq, HuggingFace)
- Hybrid retrieval pipeline
- Streamlit web application
"""

from .config import (
    NEO4J_CONFIG,
    EMBEDDING_MODELS,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    HUGGINGFACE_API_TOKEN
)

from .database import get_db, Neo4jConnection

from .cypher_queries import (
    QUERY_TEMPLATES,
    run_query,
    format_query_result,
    get_query_descriptions
)

from .embeddings import (
    get_model,
    generate_embeddings,
    generate_single_embedding,
    semantic_search_journeys,
    semantic_search_flights,
    get_embedding_context,
    generate_and_store_all_embeddings
)

from .llm_providers import (
    get_llm_manager,
    generate_response,
    LLMResponse,
    MODEL_PROVIDERS
)

from .intent_classifier import (
    classify_intent_rule_based,
    classify_intent_llm,
    extract_entities_rule_based,
    get_relevant_query_indices
)

from .pipeline import (
    run_pipeline,
    get_hybrid_context,
    get_cypher_context,
    get_embedding_retrieval_context,
    compare_retrieval_methods,
    compare_llm_models,
    ask,
    PipelineResult,
    RetrievalResult
)

__version__ = "1.0.0"
__author__ = "Airline Insights Team"
