"""
Hybrid retrieval pipeline combining Cypher queries with embedding-based semantic search.
This is the core Graph-RAG pipeline for the Airline Flight Insights Assistant.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from database import get_db
from cypher_queries import (
    QUERY_TEMPLATES, format_query_result, format_results_as_text, run_query
)
from embeddings import (
    get_embedding_context, semantic_search_journeys, semantic_search_flights,
    format_embedding_results
)
from intent_classifier import get_relevant_query_indices, classify_intent_rule_based
from llm_providers import generate_response, LLMResponse


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline."""
    method: str  # "cypher", "embedding", or "hybrid"
    
    # Cypher results
    cypher_contexts: List[str] = field(default_factory=list)
    cypher_queries: List[str] = field(default_factory=list)
    cypher_raw_results: List[List[Dict]] = field(default_factory=list)
    
    # Embedding results
    embedding_context: str = ""
    journey_results: List[Dict] = field(default_factory=list)
    flight_results: List[Dict] = field(default_factory=list)
    
    # Combined
    combined_context: str = ""
    
    # Metadata
    retrieval_time: float = 0.0
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result including LLM response."""
    query: str
    retrieval: RetrievalResult
    llm_response: LLMResponse
    total_time: float
    model_used: str
    embedding_model: Optional[str] = None


# System prompt for the LLM
SYSTEM_PROMPT = """You are an AI assistant for an airline company, helping analyze flight data and customer insights.

Your role is to:
1. Provide data-driven insights based on the context from our knowledge graph
2. Answer from the airline's operational perspective (not passenger perspective)
3. Give actionable recommendations when relevant
4. Be specific and cite data from the context
5. If information is insufficient, clearly state what you don't know

Focus areas:
- Flight delays and on-time performance
- Route analysis and optimization
- Passenger satisfaction trends
- Fleet performance monitoring
- Loyalty program insights
- Demographic travel patterns"""


def get_cypher_context(query: str, use_llm: bool = True,
                       model_name: str = "Gemini-2.0-Flash") -> RetrievalResult:
    """Get context from Cypher queries (Baseline approach)."""
    result = RetrievalResult(method="cypher")
    start_time = time.time()
    
    try:
        # Get relevant queries using intent classification
        relevant_queries = get_relevant_query_indices(query, use_llm, model_name)
        
        for rq in relevant_queries:
            query_index = rq["query_index"]
            params = rq["params"]
            
            try:
                description, cypher, results = format_query_result(query_index, **params)
                result.cypher_contexts.append(format_results_as_text(description, results))
                result.cypher_queries.append(cypher)
                result.cypher_raw_results.append(results)
            except Exception as e:
                result.cypher_contexts.append(f"Error executing query {query_index}: {e}")
        
        result.combined_context = "\n\n".join(result.cypher_contexts)
        
    except Exception as e:
        result.error = str(e)
    
    result.retrieval_time = time.time() - start_time
    return result


def get_embedding_retrieval_context(query: str,
                                    model_name: str = "all-MiniLM-L6-v2",
                                    top_k: int = 5) -> RetrievalResult:
    """Get context from embedding-based semantic search (Experiment 2)."""
    result = RetrievalResult(method="embedding")
    start_time = time.time()
    
    try:
        embedding_result = get_embedding_context(query, model_name, top_k)
        
        result.embedding_context = embedding_result['combined_context']
        result.journey_results = embedding_result['journey_results']
        result.flight_results = embedding_result['flight_results']
        result.combined_context = result.embedding_context
        
    except Exception as e:
        result.error = str(e)
        result.embedding_context = f"Embedding search error: {e}"
        result.combined_context = result.embedding_context
    
    result.retrieval_time = time.time() - start_time
    return result


def get_hybrid_context(query: str,
                       use_llm: bool = True,
                       llm_model: str = "Gemini-2.0-Flash",
                       embedding_model: str = "all-MiniLM-L6-v2",
                       top_k: int = 5) -> RetrievalResult:
    """Get context from both Cypher and embedding search (Hybrid approach)."""
    result = RetrievalResult(method="hybrid")
    start_time = time.time()
    
    # Get Cypher context
    cypher_result = get_cypher_context(query, use_llm, llm_model)
    result.cypher_contexts = cypher_result.cypher_contexts
    result.cypher_queries = cypher_result.cypher_queries
    result.cypher_raw_results = cypher_result.cypher_raw_results
    
    # Get embedding context
    embedding_result = get_embedding_retrieval_context(query, embedding_model, top_k)
    result.embedding_context = embedding_result.embedding_context
    result.journey_results = embedding_result.journey_results
    result.flight_results = embedding_result.flight_results
    
    # Combine contexts
    cypher_text = "\n\n".join(result.cypher_contexts) if result.cypher_contexts else "No structured query results."
    embedding_text = result.embedding_context if result.embedding_context else "No semantic search results."
    
    result.combined_context = f"""=== STRUCTURED QUERY RESULTS (Cypher) ===
{cypher_text}

=== SEMANTIC SEARCH RESULTS (Embeddings) ===
{embedding_text}"""
    
    if cypher_result.error:
        result.error = f"Cypher error: {cypher_result.error}"
    if embedding_result.error:
        err = f"Embedding error: {embedding_result.error}"
        result.error = f"{result.error}; {err}" if result.error else err
    
    result.retrieval_time = time.time() - start_time
    return result


def build_llm_prompt(query: str, context: str) -> str:
    """Build the full prompt for the LLM."""
    return f"""{SYSTEM_PROMPT}

Based on the following context from our airline knowledge graph, please answer the user's question.
Only use information from the provided context. If the context doesn't contain enough information, clearly state what additional data would be needed.

CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""


def run_pipeline(query: str,
                 retrieval_method: str = "hybrid",
                 llm_model: str = "Gemini-2.0-Flash",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 top_k: int = 5) -> PipelineResult:
    """
    Run the complete Graph-RAG pipeline.
    
    Args:
        query: User's natural language question
        retrieval_method: "cypher", "embedding", or "hybrid"
        llm_model: Name of the LLM to use
        embedding_model: Name of the embedding model
        top_k: Number of results for semantic search
    
    Returns:
        PipelineResult with all data and the LLM response
    """
    start_time = time.time()
    
    # Step 1: Retrieve context
    if retrieval_method == "cypher":
        retrieval = get_cypher_context(query, use_llm=True, model_name=llm_model)
    elif retrieval_method == "embedding":
        retrieval = get_embedding_retrieval_context(query, embedding_model, top_k)
    else:  # hybrid
        retrieval = get_hybrid_context(query, True, llm_model, embedding_model, top_k)
    
    # Step 2: Generate LLM response
    prompt = build_llm_prompt(query, retrieval.combined_context)
    llm_response = generate_response(llm_model, prompt)
    
    total_time = time.time() - start_time
    
    return PipelineResult(
        query=query,
        retrieval=retrieval,
        llm_response=llm_response,
        total_time=total_time,
        model_used=llm_model,
        embedding_model=embedding_model if retrieval_method != "cypher" else None
    )


def compare_retrieval_methods(query: str,
                              llm_model: str = "Gemini-2.0-Flash",
                              embedding_model: str = "all-MiniLM-L6-v2") -> Dict[str, RetrievalResult]:
    """Compare results from different retrieval methods."""
    return {
        "cypher_only": get_cypher_context(query, True, llm_model),
        f"embedding_{embedding_model}": get_embedding_retrieval_context(query, embedding_model),
        "hybrid": get_hybrid_context(query, True, llm_model, embedding_model)
    }


def compare_llm_models(query: str,
                       models: List[str],
                       retrieval_method: str = "hybrid",
                       embedding_model: str = "all-MiniLM-L6-v2") -> Dict[str, PipelineResult]:
    """Compare results from different LLM models."""
    results = {}
    
    # Get retrieval context once (to make comparison fair)
    if retrieval_method == "hybrid":
        retrieval = get_hybrid_context(query, True, models[0], embedding_model)
    elif retrieval_method == "embedding":
        retrieval = get_embedding_retrieval_context(query, embedding_model)
    else:
        retrieval = get_cypher_context(query, True, models[0])
    
    prompt = build_llm_prompt(query, retrieval.combined_context)
    
    for model in models:
        start_time = time.time()
        llm_response = generate_response(model, prompt)
        total_time = time.time() - start_time
        
        results[model] = PipelineResult(
            query=query,
            retrieval=retrieval,
            llm_response=llm_response,
            total_time=total_time,
            model_used=model,
            embedding_model=embedding_model if retrieval_method != "cypher" else None
        )
    
    return results


# Convenience function for simple Q&A
def ask(query: str,
        method: str = "hybrid",
        model: str = "Gemini-2.0-Flash") -> str:
    """Simple function to ask a question and get an answer."""
    result = run_pipeline(query, method, model)
    
    if result.llm_response.success:
        return result.llm_response.content
    else:
        return f"Error: {result.llm_response.error}"
