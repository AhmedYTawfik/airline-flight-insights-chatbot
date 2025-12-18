"""
Hybrid Retrieval Module for Airline Flight Insights Graph-RAG

This module combines Cypher-based retrieval with embedding-based semantic search
to provide comprehensive context for the LLM.

Usage in notebooks:
    from hybrid_retrieval import get_hybrid_context, answer_with_hybrid_context
"""

from typing import Dict, List, Any, Callable
from neo4j import Driver


def get_hybrid_context(
    driver: Driver,
    prompt: str,
    get_cypher_context: Callable,
    format_query_result: Callable,
    model_key: str = "minilm",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Get context from both Cypher queries and embedding-based search.
    
    Args:
        driver: Neo4j driver instance
        prompt: User's natural language question
        get_cypher_context: Function to get relevant Cypher queries (from main.ipynb)
        format_query_result: Function to format query results (from main.ipynb)
        model_key: Embedding model to use ('minilm' or 'mpnet')
        top_k: Number of embedding results per node type
        
    Returns:
        Dict with 'cypher_context', 'embedding_context', and 'combined_context'
    """
    # Import embedding functions
    from embeddings import get_embedding_context
    
    results = {
        'cypher_context': [],
        'embedding_context': '',
        'combined_context': ''
    }
    
    # Get Cypher-based context
    try:
        context_queries = get_cypher_context(prompt)
        for cq in context_queries:
            context = format_query_result(cq['query_index'], **cq['params'])
            results['cypher_context'].append(context)
    except Exception as e:
        results['cypher_context'].append(f"Cypher error: {str(e)}")
    
    # Get embedding-based context
    try:
        results['embedding_context'] = get_embedding_context(
            driver, prompt, model_key, top_k
        )
    except Exception as e:
        results['embedding_context'] = f"Embedding error: {str(e)}"
    
    # Combine contexts
    cypher_text = '\n\n'.join(results['cypher_context'])
    results['combined_context'] = f"""=== STRUCTURED QUERY RESULTS ===
{cypher_text}

=== SEMANTIC SEARCH RESULTS ===
{results['embedding_context']}
"""
    
    return results


def answer_with_hybrid_context(
    driver: Driver,
    question: str,
    llm: Any,
    get_cypher_context: Callable,
    format_query_result: Callable,
    model_key: str = "minilm"
) -> str:
    """
    Answer a question using hybrid retrieval context with the LLM.
    
    Args:
        driver: Neo4j driver instance
        question: User's question
        llm: LangChain LLM instance
        get_cypher_context: Function to get relevant Cypher queries
        format_query_result: Function to format query results
        model_key: Embedding model to use
        
    Returns:
        LLM-generated answer
    """
    # Get hybrid context
    hybrid_result = get_hybrid_context(
        driver, question, get_cypher_context, format_query_result, model_key, top_k=5
    )
    context = hybrid_result['combined_context']
    
    # Build prompt for LLM
    prompt = f"""You are an AI assistant for an airline company, helping analyze flight data.

Based on the following context from our knowledge graph, answer the user's question.
Only use information from the provided context. If the context doesn't contain
relevant information, say so.

CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""
    
    # Get LLM response
    response = llm.invoke(prompt)
    return response.content


def compare_retrieval_methods(
    driver: Driver,
    question: str,
    get_cypher_context: Callable,
    format_query_result: Callable
) -> Dict[str, Any]:
    """
    Compare results from different retrieval methods.
    
    Args:
        driver: Neo4j driver instance
        question: User's question
        get_cypher_context: Function to get relevant Cypher queries
        format_query_result: Function to format query results
        
    Returns:
        Dict with results from each method
    """
    from embeddings import get_embedding_context
    
    results = {
        'cypher_only': [],
        'embedding_minilm': '',
        'embedding_mpnet': '',
        'hybrid_minilm': None,
        'hybrid_mpnet': None
    }
    
    # Cypher only
    try:
        context_queries = get_cypher_context(question)
        for cq in context_queries:
            context = format_query_result(cq['query_index'], **cq['params'])
            results['cypher_only'].append(context)
    except Exception as e:
        results['cypher_only'] = [f"Error: {str(e)}"]
    
    # Embedding only (MiniLM)
    try:
        results['embedding_minilm'] = get_embedding_context(driver, question, "minilm", 5)
    except Exception as e:
        results['embedding_minilm'] = f"Error: {str(e)}"
    
    # Embedding only (MPNet)
    try:
        results['embedding_mpnet'] = get_embedding_context(driver, question, "mpnet", 5)
    except Exception as e:
        results['embedding_mpnet'] = f"Error: {str(e)}"
    
    # Hybrid (both models)
    results['hybrid_minilm'] = get_hybrid_context(
        driver, question, get_cypher_context, format_query_result, "minilm", 5
    )
    results['hybrid_mpnet'] = get_hybrid_context(
        driver, question, get_cypher_context, format_query_result, "mpnet", 5
    )
    
    return results


def print_comparison(results: Dict[str, Any]):
    """
    Pretty print comparison results.
    
    Args:
        results: Output from compare_retrieval_methods
    """
    print("=" * 80)
    print("RETRIEVAL METHOD COMPARISON")
    print("=" * 80)
    
    print("\n--- CYPHER ONLY (Baseline) ---")
    for ctx in results['cypher_only']:
        print(ctx)
        print()
    
    print("\n--- EMBEDDING ONLY (MiniLM) ---")
    print(results['embedding_minilm'])
    
    print("\n--- EMBEDDING ONLY (MPNet) ---")
    print(results['embedding_mpnet'])
    
    print("\n--- HYBRID (MiniLM) ---")
    if results['hybrid_minilm']:
        print(results['hybrid_minilm']['combined_context'])
    
    print("\n--- HYBRID (MPNet) ---")
    if results['hybrid_mpnet']:
        print(results['hybrid_mpnet']['combined_context'])
