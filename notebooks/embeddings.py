"""
Embeddings Module for Airline Flight Insights Graph-RAG

This module provides functionality for generating and using vector embeddings
with the Neo4j knowledge graph. Implements Experiment 2 (Embeddings) from Milestone 3.

Two embedding models are supported:
- all-MiniLM-L6-v2 (384 dimensions, fast)
- all-mpnet-base-v2 (768 dimensions, higher quality)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver
import json


# ============================================================================
# EMBEDDING MODELS
# ============================================================================

# Model configurations
EMBEDDING_MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "property_name": "embedding_minilm"
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "property_name": "embedding_mpnet"
    }
}

# Cache for loaded models
_model_cache: Dict[str, SentenceTransformer] = {}


def get_model(model_key: str) -> SentenceTransformer:
    """
    Load and cache an embedding model.
    
    Args:
        model_key: Either 'minilm' or 'mpnet'
        
    Returns:
        The loaded SentenceTransformer model
    """
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Use 'minilm' or 'mpnet'")
    
    if model_key not in _model_cache:
        model_name = EMBEDDING_MODELS[model_key]["name"]
        print(f"Loading model: {model_name}...")
        _model_cache[model_key] = SentenceTransformer(model_name)
        print(f"Model {model_name} loaded successfully!")
    
    return _model_cache[model_key]


# ============================================================================
# TEXT REPRESENTATION FUNCTIONS
# ============================================================================

def create_journey_text(properties: Dict[str, Any]) -> str:
    """
    Create a text representation of a Journey node for embedding.
    
    Args:
        properties: Dictionary of Journey node properties
        
    Returns:
        Human-readable text description of the journey
    """
    passenger_class = properties.get("passenger_class", "Unknown")
    food_score = properties.get("food_satisfaction_score", "N/A")
    delay = properties.get("arrival_delay_minutes", 0)
    miles = properties.get("actual_flown_miles", 0)
    legs = properties.get("number_of_legs", 1)
    
    # Create delay description
    if delay < 0:
        delay_text = f"arrived {abs(delay)} minutes early"
    elif delay == 0:
        delay_text = "arrived on time"
    else:
        delay_text = f"delayed by {delay} minutes"
    
    # Create satisfaction description
    if food_score == 1:
        satisfaction_text = "very poor food satisfaction"
    elif food_score == 2:
        satisfaction_text = "poor food satisfaction"
    elif food_score == 3:
        satisfaction_text = "average food satisfaction"
    elif food_score == 4:
        satisfaction_text = "good food satisfaction"
    elif food_score == 5:
        satisfaction_text = "excellent food satisfaction"
    else:
        satisfaction_text = "unknown food satisfaction"
    
    text = (
        f"A {passenger_class} class journey covering {miles:.0f} miles "
        f"over {legs} flight segment{'s' if legs > 1 else ''}. "
        f"The flight {delay_text} and had {satisfaction_text} "
        f"(score: {food_score}/5)."
    )
    
    return text


def create_flight_text(properties: Dict[str, Any], 
                       origin: Optional[str] = None, 
                       destination: Optional[str] = None) -> str:
    """
    Create a text representation of a Flight node for embedding.
    
    Args:
        properties: Dictionary of Flight node properties
        origin: Origin airport code (optional)
        destination: Destination airport code (optional)
        
    Returns:
        Human-readable text description of the flight
    """
    flight_number = properties.get("flight_number", "Unknown")
    fleet_type = properties.get("fleet_type_description", "Unknown aircraft")
    
    route_text = ""
    if origin and destination:
        route_text = f" operating the route from {origin} to {destination}"
    elif origin:
        route_text = f" departing from {origin}"
    elif destination:
        route_text = f" arriving at {destination}"
    
    text = f"Flight {flight_number} operated by {fleet_type} aircraft{route_text}."
    
    return text


def create_passenger_text(properties: Dict[str, Any]) -> str:
    """
    Create a text representation of a Passenger node for embedding.
    
    Args:
        properties: Dictionary of Passenger node properties
        
    Returns:
        Human-readable text description of the passenger
    """
    loyalty = properties.get("loyalty_program_level", "unknown")
    generation = properties.get("generation", "unknown")
    
    text = f"A {generation} passenger with {loyalty} loyalty program status."
    
    return text


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embeddings(texts: List[str], model_key: str = "minilm") -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_key: Either 'minilm' or 'mpnet'
        
    Returns:
        NumPy array of embeddings with shape (len(texts), embedding_dim)
    """
    model = get_model(model_key)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def generate_single_embedding(text: str, model_key: str = "minilm") -> List[float]:
    """
    Generate embedding for a single text (useful for query embedding).
    
    Args:
        text: Text string to embed
        model_key: Either 'minilm' or 'mpnet'
        
    Returns:
        List of floats representing the embedding
    """
    model = get_model(model_key)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


# ============================================================================
# NEO4J INTEGRATION
# ============================================================================

def fetch_journey_nodes(driver: Driver) -> List[Dict[str, Any]]:
    """
    Fetch all Journey nodes from Neo4j with their properties.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        List of dicts with 'feedback_ID' and 'properties' keys
    """
    query = """
    MATCH (j:Journey)
    RETURN j.feedback_ID AS feedback_ID,
           j.passenger_class AS passenger_class,
           j.food_satisfaction_score AS food_satisfaction_score,
           j.arrival_delay_minutes AS arrival_delay_minutes,
           j.actual_flown_miles AS actual_flown_miles,
           j.number_of_legs AS number_of_legs
    """
    
    with driver.session() as session:
        result = session.run(query)
        nodes = []
        for record in result:
            nodes.append({
                "feedback_ID": record["feedback_ID"],
                "properties": {
                    "passenger_class": record["passenger_class"],
                    "food_satisfaction_score": record["food_satisfaction_score"],
                    "arrival_delay_minutes": record["arrival_delay_minutes"],
                    "actual_flown_miles": record["actual_flown_miles"],
                    "number_of_legs": record["number_of_legs"]
                }
            })
        return nodes


def fetch_flight_nodes(driver: Driver) -> List[Dict[str, Any]]:
    """
    Fetch all Flight nodes from Neo4j with their properties and route info.
    
    Args:
        driver: Neo4j driver instance
        
    Returns:
        List of dicts with flight info including origin/destination
    """
    query = """
    MATCH (f:Flight)
    OPTIONAL MATCH (f)-[:DEPARTS_FROM]->(origin:Airport)
    OPTIONAL MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
    RETURN f.flight_number AS flight_number,
           f.fleet_type_description AS fleet_type_description,
           origin.station_code AS origin,
           dest.station_code AS destination
    """
    
    with driver.session() as session:
        result = session.run(query)
        nodes = []
        for record in result:
            nodes.append({
                "flight_number": record["flight_number"],
                "fleet_type_description": record["fleet_type_description"],
                "properties": {
                    "flight_number": record["flight_number"],
                    "fleet_type_description": record["fleet_type_description"]
                },
                "origin": record["origin"],
                "destination": record["destination"]
            })
        return nodes


def create_vector_index(driver: Driver, model_key: str, node_label: str = "Journey"):
    """
    Create a vector index in Neo4j for similarity search.
    
    Args:
        driver: Neo4j driver instance
        model_key: Either 'minilm' or 'mpnet'
        node_label: The node label to index ('Journey' or 'Flight')
    """
    config = EMBEDDING_MODELS[model_key]
    property_name = config["property_name"]
    dimensions = config["dimensions"]
    index_name = f"{node_label.lower()}_{property_name}"
    
    # Drop existing index if it exists (Neo4j syntax)
    drop_query = f"DROP INDEX {index_name} IF EXISTS"
    
    # Create vector index
    create_query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{node_label})
    ON n.{property_name}
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {dimensions},
        `vector.similarity_function`: 'cosine'
    }}}}
    """
    
    with driver.session() as session:
        try:
            session.run(drop_query)
        except Exception:
            pass  # Index might not exist
        session.run(create_query)
        print(f"Created vector index: {index_name} ({dimensions} dimensions)")


def store_journey_embeddings(driver: Driver, 
                             feedback_ids: List[str], 
                             embeddings: np.ndarray, 
                             model_key: str = "minilm",
                             batch_size: int = 100):
    """
    Store embeddings for Journey nodes in Neo4j.
    
    Args:
        driver: Neo4j driver instance
        feedback_ids: List of feedback_ID values
        embeddings: NumPy array of embeddings
        model_key: Either 'minilm' or 'mpnet'
        batch_size: Number of nodes to update per transaction
    """
    property_name = EMBEDDING_MODELS[model_key]["property_name"]
    
    query = f"""
    UNWIND $batch AS item
    MATCH (j:Journey {{feedback_ID: item.feedback_ID}})
    SET j.{property_name} = item.embedding
    """
    
    total = len(feedback_ids)
    stored = 0
    
    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch = []
            for j in range(i, min(i + batch_size, total)):
                batch.append({
                    "feedback_ID": feedback_ids[j],
                    "embedding": embeddings[j].tolist()
                })
            session.run(query, batch=batch)
            stored += len(batch)
            print(f"Stored {stored}/{total} Journey embeddings...")
    
    print(f"Successfully stored {total} Journey embeddings with {model_key} model")


def store_flight_embeddings(driver: Driver,
                            flights: List[Dict],
                            embeddings: np.ndarray,
                            model_key: str = "minilm",
                            batch_size: int = 100):
    """
    Store embeddings for Flight nodes in Neo4j.
    
    Args:
        driver: Neo4j driver instance
        flights: List of flight dicts with flight_number and fleet_type_description
        embeddings: NumPy array of embeddings
        model_key: Either 'minilm' or 'mpnet'
        batch_size: Number of nodes to update per transaction
    """
    property_name = EMBEDDING_MODELS[model_key]["property_name"]
    
    query = f"""
    UNWIND $batch AS item
    MATCH (f:Flight {{flight_number: item.flight_number, fleet_type_description: item.fleet_type_description}})
    SET f.{property_name} = item.embedding
    """
    
    total = len(flights)
    stored = 0
    
    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch = []
            for j in range(i, min(i + batch_size, total)):
                batch.append({
                    "flight_number": flights[j]["flight_number"],
                    "fleet_type_description": flights[j]["fleet_type_description"],
                    "embedding": embeddings[j].tolist()
                })
            session.run(query, batch=batch)
            stored += len(batch)
            print(f"Stored {stored}/{total} Flight embeddings...")
    
    print(f"Successfully stored {total} Flight embeddings with {model_key} model")


# ============================================================================
# SEMANTIC SEARCH
# ============================================================================

def semantic_search_journeys(driver: Driver, 
                              query_text: str, 
                              model_key: str = "minilm",
                              top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic similarity search on Journey nodes.
    
    Args:
        driver: Neo4j driver instance
        query_text: Natural language query
        model_key: Either 'minilm' or 'mpnet'
        top_k: Number of results to return
        
    Returns:
        List of matching Journey nodes with similarity scores
    """
    # Generate query embedding
    query_embedding = generate_single_embedding(query_text, model_key)
    
    config = EMBEDDING_MODELS[model_key]
    property_name = config["property_name"]
    index_name = f"journey_{property_name}"
    
    # Vector similarity search query
    search_query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
    YIELD node, score
    RETURN node.feedback_ID AS feedback_ID,
           node.passenger_class AS passenger_class,
           node.food_satisfaction_score AS food_satisfaction_score,
           node.arrival_delay_minutes AS arrival_delay_minutes,
           node.actual_flown_miles AS actual_flown_miles,
           node.number_of_legs AS number_of_legs,
           score
    ORDER BY score DESC
    """
    
    with driver.session() as session:
        result = session.run(search_query, top_k=top_k, query_embedding=query_embedding)
        matches = []
        for record in result:
            matches.append({
                "feedback_ID": record["feedback_ID"],
                "passenger_class": record["passenger_class"],
                "food_satisfaction_score": record["food_satisfaction_score"],
                "arrival_delay_minutes": record["arrival_delay_minutes"],
                "actual_flown_miles": record["actual_flown_miles"],
                "number_of_legs": record["number_of_legs"],
                "similarity_score": record["score"]
            })
        return matches


def semantic_search_flights(driver: Driver,
                            query_text: str,
                            model_key: str = "minilm",
                            top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic similarity search on Flight nodes.
    
    Args:
        driver: Neo4j driver instance
        query_text: Natural language query
        model_key: Either 'minilm' or 'mpnet'
        top_k: Number of results to return
        
    Returns:
        List of matching Flight nodes with similarity scores
    """
    # Generate query embedding
    query_embedding = generate_single_embedding(query_text, model_key)
    
    config = EMBEDDING_MODELS[model_key]
    property_name = config["property_name"]
    index_name = f"flight_{property_name}"
    
    # Vector similarity search query
    search_query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
    YIELD node, score
    MATCH (node)-[:DEPARTS_FROM]->(origin:Airport)
    MATCH (node)-[:ARRIVES_AT]->(dest:Airport)
    RETURN node.flight_number AS flight_number,
           node.fleet_type_description AS fleet_type_description,
           origin.station_code AS origin,
           dest.station_code AS destination,
           score
    ORDER BY score DESC
    """
    
    with driver.session() as session:
        result = session.run(search_query, top_k=top_k, query_embedding=query_embedding)
        matches = []
        for record in result:
            matches.append({
                "flight_number": record["flight_number"],
                "fleet_type_description": record["fleet_type_description"],
                "origin": record["origin"],
                "destination": record["destination"],
                "similarity_score": record["score"]
            })
        return matches


# ============================================================================
# HYBRID RETRIEVAL
# ============================================================================

def format_embedding_results(results: List[Dict[str, Any]], node_type: str = "Journey") -> str:
    """
    Format embedding search results as context text.
    
    Args:
        results: List of search results with similarity scores
        node_type: Either 'Journey' or 'Flight'
        
    Returns:
        Formatted context string
    """
    if not results:
        return f"No similar {node_type} nodes found."
    
    formatted = [f"Found {len(results)} semantically similar {node_type} records:"]
    
    for i, r in enumerate(results, 1):
        if node_type == "Journey":
            text = create_journey_text({
                "passenger_class": r.get("passenger_class"),
                "food_satisfaction_score": r.get("food_satisfaction_score"),
                "arrival_delay_minutes": r.get("arrival_delay_minutes"),
                "actual_flown_miles": r.get("actual_flown_miles"),
                "number_of_legs": r.get("number_of_legs")
            })
            formatted.append(f"  {i}. (similarity: {r['similarity_score']:.3f}) {text}")
        else:
            text = create_flight_text(
                {"flight_number": r.get("flight_number"),
                 "fleet_type_description": r.get("fleet_type_description")},
                origin=r.get("origin"),
                destination=r.get("destination")
            )
            formatted.append(f"  {i}. (similarity: {r['similarity_score']:.3f}) {text}")
    
    return "\n".join(formatted)


def get_embedding_context(driver: Driver, 
                          query: str, 
                          model_key: str = "minilm",
                          top_k: int = 5) -> str:
    """
    Get context from embedding-based semantic search.
    
    Args:
        driver: Neo4j driver instance
        query: User's natural language query
        model_key: Either 'minilm' or 'mpnet'
        top_k: Number of results per node type
        
    Returns:
        Combined context from Journey and Flight searches
    """
    contexts = []
    
    # Search Journeys
    try:
        journey_results = semantic_search_journeys(driver, query, model_key, top_k)
        contexts.append(format_embedding_results(journey_results, "Journey"))
    except Exception as e:
        contexts.append(f"Journey search error: {str(e)}")
    
    # Search Flights
    try:
        flight_results = semantic_search_flights(driver, query, model_key, top_k)
        contexts.append(format_embedding_results(flight_results, "Flight"))
    except Exception as e:
        contexts.append(f"Flight search error: {str(e)}")
    
    return "\n\n".join(contexts)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_and_store_all_embeddings(driver: Driver, model_key: str = "minilm"):
    """
    Generate and store embeddings for all Journey and Flight nodes.
    
    This is the main function to run when setting up embeddings.
    
    Args:
        driver: Neo4j driver instance
        model_key: Either 'minilm' or 'mpnet'
    """
    print(f"\n{'='*60}")
    print(f"Generating embeddings with {EMBEDDING_MODELS[model_key]['name']}")
    print(f"{'='*60}\n")
    
    # Process Journey nodes
    print("Fetching Journey nodes...")
    journeys = fetch_journey_nodes(driver)
    print(f"Found {len(journeys)} Journey nodes")
    
    if journeys:
        print("Creating text representations...")
        journey_texts = [create_journey_text(j["properties"]) for j in journeys]
        
        print("Generating Journey embeddings...")
        journey_embeddings = generate_embeddings(journey_texts, model_key)
        
        print("Creating vector index for Journeys...")
        create_vector_index(driver, model_key, "Journey")
        
        print("Storing Journey embeddings...")
        feedback_ids = [j["feedback_ID"] for j in journeys]
        store_journey_embeddings(driver, feedback_ids, journey_embeddings, model_key)
    
    # Process Flight nodes
    print("\nFetching Flight nodes...")
    flights = fetch_flight_nodes(driver)
    print(f"Found {len(flights)} Flight nodes")
    
    if flights:
        print("Creating text representations...")
        flight_texts = [
            create_flight_text(f["properties"], f["origin"], f["destination"])
            for f in flights
        ]
        
        print("Generating Flight embeddings...")
        flight_embeddings = generate_embeddings(flight_texts, model_key)
        
        print("Creating vector index for Flights...")
        create_vector_index(driver, model_key, "Flight")
        
        print("Storing Flight embeddings...")
        store_flight_embeddings(driver, flights, flight_embeddings, model_key)
    
    print(f"\n{'='*60}")
    print(f"Embedding generation complete for {model_key}!")
    print(f"{'='*60}\n")


def compare_models(driver: Driver, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Compare search results between the two embedding models.
    
    Args:
        driver: Neo4j driver instance
        query: Search query
        top_k: Number of results to compare
        
    Returns:
        Dict with results from both models
    """
    results = {}
    
    for model_key in ["minilm", "mpnet"]:
        model_name = EMBEDDING_MODELS[model_key]["name"]
        try:
            journey_results = semantic_search_journeys(driver, query, model_key, top_k)
            flight_results = semantic_search_flights(driver, query, model_key, top_k)
            results[model_name] = {
                "journeys": journey_results,
                "flights": flight_results
            }
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return results
