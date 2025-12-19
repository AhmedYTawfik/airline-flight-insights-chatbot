"""
Embeddings module for semantic search using vector embeddings.
Supports two models: all-MiniLM-L6-v2 and all-mpnet-base-v2
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import Driver

from config import EMBEDDING_MODELS, EmbeddingModelConfig
from database import get_db


# Model cache to avoid reloading
_model_cache: Dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Load and cache an embedding model."""
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(EMBEDDING_MODELS.keys())}")
    
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name}...")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print(f"Model {model_name} loaded successfully!")
    
    return _model_cache[model_name]


def get_model_config(model_name: str) -> EmbeddingModelConfig:
    """Get configuration for an embedding model."""
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return EMBEDDING_MODELS[model_name]


# ============================================================================
# Text Representation Functions
# ============================================================================

def create_journey_text(props: Dict[str, Any]) -> str:
    """Create a natural language text representation of a Journey node."""
    passenger_class = props.get("passenger_class", "Unknown")
    food_score = props.get("food_satisfaction_score", "N/A")
    delay = props.get("arrival_delay_minutes", 0)
    miles = props.get("actual_flown_miles", 0)
    legs = props.get("number_of_legs", 1)
    
    # Delay description
    if delay is None:
        delay_text = "with unknown arrival status"
    elif delay < 0:
        delay_text = f"arrived {abs(delay)} minutes early"
    elif delay == 0:
        delay_text = "arrived on time"
    else:
        delay_text = f"delayed {delay} minutes"
    
    # Food satisfaction description
    food_labels = {1: "very poor", 2: "poor", 3: "average", 4: "good", 5: "excellent"}
    satisfaction = food_labels.get(food_score, "unknown")
    
    return (
        f"A {passenger_class} class journey covering {miles:.0f} miles over "
        f"{legs} segment{'s' if legs > 1 else ''}. Flight {delay_text}, "
        f"{satisfaction} food satisfaction (score: {food_score}/5)."
    )


def create_flight_text(props: Dict[str, Any], origin: str = None, destination: str = None) -> str:
    """Create a natural language text representation of a Flight node."""
    flight_num = props.get("flight_number", "Unknown")
    fleet = props.get("fleet_type_description", "Unknown aircraft")
    
    route = ""
    if origin and destination:
        route = f" from {origin} to {destination}"
    elif origin:
        route = f" departing from {origin}"
    elif destination:
        route = f" arriving at {destination}"
    
    return f"Flight {flight_num} operated by {fleet}{route}."


def create_passenger_text(props: Dict[str, Any]) -> str:
    """Create a natural language text representation of a Passenger node."""
    generation = props.get("generation", "unknown")
    loyalty = props.get("loyalty_program_level", "unknown")
    return f"A {generation} passenger with {loyalty} loyalty status."


# ============================================================================
# Embedding Generation Functions
# ============================================================================

def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = get_model(model_name)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def generate_single_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Generate embedding for a single text."""
    model = get_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


# ============================================================================
# Neo4j Node Fetching Functions
# ============================================================================

def fetch_journey_nodes() -> List[Dict[str, Any]]:
    """Fetch all Journey nodes from Neo4j."""
    db = get_db()
    query = """
    MATCH (j:Journey)
    RETURN j.feedback_ID AS feedback_ID, 
           j.passenger_class AS passenger_class,
           j.food_satisfaction_score AS food_satisfaction_score,
           j.arrival_delay_minutes AS arrival_delay_minutes,
           j.actual_flown_miles AS actual_flown_miles, 
           j.number_of_legs AS number_of_legs
    """
    results = db.run_query(query)
    return [{"feedback_ID": r["feedback_ID"], "properties": dict(r)} for r in results]


def fetch_flight_nodes() -> List[Dict[str, Any]]:
    """Fetch all Flight nodes from Neo4j with route info."""
    db = get_db()
    query = """
    MATCH (f:Flight)
    OPTIONAL MATCH (f)-[:DEPARTS_FROM]->(origin:Airport)
    OPTIONAL MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
    RETURN f.flight_number AS flight_number, 
           f.fleet_type_description AS fleet_type_description,
           origin.station_code AS origin, 
           dest.station_code AS destination
    """
    results = db.run_query(query)
    return [{
        "flight_number": r["flight_number"],
        "fleet_type_description": r["fleet_type_description"],
        "properties": {
            "flight_number": r["flight_number"],
            "fleet_type_description": r["fleet_type_description"]
        },
        "origin": r["origin"],
        "destination": r["destination"]
    } for r in results]


# ============================================================================
# Vector Index Management
# ============================================================================

def create_vector_index(model_name: str, node_label: str = "Journey"):
    """Create a vector index in Neo4j for semantic search."""
    db = get_db()
    config = get_model_config(model_name)
    index_name = f"{node_label.lower()}_{config.property_name}"
    
    # Drop existing index if it exists
    try:
        db.run_write_query(f"DROP INDEX {index_name} IF EXISTS")
    except:
        pass
    
    # Create new vector index
    create_query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{node_label}) ON n.{config.property_name}
    OPTIONS {{indexConfig: {{
        `vector.dimensions`: {config.dimensions},
        `vector.similarity_function`: 'cosine'
    }}}}
    """
    db.run_write_query(create_query)
    print(f"Created vector index: {index_name}")


def check_vector_index_exists(model_name: str, node_label: str = "Journey") -> bool:
    """Check if a vector index exists."""
    db = get_db()
    config = get_model_config(model_name)
    index_name = f"{node_label.lower()}_{config.property_name}"
    
    try:
        result = db.run_query("SHOW INDEXES YIELD name RETURN name")
        existing_indexes = [r['name'] for r in result]
        return index_name in existing_indexes
    except:
        return False


# ============================================================================
# Embedding Storage Functions
# ============================================================================

def store_journey_embeddings(feedback_ids: List[str], embeddings: np.ndarray,
                             model_name: str = "all-MiniLM-L6-v2", batch_size: int = 100):
    """Store embeddings for Journey nodes in Neo4j."""
    db = get_db()
    config = get_model_config(model_name)
    prop = config.property_name
    
    query = f"""
    UNWIND $batch AS item
    MATCH (j:Journey {{feedback_ID: item.feedback_ID}})
    SET j.{prop} = item.embedding
    """
    
    for i in range(0, len(feedback_ids), batch_size):
        end_idx = min(i + batch_size, len(feedback_ids))
        batch = [
            {"feedback_ID": feedback_ids[j], "embedding": embeddings[j].tolist()}
            for j in range(i, end_idx)
        ]
        db.run_write_query(query, batch=batch)
        print(f"Stored embeddings: {end_idx}/{len(feedback_ids)}")


def store_flight_embeddings(flights: List[Dict], embeddings: np.ndarray,
                            model_name: str = "all-MiniLM-L6-v2", batch_size: int = 100):
    """Store embeddings for Flight nodes in Neo4j."""
    db = get_db()
    config = get_model_config(model_name)
    prop = config.property_name
    
    query = f"""
    UNWIND $batch AS item
    MATCH (f:Flight {{flight_number: item.flight_number, fleet_type_description: item.fleet_type_description}})
    SET f.{prop} = item.embedding
    """
    
    for i in range(0, len(flights), batch_size):
        end_idx = min(i + batch_size, len(flights))
        batch = [
            {
                "flight_number": flights[j]["flight_number"],
                "fleet_type_description": flights[j]["fleet_type_description"],
                "embedding": embeddings[j].tolist()
            }
            for j in range(i, end_idx)
        ]
        db.run_write_query(query, batch=batch)
        print(f"Stored embeddings: {end_idx}/{len(flights)}")


# ============================================================================
# Semantic Search Functions
# ============================================================================

def semantic_search_journeys(query_text: str, model_name: str = "all-MiniLM-L6-v2",
                             top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search on Journey nodes."""
    db = get_db()
    config = get_model_config(model_name)
    query_embedding = generate_single_embedding(query_text, model_name)
    index_name = f"journey_{config.property_name}"
    
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
    
    results = db.run_query(search_query, top_k=top_k, query_embedding=query_embedding)
    return [{**dict(r), "similarity_score": r["score"]} for r in results]


def semantic_search_flights(query_text: str, model_name: str = "all-MiniLM-L6-v2",
                            top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search on Flight nodes."""
    db = get_db()
    config = get_model_config(model_name)
    query_embedding = generate_single_embedding(query_text, model_name)
    index_name = f"flight_{config.property_name}"
    
    search_query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
    YIELD node, score
    OPTIONAL MATCH (node)-[:DEPARTS_FROM]->(origin:Airport)
    OPTIONAL MATCH (node)-[:ARRIVES_AT]->(dest:Airport)
    RETURN node.flight_number AS flight_number,
           node.fleet_type_description AS fleet_type_description,
           origin.station_code AS origin,
           dest.station_code AS destination,
           score
    ORDER BY score DESC
    """
    
    results = db.run_query(search_query, top_k=top_k, query_embedding=query_embedding)
    return [{**dict(r), "similarity_score": r["score"]} for r in results]


# ============================================================================
# Context Formatting Functions
# ============================================================================

def format_embedding_results(results: List[Dict], node_type: str = "Journey") -> str:
    """Format embedding search results as context text."""
    if not results:
        return f"No similar {node_type} records found."
    
    lines = [f"Found {len(results)} similar {node_type} records:"]
    
    for i, r in enumerate(results, 1):
        if node_type == "Journey":
            text = create_journey_text(r)
        else:
            text = create_flight_text(
                {"flight_number": r.get("flight_number"),
                 "fleet_type_description": r.get("fleet_type_description")},
                r.get("origin"),
                r.get("destination")
            )
        score = r.get('similarity_score', r.get('score', 0))
        lines.append(f"  {i}. (similarity: {score:.3f}) {text}")
    
    return "\n".join(lines)


def get_embedding_context(query: str, model_name: str = "all-MiniLM-L6-v2",
                          top_k: int = 5) -> Dict[str, Any]:
    """Get context from embedding-based semantic search."""
    result = {
        'journey_results': [],
        'flight_results': [],
        'journey_context': '',
        'flight_context': '',
        'combined_context': ''
    }
    
    # Search journeys
    try:
        journey_results = semantic_search_journeys(query, model_name, top_k)
        result['journey_results'] = journey_results
        result['journey_context'] = format_embedding_results(journey_results, "Journey")
    except Exception as e:
        result['journey_context'] = f"Journey search error: {e}"
    
    # Search flights
    try:
        flight_results = semantic_search_flights(query, model_name, top_k)
        result['flight_results'] = flight_results
        result['flight_context'] = format_embedding_results(flight_results, "Flight")
    except Exception as e:
        result['flight_context'] = f"Flight search error: {e}"
    
    # Combine
    result['combined_context'] = f"{result['journey_context']}\n\n{result['flight_context']}"
    
    return result


# ============================================================================
# Batch Processing Functions
# ============================================================================

def generate_and_store_all_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Generate and store embeddings for all Journey and Flight nodes."""
    print(f"\n{'='*60}")
    print(f"Generating embeddings with {model_name}")
    print(f"{'='*60}\n")
    
    # Process Journeys
    print("Fetching Journey nodes...")
    journeys = fetch_journey_nodes()
    print(f"Found {len(journeys)} journeys")
    
    if journeys:
        texts = [create_journey_text(j["properties"]) for j in journeys]
        print("Generating journey embeddings...")
        embeddings = generate_embeddings(texts, model_name)
        
        print("Creating vector index for journeys...")
        create_vector_index(model_name, "Journey")
        
        print("Storing journey embeddings...")
        store_journey_embeddings(
            [j["feedback_ID"] for j in journeys],
            embeddings,
            model_name
        )
    
    # Process Flights
    print("\nFetching Flight nodes...")
    flights = fetch_flight_nodes()
    print(f"Found {len(flights)} flights")
    
    if flights:
        texts = [create_flight_text(f["properties"], f["origin"], f["destination"]) for f in flights]
        print("Generating flight embeddings...")
        embeddings = generate_embeddings(texts, model_name)
        
        print("Creating vector index for flights...")
        create_vector_index(model_name, "Flight")
        
        print("Storing flight embeddings...")
        store_flight_embeddings(flights, embeddings, model_name)
    
    print(f"\n{'='*60}")
    print("Embedding generation complete!")
    print(f"{'='*60}")


def compare_models(query: str, top_k: int = 5) -> Dict[str, Dict[str, Any]]:
    """Compare results from both embedding models."""
    comparison = {}
    
    for model_name in EMBEDDING_MODELS.keys():
        try:
            comparison[model_name] = {
                'journeys': semantic_search_journeys(query, model_name, top_k),
                'flights': semantic_search_flights(query, model_name, top_k)
            }
        except Exception as e:
            comparison[model_name] = {'error': str(e)}
    
    return comparison
