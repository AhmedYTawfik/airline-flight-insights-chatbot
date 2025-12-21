"""
Airline Flight Insights Chatbot - Streamlit UI
A Graph-RAG powered chatbot for airline flight data analysis.
"""

import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import os
import json
import re

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Airline Flight Insights",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Environment Setup
# =============================================================================
load_dotenv(find_dotenv())

NEO4J_URI = os.getenv('NEO4J_URI') or os.getenv('URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME') or os.getenv('USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD') or os.getenv('PASSWORD')
GROQ_API_KEY = os.getenv('GROQ_API_KEY') or os.getenv('GROQ')
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# =============================================================================
# Database Connection
# =============================================================================
@st.cache_resource
def get_driver():
    """Create and cache Neo4j driver."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver

# =============================================================================
# LLM Setup
# =============================================================================
@st.cache_resource
def get_groq_llm():
    """Get Groq LLM (Llama 3.3 70B)."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0
    )

@st.cache_resource
def get_gemini_llm():
    """Get Gemini LLM (Flash 2.0)."""
    return ChatGoogleGenerativeAI(
        api_key=GEMINI_API_KEY,
        model="gemini-2.0-flash",
        temperature=0
    )

@st.cache_resource
def get_hf_models():
    """Get HuggingFace models."""
    models = {}
    if HF_TOKEN:
        try:
            base_models = {
                "Mistral-7B": HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=HF_TOKEN,
                ),
                "Llama3-8B": HuggingFaceEndpoint(
                    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    huggingfacehub_api_token=HF_TOKEN,
                ),
                "Gemma-2B": HuggingFaceEndpoint(
                    repo_id="google/gemma-2-2b-it",
                    huggingfacehub_api_token=HF_TOKEN,
                )
            }
            models = {name: ChatHuggingFace(llm=model) for name, model in base_models.items()}
        except Exception as e:
            st.warning(f"Could not load HuggingFace models: {e}")
    return models

def get_llm(model_name: str):
    """Get LLM by name."""
    if model_name == "Groq (Llama 3.3 70B)":
        return get_groq_llm()
    elif model_name == "Gemini (Flash 2.0)":
        return get_gemini_llm()
    else:
        hf_models = get_hf_models()
        return hf_models.get(model_name.replace("HF: ", ""))

# =============================================================================
# Cypher Queries
# =============================================================================
QUERIES = [
    # Intent 1: Operational Delay Diagnostics
    "MATCH (j:Journey)-[:ON]->(f:Flight)-[:ARRIVES_AT]->(a:Airport) RETURN a.station_code AS destination, SUM(j.arrival_delay_minutes) AS total_delay ORDER BY total_delay DESC LIMIT $x",
    "MATCH (j:Journey)-[:ON]->(f:Flight)-[:ARRIVES_AT]->(a:Airport) RETURN a.station_code AS destination, SUM(j.arrival_delay_minutes) AS total_delay ORDER BY total_delay ASC LIMIT $x",
    "MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport) RETURN a.station_code AS origin, SUM(j.arrival_delay_minutes) AS total_delay ORDER BY total_delay DESC LIMIT $x",
    "MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport) RETURN a.station_code AS origin, SUM(j.arrival_delay_minutes) AS total_delay ORDER BY total_delay ASC LIMIT $x",
    "MATCH (o:Airport {station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport), (j:Journey)-[:ON]->(f) WITH o, d, AVG(j.arrival_delay_minutes) AS avg_delay WHERE avg_delay > $x RETURN o.station_code AS origin, d.station_code AS destination, avg_delay",
    "MATCH (j:Journey {number_of_legs: $x}) RETURN AVG(j.arrival_delay_minutes) AS avg_delay",
    # Intent 2: Service Quality
    "MATCH (o:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport), (j:Journey {passenger_class: $class_name})-[:ON]->(f) WITH o, d, AVG(j.food_satisfaction_score) AS avg_food_score WHERE avg_food_score < $threshold RETURN o.station_code AS origin, d.station_code AS destination, avg_food_score",
    "MATCH (j:Journey {food_satisfaction_score: 1})-[:ON]->(f:Flight) WHERE j.actual_flown_miles > $x RETURN DISTINCT f.flight_number",
    # Intent 3: Fleet Performance
    "MATCH (j:Journey)-[:ON]->(f:Flight) WHERE j.arrival_delay_minutes > $x RETURN f.fleet_type_description AS aircraft_type, COUNT(j) AS delay_frequency ORDER BY delay_frequency DESC LIMIT 1",
    "MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $x}) RETURN AVG(j.food_satisfaction_score) AS avg_food_score",
    "MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $x}) RETURN AVG(j.actual_flown_miles) AS avg_miles",
    "MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $x}) WITH COUNT(j) AS total_flights, COUNT(CASE WHEN j.arrival_delay_minutes < 0 THEN 1 END) AS early_flights RETURN (TOFLOAT(early_flights) / total_flights) * 100 AS early_arrival_percentage",
    # Intent 3b: Aircraft Performance Aggregation
    "MATCH (j:Journey)-[:ON]->(f:Flight) RETURN f.fleet_type_description AS aircraft_type, AVG(j.arrival_delay_minutes) AS avg_delay, COUNT(j) AS flight_count ORDER BY avg_delay ASC LIMIT $x",
    "MATCH (j:Journey)-[:ON]->(f:Flight) RETURN f.fleet_type_description AS aircraft_type, AVG(j.arrival_delay_minutes) AS avg_delay, COUNT(j) AS flight_count ORDER BY avg_delay DESC LIMIT $x",
    "MATCH (j:Journey)-[:ON]->(f:Flight) WITH f.fleet_type_description AS aircraft_type, COUNT(j) AS total, COUNT(CASE WHEN j.arrival_delay_minutes <= 0 THEN 1 END) AS on_time RETURN aircraft_type, (toFloat(on_time) / total) * 100 AS on_time_pct, total AS flight_count ORDER BY on_time_pct DESC LIMIT $x",
    # Intent 4: Loyalty
    "MATCH (p:Passenger {loyalty_program_level: $loyalty_program_level})-[:TOOK]->(j:Journey) RETURN AVG(j.arrival_delay_minutes) AS avg_delay",
    "MATCH (p:Passenger {loyalty_program_level: $loyalty_program_level})-[:TOOK]->(j:Journey) WHERE j.arrival_delay_minutes > $x RETURN p.record_locator AS passenger_id, j.arrival_delay_minutes AS delay",
    # Intent 5: Demographics
    "MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight) WHERE j.actual_flown_miles > $threshold RETURN f.fleet_type_description AS aircraft_type, COUNT(f) AS usage_count ORDER BY usage_count DESC LIMIT 1",
    "MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight) RETURN f.fleet_type_description AS fleet_type, COUNT(f) AS usage_count ORDER BY usage_count DESC LIMIT 1",
    "MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)-[:ARRIVES_AT]->(a:Airport) RETURN a.station_code AS destination, COUNT(p) AS passenger_volume ORDER BY passenger_volume DESC LIMIT $x"
]

QUERY_DESCRIPTIONS = [
    "Identify the top ${x} destination stations with the highest accumulated arrival delay minutes.",
    "Identify the top ${x} destination stations with the lowest accumulated arrival delay minutes.",
    "Identify the top ${x} origin stations with the highest accumulated arrival delay minutes.",
    "Identify the top ${x} origin stations with the lowest accumulated arrival delay minutes.",
    "Find routes from the origin station ${origin_station_code} where the average arrival delay exceeds ${x} minutes.",
    "Calculate the average arrival delay for flights consisting of exactly ${x} legs.",
    "Identify routes for the passenger class ${class_name} where the average food satisfaction score is below ${threshold}.",
    "List the flight numbers for journeys longer than ${x} miles where the food satisfaction score was 1.",
    "Identify the aircraft type that has the highest frequency of arrival delays greater than ${x} minutes.",
    "Calculate the average food satisfaction score for passengers flying on the ${x} fleet.",
    "Calculate the average actual flown miles for the ${x} fleet.",
    "Calculate the percentage of early arrivals for the ${x} fleet.",
    "List the top ${x} aircraft types with the LOWEST average arrival delay (best on-time performance).",
    "List the top ${x} aircraft types with the HIGHEST average arrival delay (worst on-time performance).",
    "List the top ${x} aircraft types by on-time arrival percentage (arrivals with delay <= 0 minutes).",
    "Calculate the average arrival delay experienced by passengers with the loyalty level ${loyalty_program_level}.",
    "Find the record locators for passengers with loyalty level ${loyalty_program_level} who experienced a delay greater than ${x} minutes.",
    "Identify the most common aircraft type used by the ${generation} generation for journeys exceeding ${threshold} miles.",
    "Identify the most frequently used fleet type for the ${generation} generation.",
    "Identify the top ${x} destination stations for the ${generation} generation based on passenger volume."
]

# =============================================================================
# Knowledge Graph Schema
# =============================================================================
@st.cache_data(ttl=3600)
def load_kg_schema(_driver) -> Dict[str, Any]:
    """Query the KG to get valid values for each parameter field."""
    schema = {}
    with _driver.session() as session:
        result = session.run('MATCH (a:Airport) RETURN DISTINCT a.station_code AS code ORDER BY code')
        schema['airport_codes'] = [r['code'] for r in result]
        
        result = session.run('MATCH (j:Journey) RETURN DISTINCT j.passenger_class AS class ORDER BY class')
        schema['passenger_classes'] = [r['class'] for r in result if r['class']]
        
        result = session.run('MATCH (p:Passenger) RETURN DISTINCT p.generation AS gen ORDER BY gen')
        schema['generations'] = [r['gen'] for r in result if r['gen']]
        
        result = session.run('MATCH (p:Passenger) RETURN DISTINCT p.loyalty_program_level AS level ORDER BY level')
        schema['loyalty_levels'] = [r['level'] for r in result if r['level']]
        
        result = session.run('MATCH (f:Flight) RETURN DISTINCT f.fleet_type_description AS fleet ORDER BY fleet')
        schema['fleet_types'] = [r['fleet'] for r in result if r['fleet']]
        
        result = session.run('MATCH (j:Journey) RETURN DISTINCT j.number_of_legs AS legs ORDER BY legs')
        schema['number_of_legs'] = [r['legs'] for r in result if r['legs']]
    return schema

# =============================================================================
# Query Functions
# =============================================================================
def run_query(driver, query_index: int, **params) -> list:
    """Run a query by index with parameters."""
    if query_index < 0 or query_index >= len(QUERIES):
        raise ValueError(f"Query index {query_index} out of range (0-{len(QUERIES)-1})")
    with driver.session() as session:
        result = session.run(QUERIES[query_index], **params)
        return [record.data() for record in result]

def get_context(prompt: str, kg_schema: Dict, gemini_llm) -> list:
    """Use Gemini LLM to identify ALL relevant queries and extract parameters."""
    safe_descriptions = [desc.replace('${', '<').replace('}', '>') for desc in QUERY_DESCRIPTIONS]
    query_list = "\n".join([f"{i}: {desc}" for i, desc in enumerate(safe_descriptions)])
    
    schema_info = (
        "=== DATABASE SCHEMA ===\n"
        "The knowledge graph contains these entities and relationships:\n"
        "- Airport: station_code (e.g., LAX, JFK, ORD)\n"
        "- Flight: flight_number, fleet_type_description\n"
        "- Journey: passenger_class, food_satisfaction_score (1-5), arrival_delay_minutes, actual_flown_miles, number_of_legs\n"
        "- Passenger: generation, loyalty_program_level\n\n"
        "=== VALID VALUES ===\n"
        f"- airport codes: {kg_schema['airport_codes'][:20]}... ({len(kg_schema['airport_codes'])} total)\n"
        f"- generation: {kg_schema['generations']}\n"
        f"- loyalty_program_level: {kg_schema['loyalty_levels']}\n"
        f"- passenger_class: {kg_schema['passenger_classes']}\n"
        f"- fleet_type_description: {kg_schema['fleet_types']}\n"
        f"- number_of_legs: {kg_schema['number_of_legs']}\n"
    )
    
    full_prompt = f"""You are an expert at analyzing user questions about airline flight data and mapping them to database queries.

=== AVAILABLE QUERIES ===
{query_list}

{schema_info}

=== YOUR TASK ===
1. Carefully read and understand the user's question
2. Review ALL available queries above and understand what each one retrieves
3. Identify EVERY query that could help answer the user's question (even partially)
4. Extract the correct parameters for each selected query

=== PARAMETER RULES ===
- x: numeric value for counts/limits (default: 5), delay thresholds (default: 30 minutes), or miles
- origin_station_code: must be an exact airport code from the list
- generation: must match exactly (e.g., 'Baby Boomer' → 'Boomer', 'millennials' → 'Millennial')
- loyalty_program_level: must match exactly (e.g., 'gold member' → 'premier gold')
- class_name: must match exactly from passenger_class list
- For fleet-related queries (indices 9-11), x must be an EXACT fleet type string

=== OUTPUT FORMAT ===
Return a JSON array with ALL relevant queries. Each object must have:
- query_index: the index number of the query (0-{len(QUERIES)-1})
- params: object with parameter names and values

Example: [{{"query_index": 0, "params": {{"x": 5}}}}, {{"query_index": 15, "params": {{"loyalty_program_level": "premier gold"}}}}]

IMPORTANT: 
- Return ALL queries that are relevant, not just one
- Return ONLY the JSON array, no explanation or markdown
- If multiple queries can answer different aspects of the question, include them all
- If no queries are relevant, return an empty array

=== USER QUESTION ===
{prompt}

JSON:"""
    
    response = gemini_llm.invoke(full_prompt)
    response_text = response.content.strip()
    response_text = response_text.replace('```json', '').replace('```', '').strip()

    for line in response_text.split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    json_match = re.search(r'\[\s*\{[^\[]*\}\s*\]', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return []

def format_cypher_query(query: str) -> str:
    """Format a Cypher query with proper line breaks for readability."""
    # Keywords to put on new lines
    keywords = ['MATCH', 'OPTIONAL MATCH', 'WHERE', 'WITH', 'RETURN', 'ORDER BY', 'LIMIT', 'CREATE', 'SET', 'DELETE', 'MERGE']
    
    formatted = query.strip()
    
    # Add newlines before major keywords (except the first one)
    for keyword in keywords:
        # Replace keyword with newline + keyword (but not at the start)
        formatted = re.sub(r'(?<!^)\s+(' + keyword + r')\s+', r'\n\1 ', formatted)
    
    # Clean up multiple spaces
    formatted = re.sub(r' +', ' ', formatted)
    
    return formatted

def format_query_result(driver, query_index: int, **params) -> Dict[str, Any]:
    """Run query and format result as context. Returns dict with query info and results."""
    if query_index < 0 or query_index >= len(QUERIES):
        return {"error": f"Query index {query_index} out of range."}
    
    description = QUERY_DESCRIPTIONS[query_index]
    for name, value in params.items():
        description = description.replace(f"${{{name}}}", str(value))
    
    try:
        results = run_query(driver, query_index, **params)
    except Exception as e:
        return {"query": QUERIES[query_index], "query_formatted": format_cypher_query(QUERIES[query_index]), "params": params, "description": description, "error": str(e), "results": []}
    
    return {
        "query": QUERIES[query_index],
        "query_formatted": format_cypher_query(QUERIES[query_index]),
        "params": params,
        "description": description,
        "results": results
    }

# =============================================================================
# Embedding Functions
# =============================================================================
EMBEDDING_MODELS = {
    "minilm": {"name": "all-MiniLM-L6-v2", "dimensions": 384, "property_name": "embedding_minilm"},
    "mpnet": {"name": "all-mpnet-base-v2", "dimensions": 768, "property_name": "embedding_mpnet"}
}

@st.cache_resource
def get_embedding_model(model_key: str = "minilm") -> SentenceTransformer:
    """Load and cache an embedding model."""
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Use 'minilm' or 'mpnet'")
    return SentenceTransformer(EMBEDDING_MODELS[model_key]["name"])

def create_journey_text(props: Dict[str, Any]) -> str:
    """Create text representation of a Journey node."""
    flight_number = props.get('flight_number', '')
    fleet_type = props.get('fleet_type', '')
    origin = props.get('origin', '')
    destination = props.get('destination', '')
    generation = props.get('generation', '')
    loyalty = props.get('loyalty_program_level', '')
    passenger_class = props.get('passenger_class', 'Economy')
    miles = props.get('actual_flown_miles', 0)
    delay = props.get('arrival_delay_minutes', 0)
    legs = props.get('number_of_legs', 1)
    food_score = props.get('food_satisfaction_score', 3)
    
    parts = []
    if flight_number and origin and destination:
        route_text = f"Flight {flight_number} from {origin} to {destination}"
        if fleet_type:
            route_text += f" operated by {fleet_type}"
        parts.append(route_text + ".")
    
    if generation or loyalty:
        passenger_text = f"Passenger is a {generation}" if generation else "Passenger"
        if loyalty:
            passenger_text += f" with {loyalty} loyalty level"
        parts.append(passenger_text + ".")
    
    delay_text = f"arrived {abs(delay)} minutes early" if delay < 0 else "on time" if delay == 0 else f"delayed {delay} minutes"
    food_labels = {1: "very poor", 2: "poor", 3: "average", 4: "good", 5: "excellent"}
    parts.append(f"{passenger_class} class, {miles:.0f} miles, {legs} leg(s), {delay_text}, {food_labels.get(food_score, 'average')} food.")
    
    return " ".join(parts)

def generate_single_embedding(text: str, model_key: str = "minilm") -> List[float]:
    """Generate embedding for a single text."""
    model = get_embedding_model(model_key)
    return model.encode(text, convert_to_numpy=True).tolist()

def semantic_search_journeys(driver, query_text: str, model_key: str = "minilm", top_k: int = 100) -> List[Dict]:
    """Semantic search on Journey nodes - returns enriched data."""
    query_embedding = generate_single_embedding(query_text, model_key)
    index_name = f"journey_{EMBEDDING_MODELS[model_key]['property_name']}"
    
    search_query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
    YIELD node, score
    MATCH (p:Passenger)-[:TOOK]->(node)-[:ON]->(f:Flight)
    OPTIONAL MATCH (f)-[:DEPARTS_FROM]->(o:Airport)
    OPTIONAL MATCH (f)-[:ARRIVES_AT]->(d:Airport)
    RETURN node.feedback_ID AS feedback_ID, 
           node.passenger_class AS passenger_class,
           node.food_satisfaction_score AS food_satisfaction_score,
           node.arrival_delay_minutes AS arrival_delay_minutes,
           node.actual_flown_miles AS actual_flown_miles,
           node.number_of_legs AS number_of_legs,
           p.generation AS generation,
           p.loyalty_program_level AS loyalty_program_level,
           f.flight_number AS flight_number,
           f.fleet_type_description AS fleet_type,
           o.station_code AS origin,
           d.station_code AS destination,
           score
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(search_query, top_k=top_k, query_embedding=query_embedding)
        return [{**dict(r), "similarity_score": r["score"]} for r in result]

def semantic_search_flights(driver, query_text: str, model_key: str = "minilm", top_k: int = 100) -> List[Dict]:
    """Semantic search on Flight nodes."""
    query_embedding = generate_single_embedding(query_text, model_key)
    index_name = f"flight_{EMBEDDING_MODELS[model_key]['property_name']}"
    
    search_query = f"""
    CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_embedding)
    YIELD node, score
    MATCH (node)-[:DEPARTS_FROM]->(origin:Airport)
    MATCH (node)-[:ARRIVES_AT]->(dest:Airport)
    RETURN node.flight_number AS flight_number, node.fleet_type_description AS fleet_type_description,
           origin.station_code AS origin, dest.station_code AS destination, score
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(search_query, top_k=top_k, query_embedding=query_embedding)
        return [{**dict(r), "similarity_score": r["score"]} for r in result]

def format_embedding_results(results: List[Dict], node_type: str = "Journey") -> str:
    """Format embedding search results as context for LLM."""
    if not results:
        return f"No similar {node_type} nodes found."
    
    lines = [f"Found {len(results)} relevant {node_type} records:"]
    for i, r in enumerate(results[:100], 1):  # Include all 100 for LLM context
        if node_type == "Journey":
            text = create_journey_text(r)
        else:
            text = f"Flight {r.get('flight_number', 'N/A')} operated by {r.get('fleet_type_description', 'N/A')} from {r.get('origin', 'N/A')} to {r.get('destination', 'N/A')}."
        lines.append(f"  {i}. (score: {r['similarity_score']:.3f}) {text}")
    return "\n".join(lines)

def get_embedding_context(driver, query: str, model_key: str = "minilm", top_k: int = 100) -> Dict[str, Any]:
    """Get context from embedding-based semantic search."""
    context = {"journeys": [], "flights": [], "formatted": ""}
    try:
        journey_results = semantic_search_journeys(driver, query, model_key, top_k)
        context["journeys"] = journey_results
        journey_text = format_embedding_results(journey_results, "Journey")
    except Exception as e:
        journey_text = f"Journey search error: {e}"
    
    try:
        flight_results = semantic_search_flights(driver, query, model_key, top_k // 2)
        context["flights"] = flight_results
        flight_text = format_embedding_results(flight_results, "Flight")
    except Exception as e:
        flight_text = f"Flight search error: {e}"
    
    context["formatted"] = f"{journey_text}\n\n{flight_text}"
    return context

# =============================================================================
# Hybrid Retrieval
# =============================================================================
def get_hybrid_context(driver, prompt: str, kg_schema: Dict, gemini_llm, model_key: str = "minilm", top_k: int = 20, 
                       use_cypher: bool = True, use_embeddings: bool = True) -> Dict[str, Any]:
    """Get context from both Cypher queries and embedding search."""
    results = {
        'cypher_results': [],
        'embedding_context': None,
        'combined_context': '',
        'queries_executed': []
    }
    
    # Cypher context
    if use_cypher:
        try:
            query_specs = get_context(prompt, kg_schema, gemini_llm)
            for cq in query_specs:
                query_result = format_query_result(driver, cq['query_index'], **cq['params'])
                results['cypher_results'].append(query_result)
                results['queries_executed'].append({
                    'query': query_result.get('query', ''),
                    'query_formatted': query_result.get('query_formatted', query_result.get('query', '')),
                    'params': query_result.get('params', {}),
                    'description': query_result.get('description', '')
                })
        except Exception as e:
            results['cypher_results'].append({"error": f"Cypher error: {e}"})
    
    # Embedding context
    if use_embeddings:
        try:
            results['embedding_context'] = get_embedding_context(driver, prompt, model_key, top_k)
        except Exception as e:
            results['embedding_context'] = {"formatted": f"Embedding error: {e}"}
    
    # Combine contexts
    context_parts = []
    if use_cypher and results['cypher_results']:
        cypher_text_parts = []
        for cr in results['cypher_results']:
            if 'error' in cr:
                cypher_text_parts.append(f"Error: {cr['error']}")
            else:
                desc = cr.get('description', '')
                res_list = cr.get('results', [])
                if res_list:
                    formatted_results = []
                    for r in res_list:
                        parts = [f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in r.items()]
                        formatted_results.append("  - " + ", ".join(parts))
                    cypher_text_parts.append(f'"{desc}":\n' + "\n".join(formatted_results))
                else:
                    cypher_text_parts.append(f'"{desc}": No data found.')
        context_parts.append("=== STRUCTURED QUERY RESULTS ===\n" + "\n\n".join(cypher_text_parts))
    
    if use_embeddings and results['embedding_context']:
        context_parts.append("=== SEMANTIC SEARCH RESULTS ===\n" + results['embedding_context'].get('formatted', ''))
    
    results['combined_context'] = "\n\n".join(context_parts)
    return results

def answer_with_context(driver, llm, question: str, kg_schema: Dict, gemini_llm,
                        model_key: str = "minilm", use_cypher: bool = True, use_embeddings: bool = True) -> Dict[str, Any]:
    """Answer a question using the specified retrieval method(s)."""
    context_obj = get_hybrid_context(driver, question, kg_schema, gemini_llm, model_key, 
                                     use_cypher=use_cypher, use_embeddings=use_embeddings)
    context = context_obj['combined_context']
    
    prompt = f"""You are an AI assistant for an airline company analyzing flight data.

Based on this context from our knowledge graph, answer the user's question.
Only use information from the context. If insufficient, say so.

CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""
    
    response = llm.invoke(prompt).content
    return {
        "answer": response,
        "context": context_obj
    }

# =============================================================================
# Streamlit UI
# =============================================================================
def main():
    # Custom CSS for clean, professional look
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .context-expander {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown('<div class="main-header">✈️ Airline Flight Insights</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-powered flight data analysis using Graph-RAG</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Restart session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.contexts = []
            st.rerun()
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_options = ["Groq (Llama 3.3 70B)", "Gemini (Flash 2.0)"]
        if HF_TOKEN:
            model_options.extend(["HF: Mistral-7B", "HF: Llama3-8B", "HF: Gemma-2B"])
        
        selected_model = st.selectbox(
            "🤖 LLM Model",
            model_options,
            index=0,
            help="Select the language model for generating answers"
        )
        
        # Retrieval method
        retrieval_method = st.radio(
            "📊 Retrieval Method",
            ["Hybrid (Recommended)", "Cypher Only", "Embeddings Only"],
            index=0,
            help="Choose how to retrieve context from the knowledge graph"
        )
        
        # Embedding model
        embedding_model = st.selectbox(
            "🔍 Embedding Model",
            ["minilm", "mpnet"],
            index=0,
            help="Select embedding model for semantic search"
        )
        
        st.divider()
        st.markdown("### 📈 Database Status")
        
        try:
            driver = get_driver()
            st.success("✅ Connected to Neo4j")
            kg_schema = load_kg_schema(driver)
            st.info(f"📍 {len(kg_schema['airport_codes'])} airports")
            st.info(f"✈️ {len(kg_schema['fleet_types'])} fleet types")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
            st.stop()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "contexts" not in st.session_state:
        st.session_state.contexts = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    
    # Get Gemini for query extraction (always needed)
    gemini_llm = get_gemini_llm()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show context expander for assistant messages
            if message["role"] == "assistant" and i < len(st.session_state.contexts):
                ctx = st.session_state.contexts[i]
                with st.expander("🔍 View Retrieved Context", expanded=False):
                    # Queries executed
                    if ctx.get('queries_executed'):
                        st.markdown("#### 📋 Cypher Queries Executed")
                        for q in ctx['queries_executed']:
                            st.markdown(f"**{q.get('description', 'Query')}**")
                            st.code(q.get('query_formatted', q.get('query', '')), language="cypher")
                            if q.get('params'):
                                st.markdown(f"*Parameters:* `{q['params']}`")
                            st.divider()
                    
                    # Cypher results
                    if ctx.get('cypher_results'):
                        st.markdown("#### 📊 Structured Query Results")
                        for cr in ctx['cypher_results']:
                            if 'error' not in cr:
                                st.markdown(f"**{cr.get('description', '')}**")
                                if cr.get('results'):
                                    st.json(cr['results'][:5])  # Show first 5 results
                    
                    # Embedding results - scrollable container
                    if ctx.get('embedding_context'):
                        st.markdown("#### 🔗 Semantic Search Results")
                        emb_ctx = ctx['embedding_context']
                        
                        # Journey results in scrollable container
                        if emb_ctx.get('journeys'):
                            st.markdown(f"**Journey Records** ({len(emb_ctx['journeys'])} found)")
                            with st.container(height=400):
                                for i, j in enumerate(emb_ctx['journeys'], 1):
                                    text = create_journey_text(j)
                                    st.markdown(f"{i}. **(score: {j['similarity_score']:.3f})** {text}")
                        
                        # Flight results in scrollable container
                        if emb_ctx.get('flights'):
                            st.markdown(f"**Flight Records** ({len(emb_ctx['flights'])} found)")
                            with st.container(height=300):
                                for i, f in enumerate(emb_ctx['flights'], 1):
                                    text = f"Flight {f.get('flight_number', 'N/A')} operated by {f.get('fleet_type_description', 'N/A')} from {f.get('origin', 'N/A')} to {f.get('destination', 'N/A')}."
                                    st.markdown(f"{i}. **(score: {f['similarity_score']:.3f})** {text}")
    
    # Chat input
    user_input = st.chat_input("Ask a question about flight data...")
    
    # Check for pending question from sample buttons
    if st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    # Get the selected LLM
                    llm = get_llm(selected_model)
                    
                    # Determine retrieval settings
                    use_cypher = retrieval_method in ["Hybrid (Recommended)", "Cypher Only"]
                    use_embeddings = retrieval_method in ["Hybrid (Recommended)", "Embeddings Only"]
                    
                    # Get answer
                    result = answer_with_context(
                        driver, llm, user_input, kg_schema, gemini_llm,
                        model_key=embedding_model,
                        use_cypher=use_cypher,
                        use_embeddings=use_embeddings
                    )
                    
                    response = result["answer"]
                    context = result["context"]
                    
                    st.markdown(response)
                    
                    # Show context expander
                    with st.expander("🔍 View Retrieved Context", expanded=False):
                        if context.get('queries_executed'):
                            st.markdown("#### 📋 Cypher Queries Executed")
                            for q in context['queries_executed']:
                                st.markdown(f"**{q.get('description', 'Query')}**")
                                st.code(q.get('query_formatted', q.get('query', '')), language="cypher")
                                if q.get('params'):
                                    st.markdown(f"*Parameters:* `{q['params']}`")
                                st.divider()
                        
                        if context.get('cypher_results'):
                            st.markdown("#### 📊 Structured Query Results")
                            for cr in context['cypher_results']:
                                if 'error' not in cr:
                                    st.markdown(f"**{cr.get('description', '')}**")
                                    if cr.get('results'):
                                        st.json(cr['results'][:5])
                        
                        # Embedding results - scrollable container
                        if context.get('embedding_context'):
                            st.markdown("#### 🔗 Semantic Search Results")
                            emb_ctx = context['embedding_context']
                            
                            # Journey results in scrollable container
                            if emb_ctx.get('journeys'):
                                st.markdown(f"**Journey Records** ({len(emb_ctx['journeys'])} found)")
                                with st.container(height=400):
                                    for i, j in enumerate(emb_ctx['journeys'], 1):
                                        text = create_journey_text(j)
                                        st.markdown(f"{i}. **(score: {j['similarity_score']:.3f})** {text}")
                            
                            # Flight results in scrollable container  
                            if emb_ctx.get('flights'):
                                st.markdown(f"**Flight Records** ({len(emb_ctx['flights'])} found)")
                                with st.container(height=300):
                                    for i, f in enumerate(emb_ctx['flights'], 1):
                                        text = f"Flight {f.get('flight_number', 'N/A')} operated by {f.get('fleet_type_description', 'N/A')} from {f.get('origin', 'N/A')} to {f.get('destination', 'N/A')}."
                                        st.markdown(f"{i}. **(score: {f['similarity_score']:.3f})** {text}")
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.contexts.append(context)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.contexts.append({})

    # Sample questions
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        cols = st.columns(2)
        sample_questions = [
            "What are the top 5 airports with the most delays?",
            "How do Millennials travel compared to Baby Boomers?",
            "Which aircraft type has the best on-time performance?",
            "What are the loyalty program levels for flight number 2?"
        ]
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()


if __name__ == "__main__":
    main()
