"""
Cypher query templates and execution module.
Contains 17+ query templates covering all required intents.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from database import get_db


@dataclass
class QueryTemplate:
    """Represents a Cypher query template with metadata."""
    query: str
    description: str
    intent: str
    parameters: List[str]


# Query templates organized by intent
QUERY_TEMPLATES: List[QueryTemplate] = [
    # Intent 1: Operational Delay Diagnostics (Queries 0-5)
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:ARRIVES_AT]->(a:Airport)
        RETURN a.station_code AS destination, SUM(j.arrival_delay_minutes) AS total_delay
        ORDER BY total_delay DESC LIMIT $x
        """,
        description="Find the top ${x} destination stations with the highest accumulated arrival delay minutes",
        intent="delay_diagnostics",
        parameters=["x"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:ARRIVES_AT]->(a:Airport)
        RETURN a.station_code AS destination, SUM(j.arrival_delay_minutes) AS total_delay
        ORDER BY total_delay ASC LIMIT $x
        """,
        description="Find the top ${x} destination stations with the lowest accumulated arrival delay minutes",
        intent="delay_diagnostics",
        parameters=["x"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport)
        RETURN a.station_code AS origin, SUM(j.arrival_delay_minutes) AS total_delay
        ORDER BY total_delay DESC LIMIT $x
        """,
        description="Find the top ${x} origin stations with the highest accumulated arrival delay minutes",
        intent="delay_diagnostics",
        parameters=["x"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(a:Airport)
        RETURN a.station_code AS origin, SUM(j.arrival_delay_minutes) AS total_delay
        ORDER BY total_delay ASC LIMIT $x
        """,
        description="Find the top ${x} origin stations with the lowest accumulated arrival delay minutes",
        intent="delay_diagnostics",
        parameters=["x"]
    ),
    QueryTemplate(
        query="""
        MATCH (o:Airport {station_code: $origin_station_code})<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport)
        MATCH (j:Journey)-[:ON]->(f)
        WITH o, d, AVG(j.arrival_delay_minutes) AS avg_delay
        WHERE avg_delay > $threshold
        RETURN o.station_code AS origin, d.station_code AS destination, avg_delay
        ORDER BY avg_delay DESC
        """,
        description="Find routes from ${origin_station_code} where the average arrival delay exceeds ${threshold} minutes",
        intent="delay_diagnostics",
        parameters=["origin_station_code", "threshold"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey {number_of_legs: $legs})
        RETURN AVG(j.arrival_delay_minutes) AS avg_delay, COUNT(j) AS flight_count
        """,
        description="Calculate the average arrival delay for flights with ${legs} legs",
        intent="delay_diagnostics",
        parameters=["legs"]
    ),
    
    # Intent 2: Service Quality & Product Optimization (Queries 6-7)
    QueryTemplate(
        query="""
        MATCH (o:Airport)<-[:DEPARTS_FROM]-(f:Flight)-[:ARRIVES_AT]->(d:Airport)
        MATCH (j:Journey {passenger_class: $class_name})-[:ON]->(f)
        WITH o, d, AVG(j.food_satisfaction_score) AS avg_food_score
        WHERE avg_food_score < $threshold
        RETURN o.station_code AS origin, d.station_code AS destination, avg_food_score
        ORDER BY avg_food_score ASC
        """,
        description="Find routes where ${class_name} class passengers have food satisfaction below ${threshold}",
        intent="service_quality",
        parameters=["class_name", "threshold"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey {food_satisfaction_score: 1})-[:ON]->(f:Flight)
        WHERE j.actual_flown_miles > $miles
        RETURN DISTINCT f.flight_number AS flight_number, f.fleet_type_description AS aircraft,
               j.actual_flown_miles AS miles
        """,
        description="Find flights longer than ${miles} miles where food satisfaction was rated 1",
        intent="service_quality",
        parameters=["miles"]
    ),
    
    # Intent 3: Fleet Performance Monitoring (Queries 8-11)
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WHERE j.arrival_delay_minutes > $delay_threshold
        RETURN f.fleet_type_description AS aircraft_type, COUNT(j) AS delay_count
        ORDER BY delay_count DESC LIMIT 1
        """,
        description="Find the aircraft type with the highest frequency of delays over ${delay_threshold} minutes",
        intent="fleet_performance",
        parameters=["delay_threshold"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $fleet_type})
        RETURN AVG(j.food_satisfaction_score) AS avg_food_score, COUNT(j) AS sample_size
        """,
        description="Calculate the average food satisfaction score for the ${fleet_type} fleet",
        intent="fleet_performance",
        parameters=["fleet_type"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $fleet_type})
        RETURN AVG(j.actual_flown_miles) AS avg_miles, COUNT(j) AS flight_count
        """,
        description="Calculate the average flown miles for the ${fleet_type} fleet",
        intent="fleet_performance",
        parameters=["fleet_type"]
    ),
    QueryTemplate(
        query="""
        MATCH (j:Journey)-[:ON]->(f:Flight {fleet_type_description: $fleet_type})
        WITH COUNT(j) AS total_flights, 
             COUNT(CASE WHEN j.arrival_delay_minutes < 0 THEN 1 END) AS early_flights
        RETURN (toFloat(early_flights) / total_flights) * 100 AS early_arrival_percentage,
               total_flights, early_flights
        """,
        description="Calculate the percentage of early arrivals for the ${fleet_type} fleet",
        intent="fleet_performance",
        parameters=["fleet_type"]
    ),
    
    # Intent 4: High-Value Customer (Loyalty) Retention (Queries 12-13)
    QueryTemplate(
        query="""
        MATCH (p:Passenger {loyalty_program_level: $loyalty_level})-[:TOOK]->(j:Journey)
        RETURN AVG(j.arrival_delay_minutes) AS avg_delay, COUNT(j) AS journey_count
        """,
        description="Calculate the average arrival delay for ${loyalty_level} passengers",
        intent="loyalty_retention",
        parameters=["loyalty_level"]
    ),
    QueryTemplate(
        query="""
        MATCH (p:Passenger {loyalty_program_level: $loyalty_level})-[:TOOK]->(j:Journey)
        WHERE j.arrival_delay_minutes > $delay_threshold
        RETURN p.record_locator AS passenger_id, j.arrival_delay_minutes AS delay,
               j.passenger_class AS class, j.food_satisfaction_score AS food_score
        ORDER BY delay DESC
        """,
        description="Find ${loyalty_level} passengers who experienced delays over ${delay_threshold} minutes",
        intent="loyalty_retention",
        parameters=["loyalty_level", "delay_threshold"]
    ),
    
    # Intent 5: Demographic Market Analysis (Queries 14-16)
    QueryTemplate(
        query="""
        MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        WHERE j.actual_flown_miles > $miles_threshold
        RETURN f.fleet_type_description AS aircraft_type, COUNT(f) AS usage_count
        ORDER BY usage_count DESC LIMIT 1
        """,
        description="Find the most common aircraft used by ${generation} for journeys over ${miles_threshold} miles",
        intent="demographic_analysis",
        parameters=["generation", "miles_threshold"]
    ),
    QueryTemplate(
        query="""
        MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        RETURN f.fleet_type_description AS fleet_type, COUNT(f) AS usage_count
        ORDER BY usage_count DESC LIMIT 5
        """,
        description="Find the most frequently used fleet types by ${generation} passengers",
        intent="demographic_analysis",
        parameters=["generation"]
    ),
    QueryTemplate(
        query="""
        MATCH (p:Passenger {generation: $generation})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)-[:ARRIVES_AT]->(a:Airport)
        RETURN a.station_code AS destination, COUNT(p) AS passenger_volume
        ORDER BY passenger_volume DESC LIMIT $x
        """,
        description="Find the top ${x} destination stations for ${generation} passengers",
        intent="demographic_analysis",
        parameters=["generation", "x"]
    ),
]


def get_query_descriptions() -> List[str]:
    """Get all query descriptions for LLM context."""
    return [qt.description for qt in QUERY_TEMPLATES]


def run_query(query_index: int, **params) -> List[Dict[str, Any]]:
    """Run a query by index with parameters."""
    if query_index < 0 or query_index >= len(QUERY_TEMPLATES):
        raise ValueError(f"Query index {query_index} out of range (0-{len(QUERY_TEMPLATES)-1})")
    
    db = get_db()
    query = QUERY_TEMPLATES[query_index].query
    return db.run_query(query, **params)


def format_query_result(query_index: int, **params) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Run a query and format the result as context.
    Returns: (description, cypher_query, results)
    """
    if query_index < 0 or query_index >= len(QUERY_TEMPLATES):
        return f"Error: Query index {query_index} out of range.", "", []
    
    template = QUERY_TEMPLATES[query_index]
    description = template.description
    
    # Replace parameters in description
    for name, value in params.items():
        description = description.replace(f"${{{name}}}", str(value))
    
    try:
        results = run_query(query_index, **params)
        cypher = template.query.strip()
        
        # Format parameters in cypher for display
        for name, value in params.items():
            if isinstance(value, str):
                cypher = cypher.replace(f"${name}", f"'{value}'")
            else:
                cypher = cypher.replace(f"${name}", str(value))
        
        return description, cypher, results
    except Exception as e:
        return f'Error for "{description}": {e}', template.query, []


def format_results_as_text(description: str, results: List[Dict[str, Any]]) -> str:
    """Format query results as human-readable text."""
    if not results:
        return f'"{description}": No data found.'
    
    lines = [f'"{description}":']
    for r in results:
        parts = []
        for k, v in r.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.2f}")
            else:
                parts.append(f"{k}: {v}")
        lines.append("  - " + ", ".join(parts))
    
    return "\n".join(lines)


def get_queries_by_intent(intent: str) -> List[Tuple[int, QueryTemplate]]:
    """Get all queries for a specific intent."""
    return [(i, qt) for i, qt in enumerate(QUERY_TEMPLATES) if qt.intent == intent]


def search_queries(keywords: List[str]) -> List[Tuple[int, QueryTemplate]]:
    """Search queries by keywords in description."""
    results = []
    for i, qt in enumerate(QUERY_TEMPLATES):
        desc_lower = qt.description.lower()
        if any(kw.lower() in desc_lower for kw in keywords):
            results.append((i, qt))
    return results
