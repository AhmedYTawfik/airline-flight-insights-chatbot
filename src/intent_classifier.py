"""
Intent classification and entity extraction module.
Uses LLM to classify user intent and extract entities from queries.
"""

import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from llm_providers import generate_response, LLMResponse
from cypher_queries import QUERY_TEMPLATES, get_query_descriptions


@dataclass
class ExtractedIntent:
    """Represents extracted intent and entities from a user query."""
    intent: str
    entities: Dict[str, Any]
    query_indices: List[int]
    confidence: float


# Intent keywords for rule-based classification
INTENT_KEYWORDS = {
    "delay_diagnostics": [
        "delay", "delayed", "late", "arrival", "depart", "on-time", "ontime",
        "punctual", "behind schedule", "wait", "waiting"
    ],
    "service_quality": [
        "food", "satisfaction", "quality", "service", "meal", "catering",
        "rating", "rated", "experience", "comfortable", "comfort"
    ],
    "fleet_performance": [
        "aircraft", "fleet", "plane", "airplane", "boeing", "airbus", "737", "787",
        "a320", "erj", "jet", "performance"
    ],
    "loyalty_retention": [
        "loyalty", "premier", "gold", "silver", "platinum", "elite", "frequent",
        "member", "status", "vip", "reward"
    ],
    "demographic_analysis": [
        "generation", "millennial", "boomer", "gen x", "gen z", "age", "demographic",
        "young", "old", "senior"
    ]
}

# Entity patterns for extraction
ENTITY_PATTERNS = {
    "airport_code": r'\b([A-Z]{3})\b',
    "number": r'\b(\d+(?:\.\d+)?)\b',
    "passenger_class": r'\b(economy|business|first)\s*(?:class)?\b',
    "loyalty_level": r'\b(premier\s*(?:gold|silver|1k)|gold|silver|platinum|elite|non-elite)\b',
    "generation": r'\b(millennial|boomer|baby\s*boomer|gen\s*[xz]|generation\s*[xz])\b',
    "fleet_type": r'\b(b7[0-9]{2}(?:-[0-9]+)?|a3[0-9]{2}(?:-[0-9]+)?|erj-[0-9]+|b737(?:-max[0-9])?)\b'
}


def classify_intent_rule_based(query: str) -> Tuple[str, float]:
    """Classify intent using rule-based keyword matching."""
    query_lower = query.lower()
    
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            intent_scores[intent] = score
    
    if not intent_scores:
        return "general", 0.5
    
    best_intent = max(intent_scores, key=intent_scores.get)
    confidence = min(intent_scores[best_intent] / 3.0, 1.0)  # Normalize to max 1.0
    
    return best_intent, confidence


def extract_entities_rule_based(query: str) -> Dict[str, Any]:
    """Extract entities using regex patterns."""
    entities = {}
    query_upper = query.upper()
    query_lower = query.lower()
    
    # Airport codes
    airport_matches = re.findall(ENTITY_PATTERNS["airport_code"], query_upper)
    if airport_matches:
        # Filter out common words that look like airport codes
        valid_codes = [code for code in airport_matches if code not in ['THE', 'AND', 'FOR', 'ARE', 'NOT']]
        if valid_codes:
            entities["origin_station_code"] = valid_codes[0]
    
    # Numbers
    number_matches = re.findall(ENTITY_PATTERNS["number"], query)
    if number_matches:
        # Try to infer the purpose of the number
        entities["x"] = int(float(number_matches[0]))
        if len(number_matches) > 1:
            entities["threshold"] = float(number_matches[1])
    else:
        # Default values
        entities["x"] = 5
    
    # Passenger class
    class_match = re.search(ENTITY_PATTERNS["passenger_class"], query_lower)
    if class_match:
        class_name = class_match.group(1).title()
        entities["class_name"] = class_name
    
    # Loyalty level
    loyalty_match = re.search(ENTITY_PATTERNS["loyalty_level"], query_lower)
    if loyalty_match:
        level = loyalty_match.group(1).replace("  ", " ").title()
        # Normalize the loyalty level format
        if "gold" in level.lower():
            level = "premier gold"
        elif "silver" in level.lower():
            level = "premier silver"
        elif "1k" in level.lower():
            level = "premier 1k"
        entities["loyalty_level"] = level
    
    # Generation
    gen_match = re.search(ENTITY_PATTERNS["generation"], query_lower)
    if gen_match:
        gen = gen_match.group(1).title()
        # Normalize generation names
        if "boomer" in gen.lower():
            gen = "Boomer"
        elif "millennial" in gen.lower():
            gen = "Millennial"
        elif "gen x" in gen.lower() or "generation x" in gen.lower():
            gen = "Gen X"
        elif "gen z" in gen.lower() or "generation z" in gen.lower():
            gen = "Gen Z"
        entities["generation"] = gen
    
    # Fleet type
    fleet_match = re.search(ENTITY_PATTERNS["fleet_type"], query_lower)
    if fleet_match:
        fleet = fleet_match.group(1).upper()
        entities["fleet_type"] = fleet
    
    return entities


def classify_intent_llm(query: str, model_name: str = "Gemini-2.0-Flash") -> ExtractedIntent:
    """Use LLM to classify intent and extract parameters."""
    
    # Build query descriptions
    safe_descriptions = [desc.replace('${', '<').replace('}', '>') for desc in get_query_descriptions()]
    query_list = "\n".join([f"{i}: {desc}" for i, desc in enumerate(safe_descriptions)])
    
    prompt = f"""You are an expert at analyzing user questions about airline flight data.

Available database queries:
{query_list}

Your task:
1. Identify which query indices (0-{len(QUERY_TEMPLATES)-1}) are relevant to answer the user's question
2. Extract ALL required parameters for those queries

Parameter guidelines:
- x: number (default: 5 for counts/limits, 30 for delay thresholds, 4000 for miles)
- origin_station_code: 3-letter airport code (e.g., 'LAX', 'ORD', 'JFK')
- class_name: 'Economy', 'Business', or 'First'
- threshold: numeric value (for food scores use 2-3, for miles use 1000-5000)
- loyalty_level: 'premier gold', 'premier silver', 'premier 1k', or 'non-elite'
- generation: 'Millennial', 'Gen X', 'Boomer'
- fleet_type: Aircraft type like 'B787-9', 'B737-800', 'A320-200'
- legs: 1, 2, or 3
- delay_threshold: delay in minutes (default: 30-60)
- miles: distance threshold (default: 4000)
- miles_threshold: same as miles

Return ONLY a valid JSON array with objects containing "query_index" and "params":
[{{"query_index": 0, "params": {{"x": 5}}}}, {{"query_index": 5, "params": {{"legs": 2}}}}]

User question: {query}

JSON:"""

    response = generate_response(model_name, prompt)
    
    if not response.success:
        # Fallback to rule-based
        intent, confidence = classify_intent_rule_based(query)
        entities = extract_entities_rule_based(query)
        return ExtractedIntent(
            intent=intent,
            entities=entities,
            query_indices=[],
            confidence=confidence
        )
    
    # Parse LLM response
    response_text = response.content.strip()
    response_text = response_text.replace('```json', '').replace('```', '').strip()
    
    try:
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            
            query_indices = []
            all_params = {}
            
            for item in parsed:
                if isinstance(item, dict) and 'query_index' in item:
                    idx = item['query_index']
                    if 0 <= idx < len(QUERY_TEMPLATES):
                        query_indices.append(idx)
                        if 'params' in item:
                            all_params.update(item['params'])
            
            # Determine intent from query indices
            if query_indices:
                intents = [QUERY_TEMPLATES[i].intent for i in query_indices]
                intent = max(set(intents), key=intents.count)
            else:
                intent, _ = classify_intent_rule_based(query)
            
            return ExtractedIntent(
                intent=intent,
                entities=all_params,
                query_indices=query_indices,
                confidence=0.9 if query_indices else 0.5
            )
    except (json.JSONDecodeError, Exception) as e:
        print(f"LLM response parsing error: {e}")
    
    # Fallback
    intent, confidence = classify_intent_rule_based(query)
    entities = extract_entities_rule_based(query)
    return ExtractedIntent(
        intent=intent,
        entities=entities,
        query_indices=[],
        confidence=confidence
    )


def get_relevant_query_indices(query: str, use_llm: bool = True,
                                model_name: str = "Gemini-2.0-Flash") -> List[Dict[str, Any]]:
    """Get relevant query indices with parameters for a user query."""
    
    if use_llm:
        extracted = classify_intent_llm(query, model_name)
        if extracted.query_indices:
            return [
                {"query_index": idx, "params": extracted.entities}
                for idx in extracted.query_indices
            ]
    
    # Rule-based fallback
    intent, _ = classify_intent_rule_based(query)
    entities = extract_entities_rule_based(query)
    
    # Map intent to query indices
    intent_to_queries = {
        "delay_diagnostics": [0, 2, 5],  # Top delays, avg delay by legs
        "service_quality": [6, 7],        # Food satisfaction
        "fleet_performance": [8, 9, 10, 11],  # Aircraft performance
        "loyalty_retention": [12, 13],     # Loyalty program
        "demographic_analysis": [14, 15, 16],  # Demographics
        "general": [0, 6, 12]  # Default mix
    }
    
    query_indices = intent_to_queries.get(intent, [0])
    
    return [{"query_index": idx, "params": entities} for idx in query_indices]
