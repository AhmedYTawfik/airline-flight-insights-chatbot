"""
Configuration module for the Airline Flight Insights Assistant.
Loads environment variables and provides configuration settings.
"""

import os
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv(find_dotenv())


@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str
    username: str
    password: str


@dataclass
class EmbeddingModelConfig:
    """Embedding model configuration."""
    name: str
    dimensions: int
    property_name: str


# Neo4j configuration
NEO4J_CONFIG = Neo4jConfig(
    uri=os.getenv('NEO4J_URI') or os.getenv('URI') or '',
    username=os.getenv('NEO4J_USERNAME') or 'neo4j',
    password=os.getenv('NEO4J_PASSWORD') or os.getenv('PASSWORD') or ''
)

# API Keys
GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY: Optional[str] = os.getenv('GROQ')
HUGGINGFACE_API_TOKEN: Optional[str] = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Embedding model configurations
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": EmbeddingModelConfig(
        name="all-MiniLM-L6-v2",
        dimensions=384,
        property_name="embedding_minilm"
    ),
    "all-mpnet-base-v2": EmbeddingModelConfig(
        name="all-mpnet-base-v2",
        dimensions=768,
        property_name="embedding_mpnet"
    )
}

# LLM model configurations
LLM_MODELS = {
    "GPT-4": {"provider": "openai", "model": "gpt-4"},
    "GPT-3.5-Turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
    "Gemini-2.0-Flash": {"provider": "google", "model": "gemini-2.0-flash"},
    "Llama-3-8B": {"provider": "groq", "model": "llama3-8b-8192"},
    "Mixtral-8x7B": {"provider": "groq", "model": "mixtral-8x7b-32768"},
}

# Retrieval settings
DEFAULT_TOP_K = 5
DEFAULT_DELAY_THRESHOLD = 30
