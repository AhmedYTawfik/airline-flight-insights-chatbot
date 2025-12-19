# ✈️ Airline Flight Insights Assistant

A **Graph-RAG (Retrieval-Augmented Generation)** powered flight analysis system that combines Neo4j Knowledge Graph with multiple LLMs to provide actionable insights for airline operations.

## 🎯 Overview

This application serves as an AI-powered assistant for airline companies to analyze:

- **Flight Delays & On-Time Performance**
- **Route Analysis & Optimization**
- **Passenger Satisfaction Trends**
- **Fleet Performance Monitoring**
- **Loyalty Program Insights**
- **Demographic Travel Patterns**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                       │
├─────────────────────────────────────────────────────────────────┤
│                       Pipeline Layer                             │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │   Cypher    │  │   Embeddings    │  │      Hybrid      │    │
│  │  Retrieval  │  │   Retrieval     │  │    Retrieval     │    │
│  └─────────────┘  └─────────────────┘  └──────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Intent Classification │ Entity Extraction │ Query Selection    │
├─────────────────────────────────────────────────────────────────┤
│                       LLM Providers                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐    │
│  │ Gemini  │  │  Groq   │  │ HuggingFace │  │ (Extensible) │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                Neo4j Knowledge Graph (AuraDB)                    │
│         Journeys • Flights • Airports • Passengers              │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Retrieval Methods

1. **Cypher Only (Baseline)** - Structured queries against the knowledge graph
2. **Embeddings Only** - Semantic similarity search using vector embeddings
3. **Hybrid** - Combines both methods for comprehensive context

### Embedding Models

- **all-MiniLM-L6-v2** - 384 dimensions, faster inference
- **all-mpnet-base-v2** - 768 dimensions, higher quality

### Supported LLMs

- **Google Gemini** (gemini-2.0-flash, gemini-1.5-flash)
- **Groq** (Llama-3-8B, Llama-3-70B, Mixtral-8x7B)
- **HuggingFace** (Mistral-7B and more)

### Query Templates (17+ Cypher Queries)

Covering 5 key intents:

1. **Operational Delay Diagnostics** - Identify delay patterns
2. **Service Quality & Product Optimization** - Food satisfaction analysis
3. **Fleet Performance Monitoring** - Aircraft performance metrics
4. **High-Value Customer Retention** - Loyalty program analysis
5. **Demographic Market Analysis** - Generation-based insights

## 📁 Project Structure

```
AirLine_MS3/
├── .env                          # Environment variables (API keys)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── Airline_surveys_sample.csv    # Sample dataset
├── airline-queries.txt           # Query examples
│
├── src/
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Streamlit web application
│   ├── config.py                # Configuration settings
│   ├── database.py              # Neo4j connection management
│   ├── cypher_queries.py        # Cypher query templates
│   ├── embeddings.py            # Vector embeddings module
│   ├── llm_providers.py         # LLM provider integrations
│   ├── intent_classifier.py     # Intent & entity extraction
│   ├── pipeline.py              # Main RAG pipeline
│   └── Full_pipline.py          # Legacy UI (demo)
│
├── notebooks/
│   ├── main.ipynb               # Full pipeline notebook
│   ├── LLM.ipynb                # LLM comparison notebook
│   └── embeddings_demo.ipynb    # Embeddings demonstration
│
└── Descriptions/
    └── description.txt          # Project requirements
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Neo4j AuraDB instance with airline data
- API keys for LLM providers

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd AirLine_MS3
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create or update `.env` file:

```env
# Neo4j Database
URI=neo4j+s://your-database.databases.neo4j.io
USERNAME=neo4j
PASSWORD=your-password

# LLM API Keys
GOOGLE_API_KEY=your-google-api-key
GROQ=your-groq-api-key
HUGGINGFACEHUB_API_TOKEN=your-hf-token
```

5. **Run the application**

```bash
cd src
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Generate Embeddings (First Time Setup)

Before using semantic search, generate embeddings:

1. Open the Streamlit app
2. Expand "🔧 Embedding Management" in the sidebar
3. Click "Generate MiniLM Embeddings" or "Generate MPNet Embeddings"

## 💡 Usage Examples

### Delay Analysis

- "Which airports have the highest total delays?"
- "What is the average delay for 2-leg flights?"
- "Find routes from LAX with delays over 30 minutes"

### Passenger Satisfaction

- "Which routes have poor food satisfaction in Economy class?"
- "Find flights over 4000 miles with satisfaction rating of 1"

### Fleet Performance

- "Which aircraft has the most delays over 60 minutes?"
- "What percentage of B737-800 flights arrive early?"

### Customer Segments

- "Average delay experienced by Premier Gold passengers?"
- "Which destinations are most popular with Millennials?"

## 📊 Evaluation Metrics

### Quantitative

- Response time (seconds)
- Token count
- Query execution time
- Retrieval accuracy

### Qualitative

- Answer relevance
- Factual accuracy
- Naturalness of response
- Actionable insights

## 🛠️ Development

### Adding New Cypher Queries

Edit `src/cypher_queries.py`:

```python
QueryTemplate(
    query="MATCH (j:Journey)...",
    description="Your query description with ${parameter}",
    intent="your_intent_category",
    parameters=["parameter"]
)
```

### Adding New LLM Providers

Edit `src/llm_providers.py`:

1. Create a new provider class extending `BaseLLMProvider`
2. Add to `MODEL_PROVIDERS` dictionary

## 📝 License

This project is part of the CSEN1095 Advanced Database Lab course at the German University in Cairo.

## 👥 Team

Developed for Milestone 3 - Graph-RAG Travel Assistant

---

**Built with ❤️ using Neo4j, LangChain, and Streamlit**
