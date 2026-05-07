# Airline Flight Insights Chatbot

AI-powered flight data analysis using Graph-RAG, Neo4j, and Streamlit. This project combines structured Cypher retrieval with vector-based semantic search to ground LLM responses in airline operational data.

## Features
- Hybrid retrieval (Cypher + embeddings) with a transparent context view
- Multiple LLM options: Groq (Llama 3.3 70B), Gemini (Flash 2.0), optional Hugging Face models
- Embedding model selection (MiniLM or MPNet)
- Streamlit chat UI with sample questions and query/result inspection

## How it works (high level)
1. The user asks a question in the Streamlit UI.
2. Gemini extracts relevant query intents and parameters.
3. Cypher queries and vector search retrieve context from Neo4j.
4. The selected LLM answers using only the retrieved context.
5. The UI displays the answer alongside the raw context and queries.

## Prerequisites
- Python 3.10+
- Neo4j 5.x with vector index support
- At least one LLM API key (Groq or Gemini)

## Setup
```bash
python -m venv .venv
```

```bash
# macOS/Linux
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## Configuration
Set environment variables (a `.env` file is supported via `python-dotenv`):

| Variable | Description |
| -------- | ----------- |
| `NEO4J_URI` | Neo4j connection URI (bolt or neo4j scheme) |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `GROQ_API_KEY` | Groq API key (optional if using Gemini only) |
| `GOOGLE_API_KEY` | Gemini API key |
| `HUGGINGFACEHUB_API_TOKEN` | Optional, enables Hugging Face models |

The app also accepts legacy aliases:
- `URI` → `NEO4J_URI`
- `USERNAME` → `NEO4J_USERNAME`
- `PASSWORD` → `NEO4J_PASSWORD`
- `GROQ` → `GROQ_API_KEY`

## Neo4j data model & indexes
Load the airline knowledge graph into Neo4j with these labels and relationships:
- **Labels**: `Airport`, `Flight`, `Journey`, `Passenger`
- **Relationships**:
  - `(:Flight)-[:DEPARTS_FROM]->(:Airport)`
  - `(:Flight)-[:ARRIVES_AT]->(:Airport)`
  - `(:Journey)-[:ON]->(:Flight)`
  - `(:Passenger)-[:TOOK]->(:Journey)`

For embedding retrieval, create vector indexes and store embeddings on nodes:
- `journey_embedding_minilm` on `Journey.embedding_minilm` (384 dims)
- `journey_embedding_mpnet` on `Journey.embedding_mpnet` (768 dims)
- `flight_embedding_minilm` on `Flight.embedding_minilm` (384 dims)
- `flight_embedding_mpnet` on `Flight.embedding_mpnet` (768 dims)

When creating indexes, match the dimensions and use a similarity metric that aligns with your embedding model (commonly cosine).

Example (adjust label, property, and dimensions for each index):
```cypher
CREATE VECTOR INDEX journey_embedding_minilm
FOR (j:Journey)
ON (j.embedding_minilm)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
```

If embeddings are not available, select **Cypher Only** in the UI.

## Run the app
```bash
streamlit run src/app.py
```

## Project structure
```
src/app.py                     Streamlit app and Graph-RAG logic
notebooks/                     Experiment notebooks and evaluation artifacts
Airline_surveys_sample.csv     Sample dataset used to build the KG
airline-queries.txt            Example airline-related prompts
Descriptions/                  Project requirements and milestone docs
```

## Notebooks
The `notebooks/` folder includes:
- `main.ipynb`: end-to-end workflow and data exploration
- `embeddings_demo.ipynb`: embedding generation and retrieval demo
- `embedding_comparison.ipynb`: embedding model comparison
- `LLM.ipynb`: LLM experiments and prompt testing
- `llm_comparison_results.json`: saved LLM comparison results

## License
MIT (see [LICENSE](./LICENSE)).
