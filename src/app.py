"""
Airline Flight Insights Assistant - Full Pipeline Application
A Graph-RAG powered flight analysis system using Neo4j and multiple LLMs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMBEDDING_MODELS, NEO4J_CONFIG
from database import get_db
from pipeline import run_pipeline, compare_retrieval_methods, PipelineResult
from llm_providers import get_llm_manager, MODEL_PROVIDERS
from embeddings import check_vector_index_exists, generate_and_store_all_embeddings


# Page configuration
st.set_page_config(
    page_title="Airline Flight Insights Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #1e293b, #334155);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #a78bfa;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    .icon-container {
        width: 60px;
        height: 60px;
        margin: 0 auto 1rem;
        background: #6366f1;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }
    
    .stButton > button {
        background: #6366f1 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover {
        background: #4f46e5 !important;
    }
    
    .metric-card {
        background: rgba(139, 92, 246, 0.1);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .result-card {
        background: rgba(139, 92, 246, 0.05);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #c4b5fd;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: white;
    }
    
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-connected {
        color: #22c55e;
    }
    
    .status-disconnected {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


@st.cache_resource
def init_database():
    """Initialize database connection."""
    try:
        db = get_db()
        db.connect()
        return db, True
    except Exception as e:
        return None, False


@st.cache_resource
def get_available_models():
    """Get list of available LLM models."""
    manager = get_llm_manager()
    available = []
    for model_name in MODEL_PROVIDERS.keys():
        try:
            provider = manager.get_provider(model_name)
            if provider and provider.is_available():
                available.append(model_name)
        except:
            pass
    return available if available else ["Gemini-2.0-Flash"]  # Default fallback


# Header
st.markdown("""
<div class="icon-container">✈️</div>
<h1 class="main-title">Flight Insights Assistant</h1>
<p class="subtitle">Graph-RAG powered flight analysis with Neo4j Knowledge Graph</p>
""", unsafe_allow_html=True)


# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Database connection status
    db, connected = init_database()
    st.session_state.db_connected = connected
    
    if connected:
        st.markdown('<p class="status-connected">✅ Connected to Neo4j</p>', unsafe_allow_html=True)
        
        # Show database stats
        with st.expander("📊 Database Stats"):
            try:
                stats = db.get_database_stats()
                for label, count in stats.items():
                    if label != 'error':
                        st.write(f"**{label}**: {count:,} nodes")
            except Exception as e:
                st.error(f"Error fetching stats: {e}")
    else:
        st.markdown('<p class="status-disconnected">❌ Not connected to Neo4j</p>', unsafe_allow_html=True)
        st.warning("Check your .env file for Neo4j credentials")
    
    st.markdown("---")
    
    # Model selection
    available_models = get_available_models()
    selected_model = st.selectbox(
        "🧠 LLM Model",
        available_models,
        help="Select which language model to use"
    )
    
    # Retrieval method
    retrieval_method = st.radio(
        "🔍 Retrieval Method",
        ["Hybrid (Cypher + Embeddings)", "Cypher Only (Baseline)", "Embeddings Only"],
        help="Select how to retrieve information from the knowledge graph"
    )
    
    # Map display names to internal names
    retrieval_map = {
        "Hybrid (Cypher + Embeddings)": "hybrid",
        "Cypher Only (Baseline)": "cypher",
        "Embeddings Only": "embedding"
    }
    retrieval_key = retrieval_map[retrieval_method]
    
    # Embedding model selection
    if retrieval_key in ["embedding", "hybrid"]:
        embedding_model = st.selectbox(
            "📊 Embedding Model",
            list(EMBEDDING_MODELS.keys()),
            help="Select embedding model for semantic search"
        )
    else:
        embedding_model = "all-MiniLM-L6-v2"
    
    st.markdown("---")
    st.markdown("### 📊 Display Options")
    show_cypher = st.checkbox("Show Cypher Queries", value=True)
    show_kg_context = st.checkbox("Show Retrieved Context", value=True)
    show_graph_viz = st.checkbox("Show Graph Visualization", value=True)
    show_metadata = st.checkbox("Show Metadata", value=True)
    
    st.markdown("---")
    
    # Embedding management
    with st.expander("🔧 Embedding Management"):
        st.write("Generate embeddings for semantic search:")
        if st.button("Generate MiniLM Embeddings"):
            with st.spinner("Generating embeddings..."):
                try:
                    generate_and_store_all_embeddings("all-MiniLM-L6-v2")
                    st.success("MiniLM embeddings generated!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("Generate MPNet Embeddings"):
            with st.spinner("Generating embeddings..."):
                try:
                    generate_and_store_all_embeddings("all-mpnet-base-v2")
                    st.success("MPNet embeddings generated!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.markdown("---")
    st.info("💡 **Tip**: Try asking about flight delays, route analysis, or passenger satisfaction!")


# Main query section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Ask Your Question")

user_query = st.text_area(
    "",
    placeholder="Example: Which airports have the most flight delays?\nExample: Show me passenger satisfaction for economy class\nExample: What aircraft type has the best on-time performance?",
    height=120,
    label_visibility="collapsed"
)

col1, col2 = st.columns([2, 1])
with col1:
    submit_button = st.button("🚀 Get Insights", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if clear_button:
    st.session_state.last_result = None
    st.rerun()


# Process query
if submit_button and user_query:
    if not st.session_state.db_connected:
        st.error("❌ Cannot process query: Not connected to Neo4j database")
    else:
        # Show loading
        with st.spinner("🔄 Processing your query..."):
            try:
                # Run the pipeline
                result = run_pipeline(
                    query=user_query,
                    retrieval_method=retrieval_key,
                    llm_model=selected_model,
                    embedding_model=embedding_model,
                    top_k=5
                )
                st.session_state.last_result = result
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.session_state.last_result = None


# Display results
if st.session_state.last_result:
    result: PipelineResult = st.session_state.last_result
    
    # Success message
    if result.llm_response.success:
        st.success("✅ Query processed successfully!")
    else:
        st.warning(f"⚠️ LLM Error: {result.llm_response.error}")
    
    # Metadata metrics
    if show_metadata:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🧠</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Model</div>
                <div style="font-size: 1.2rem; font-weight: 700;">{result.model_used}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">⚡</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Tokens (est.)</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{result.llm_response.tokens_estimate}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">⏱️</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Total Time</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{result.total_time:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            method_display = result.retrieval.method.title()
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🔍</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Method</div>
                <div style="font-size: 1.2rem; font-weight: 700;">{method_display}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main LLM Response
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💡 AI-Generated Insights")
    
    if result.llm_response.success:
        st.markdown(result.llm_response.content)
    else:
        st.error(f"Failed to generate response: {result.llm_response.error}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs for additional info
    tab_list = []
    if show_cypher and (result.retrieval.cypher_queries or result.retrieval.cypher_contexts):
        tab_list.append("📝 Cypher Queries")
    if show_kg_context:
        tab_list.append("🗂️ Retrieved Context")
    if show_graph_viz:
        tab_list.append("🕸️ Graph Visualization")
    
    if tab_list:
        tabs = st.tabs(tab_list)
        tab_idx = 0
        
        # Cypher Queries Tab
        if show_cypher and (result.retrieval.cypher_queries or result.retrieval.cypher_contexts):
            with tabs[tab_idx]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                if result.retrieval.cypher_queries:
                    for i, (query, context) in enumerate(zip(
                        result.retrieval.cypher_queries,
                        result.retrieval.cypher_contexts
                    )):
                        st.markdown(f"**Query {i+1}:**")
                        st.code(query.strip(), language="cypher")
                        
                        # Show results
                        if i < len(result.retrieval.cypher_raw_results):
                            raw_results = result.retrieval.cypher_raw_results[i]
                            if raw_results:
                                df = pd.DataFrame(raw_results)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No results for this query")
                        st.markdown("---")
                else:
                    st.info("No Cypher queries executed (using embedding-only retrieval)")
                
                st.markdown('</div>', unsafe_allow_html=True)
            tab_idx += 1
        
        # Retrieved Context Tab
        if show_kg_context:
            with tabs[tab_idx]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Cypher Query Results**")
                    if result.retrieval.cypher_contexts:
                        for ctx in result.retrieval.cypher_contexts:
                            st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                    else:
                        st.info("No Cypher results")
                
                with col2:
                    st.markdown("**Embedding Search Results**")
                    if result.retrieval.embedding_context:
                        st.text(result.retrieval.embedding_context[:1000] + "..." 
                               if len(result.retrieval.embedding_context) > 1000 
                               else result.retrieval.embedding_context)
                    else:
                        st.info("No embedding results")
                
                st.markdown('</div>', unsafe_allow_html=True)
            tab_idx += 1
        
        # Graph Visualization Tab
        if show_graph_viz:
            with tabs[tab_idx]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                # Build graph from results
                G = nx.DiGraph()
                
                # Add nodes from cypher results
                airports = set()
                for raw_result in result.retrieval.cypher_raw_results:
                    for r in raw_result:
                        if 'origin' in r and r['origin']:
                            airports.add(r['origin'])
                        if 'destination' in r and r['destination']:
                            airports.add(r['destination'])
                        # Check for airport codes in other fields
                        for key in ['station_code', 'airport', 'source', 'target']:
                            if key in r and r[key]:
                                airports.add(r[key])
                
                # Add nodes from flight results
                for flight in result.retrieval.flight_results:
                    if flight.get('origin'):
                        airports.add(flight['origin'])
                    if flight.get('destination'):
                        airports.add(flight['destination'])
                
                if airports:
                    for airport in airports:
                        G.add_node(airport)
                    
                    # Add edges from results
                    for raw_result in result.retrieval.cypher_raw_results:
                        for r in raw_result:
                            if 'origin' in r and 'destination' in r:
                                if r['origin'] and r['destination']:
                                    G.add_edge(r['origin'], r['destination'])
                    
                    for flight in result.retrieval.flight_results:
                        if flight.get('origin') and flight.get('destination'):
                            G.add_edge(flight['origin'], flight['destination'])
                    
                    if len(G.nodes()) > 0:
                        pos = nx.spring_layout(G, k=2, iterations=50)
                        
                        # Create edge traces
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=3, color='#8b5cf6'),
                            hoverinfo='none',
                            mode='lines'
                        )
                        
                        # Create node traces
                        node_x, node_y, node_text = [], [], []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)
                        
                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            text=node_text,
                            textposition="top center",
                            textfont=dict(size=14, color='white'),
                            marker=dict(
                                color='#3b82f6',
                                size=40,
                                line=dict(width=3, color='#8b5cf6')
                            )
                        )
                        
                        fig = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=400
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Graph metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Nodes (Airports)", len(G.nodes()))
                        with col2:
                            st.metric("Edges (Routes)", len(G.edges()))
                        with col3:
                            density = nx.density(G) if len(G.nodes()) > 1 else 0
                            st.metric("Graph Density", f"{density:.3f}")
                    else:
                        st.info("No graph data to visualize from this query")
                else:
                    st.info("No airports found in the query results to visualize")
                
                st.markdown('</div>', unsafe_allow_html=True)


# Example queries section
with st.expander("📚 Example Queries & Use Cases"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Flight Delay Analysis**")
        st.markdown("- Which airports have the highest total delays?")
        st.markdown("- What is the average delay for 2-leg flights?")
        st.markdown("- Which routes from LAX have delays over 30 minutes?")
        
        st.markdown("**⭐ Passenger Satisfaction**")
        st.markdown("- Which routes have poor food satisfaction in Economy?")
        st.markdown("- Find flights over 4000 miles with rating of 1")
        st.markdown("- What is the average food score for B787 flights?")
    
    with col2:
        st.markdown("**✈️ Fleet Performance**")
        st.markdown("- Which aircraft has the most delays over 60 minutes?")
        st.markdown("- What percentage of B737-800 flights arrive early?")
        st.markdown("- Average miles flown by each fleet type?")
        
        st.markdown("**👥 Customer Segments**")
        st.markdown("- Average delay for Premier Gold passengers?")
        st.markdown("- Which destinations are popular with Millennials?")
        st.markdown("- What aircraft do Boomers prefer for long flights?")
    
    st.markdown('</div>', unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #c4b5fd;'>
    <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
        <span class='badge'>Neo4j Knowledge Graph</span>
        <span class='badge'>Graph-RAG</span>
        <span class='badge'>Multi-LLM Support</span>
    </div>
    <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;'>
        German University in Cairo - CSEN1095 Advanced Database Lab - Milestone 3
    </p>
</div>
""", unsafe_allow_html=True)
