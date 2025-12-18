import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Airline Flight Insights Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with proper z-index for interactivity
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #312e81, #1e3a8a);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass effect cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glowing title */
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #a78bfa, #ec4899, #60a5fa);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        margin-bottom: 1rem;
        text-shadow: 0 0 40px rgba(167, 139, 250, 0.5);
    }
    
    @keyframes shine {
        to {
            background-position: 200% center;
        }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #c4b5fd;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Icon container */
    .icon-container {
        width: 100px;
        height: 100px;
        margin: 0 auto 2rem;
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        border-radius: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        animation: pulse 2s ease-in-out infinite;
        box-shadow: 0 0 60px rgba(139, 92, 246, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Ensure all Streamlit widgets are interactive */
    .stButton, .stTextArea, .stTextInput, .stSelectbox, .stRadio, .stCheckbox {
        position: relative;
        z-index: 999;
        pointer-events: auto;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
        cursor: pointer !important;
        pointer-events: auto !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(139, 92, 246, 0.6) !important;
    }
    
    /* Input fields */
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        background: rgba(30, 27, 75, 0.5) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        pointer-events: auto !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox select:focus {
        border-color: rgba(139, 92, 246, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.2));
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.3);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.4);
    }
    
    /* Loading animation */
    .loading-container {
        text-align: center;
        padding: 3rem;
    }
    
    .loading-spinner {
        width: 80px;
        height: 80px;
        margin: 0 auto 2rem;
        border: 4px solid rgba(139, 92, 246, 0.2);
        border-top: 4px solid #8b5cf6;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        color: #c4b5fd;
        font-size: 1.2rem;
        margin: 0.5rem 0;
        animation: fadeInOut 2s ease-in-out infinite;
    }
    
    @keyframes fadeInOut {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1));
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        animation: slideUp 0.8s ease-out;
    }
    
    .result-card:hover {
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.3);
    }
    
    /* Tab styling */
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
        transition: all 0.3s ease;
        pointer-events: auto;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(139, 92, 246, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        pointer-events: auto;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: rgba(139, 92, 246, 0.1) !important;
        border-left: 4px solid #8b5cf6 !important;
        border-radius: 10px !important;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
        animation: slideUp 0.4s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div class="icon-container">✈️</div>
<h1 class="main-title">Flight Insights Assistant</h1>
<p class="subtitle">
    🚀 Powered by Graph-RAG • Neo4j Knowledge Graph + Advanced LLMs
</p>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    selected_model = st.selectbox(
        "🧠 Select LLM Model",
        ["GPT-4", "GPT-3.5-Turbo", "Claude-3-Sonnet", "Llama-3-8B", "Mistral-7B"],
        help="Choose which language model to use"
    )
    
    retrieval_method = st.radio(
        "🔍 Retrieval Method",
        ["Baseline (Cypher Only)", "Embeddings Only", "Hybrid (Both)"],
        help="Select how to retrieve information"
    )
    
    if retrieval_method in ["Embeddings Only", "Hybrid (Both)"]:
        embedding_model = st.selectbox(
            "📊 Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
        )
    
    st.markdown("---")
    st.markdown("### 📊 Display Options")
    show_cypher = st.checkbox("Show Cypher Queries", value=True)
    show_kg_context = st.checkbox("Show KG Context", value=True)
    show_graph_viz = st.checkbox("Show Graph Viz", value=True)
    show_metadata = st.checkbox("Show Metadata", value=True)
    
    st.markdown("---")
    st.info("💡 **Tip**: Try asking about flight delays, route analysis, or passenger satisfaction!")

# Main query section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Ask Your Question")

user_query = st.text_area(
    "",
    placeholder="Example: Which routes from ORD have the most delays?\nExample: Show me passenger satisfaction for flights to LAX",
    height=120,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    submit_button = st.button("🚀 Get Insights", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if clear_button:
    st.rerun()

# Process query
if submit_button and user_query:
    # Loading animation with placeholder
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        st.markdown("""
        <div class="glass-card loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">🔄 Querying Knowledge Graph...</div>
            <div class="loading-text" style="animation-delay: 0.5s;">🧠 Processing with """ + selected_model + """...</div>
            <div class="loading-text" style="animation-delay: 1s;">⚡ Generating Insights...</div>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(2)  # Simulate processing
    
    # Clear loading animation
    loading_placeholder.empty()
    
    # Success message
    st.success("✅ Query processed successfully!")
    
    # Metadata metrics
    if show_metadata:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🧠</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Model</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{selected_model}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">⚡</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Tokens</div>
                <div style="font-size: 1.5rem; font-weight: 700;">250</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">⏱️</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Response</div>
                <div style="font-size: 1.5rem; font-weight: 700;">1.8s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">🔍</div>
                <div style="color: #c4b5fd; font-size: 0.9rem; margin: 0.5rem 0;">Method</div>
                <div style="font-size: 1.5rem; font-weight: 700;">{retrieval_method.split()[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main insights
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💡 Flight Insights")
    
    st.markdown("""
    Based on the flight data analysis, routes from **ORD (Chicago O'Hare)** with the most delays are:
    """)
    
    # Result cards
    results = [
        {"dest": "LAX", "city": "Los Angeles", "delays": 45, "avg": 67.5, "sat": 3.2},
        {"dest": "SFO", "city": "San Francisco", "delays": 38, "avg": 52.3, "sat": 3.5},
        {"dest": "DFW", "city": "Dallas", "delays": 32, "avg": 48.7, "sat": 3.8}
    ]
    
    for i, r in enumerate(results, 1):
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #8b5cf6, #3b82f6); 
                         border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                         font-size: 1.5rem; font-weight: 800;">{i}</div>
                    <div>
                        <div style="font-size: 1.8rem; font-weight: 700;">ORD → {r['dest']}</div>
                        <div style="color: #c4b5fd; font-size: 0.9rem;">{r['city']}</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.5rem;">⭐</span>
                    <span style="font-size: 1.5rem; font-weight: 700;">{r['sat']}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); 
                     border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="color: #fca5a5; font-size: 0.9rem; margin-bottom: 0.5rem;">Delayed Flights</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #ef4444;">{r['delays']}</div>
                </div>
                <div style="background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.3); 
                     border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="color: #fdba74; font-size: 0.9rem; margin-bottom: 0.5rem;">Avg Delay</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #f97316;">{r['avg']} min</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### 📊 Key Insights")
    st.markdown("""
    - The **LAX route** shows the highest delay frequency and duration, requiring immediate attention
    - Weather patterns and air traffic congestion on West Coast routes may be contributing factors
    - Consider scheduling adjustments or additional buffer time for these high-delay routes
    - Passenger satisfaction scores correlate with delay times
    """)
    
    st.markdown("#### 🎯 Recommendations")
    st.markdown("""
    1. Investigate root causes of LAX route delays
    2. Implement proactive communication strategies for passengers
    3. Review crew scheduling and aircraft allocation
    4. Monitor weather patterns and adjust schedules accordingly
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs for additional info
    tab_list = []
    if show_cypher:
        tab_list.append("📝 Cypher Queries")
    if show_kg_context:
        tab_list.append("🗂️ Retrieved Context")
    if show_graph_viz:
        tab_list.append("🕸️ Graph Visualization")
    
    if tab_list:
        tabs = st.tabs(tab_list)
        tab_idx = 0
        
        if show_cypher:
            with tabs[tab_idx]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**Executed Cypher Query:**")
                st.code("""
MATCH (a:Airport {code: 'ORD'})-[r:ROUTE]->(b:Airport)
MATCH (j:Journey)-[:DEPARTED_FROM]->(a)
WHERE j.delay > 30
RETURN b.code AS destination, COUNT(j) AS delay_count, 
       AVG(j.delay) AS avg_delay
ORDER BY delay_count DESC
LIMIT 5
                """, language="cypher")
                
                st.markdown("**Query Results:**")
                df = pd.DataFrame(results)
                df.columns = ['Destination', 'City', 'Delayed Flights', 'Avg Delay (min)', 'Satisfaction']
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            tab_idx += 1
        
        if show_kg_context:
            with tabs[tab_idx]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline Results (Cypher)**")
                    st.json({"routes_analyzed": 3, "total_delays": 115, "avg_satisfaction": 3.5})
                with col2:
                    st.markdown("**Embedding Results**")
                    st.json({"similar_nodes": 8, "avg_similarity": 0.89, "context_relevance": 0.92})
                st.markdown('</div>', unsafe_allow_html=True)
            tab_idx += 1
        
        if show_graph_viz:
            with tabs[tab_idx]:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                # Create graph
                G = nx.DiGraph()
                nodes = ["ORD", "LAX", "SFO", "DFW"]
                for node in nodes:
                    G.add_node(node)
                edges = [("ORD", "LAX"), ("ORD", "SFO"), ("ORD", "DFW")]
                for edge in edges:
                    G.add_edge(edge[0], edge[1])
                
                pos = nx.spring_layout(G, k=2, iterations=50)
                
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
                    textfont=dict(size=16, color='white', family='Inter'),
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
                        height=500
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nodes Retrieved", len(G.nodes()))
                with col2:
                    st.metric("Relationships", len(G.edges()))
                with col3:
                    st.metric("Graph Density", f"{nx.density(G):.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)

# Example queries expander
with st.expander("📚 Example Queries & Use Cases"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Flight Delay Analysis**")
        st.markdown("- Which routes have the highest delay rates?")
        st.markdown("- Average delay for flights from JFK to LAX?")
        st.markdown("- Show flights with delays over 60 minutes")
        
        st.markdown("**⭐ Passenger Satisfaction**")
        st.markdown("- What are the lowest rated flights?")
        st.markdown("- Routes with best satisfaction scores?")
        st.markdown("- Feedback for business class passengers")
    
    with col2:
        st.markdown("**✈️ Route Performance**")
        st.markdown("- Compare delay rates between airports")
        st.markdown("- Most consistent on-time performance?")
        st.markdown("- Busiest routes by passenger volume")
        
        st.markdown("**🎯 Operational Insights**")
        st.markdown("- Flights needing improvement")
        st.markdown("- Routes to prioritize for enhancement")
        st.markdown("- Delays vs satisfaction correlations")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #c4b5fd;'>
    <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
        <span class='badge'>Neo4j Knowledge Graph</span>
        <span class='badge'>Graph-RAG</span>
        <span class='badge'>Advanced LLMs</span>
    </div>
    <p style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;'>
        German University in Cairo - Milestone 3
    </p>
</div>
""", unsafe_allow_html=True)