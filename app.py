"""
Streamlit Web UI for the VGC RAG Multi-Agent Assistant.
Connects directly to LangGraph orchestration and visualizes agent reasoning,
retrieved Supabase vectors, and hallucination validation metrics.
"""
import streamlit as st
from src.graph import run_query

# 1. Page Configuration & Header
st.set_page_config(
    page_title="VGC Grandmaster AI",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown("""
    <style>
    .stChatMessage {border_radius: 10px; padding: 10px;}
    .stat-box {background-color: #1E1E1E; padding: 15px; border-radius: 8px; border-left: 4px solid #FF4B4B;}
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar - Architecture Observability
with st.sidebar:
    st.title("🏆 VGC RAG Agent")
    st.caption("Regulation M-B Competitive Engine")
    st.markdown("---")
    
    st.subheader("🧠 System Architecture")
    st.markdown("""
    - **Inference:** Groq LPU (`llama-3.3-70b`)
    - **Database:** Supabase (`pgvector`)
    - **Search:** Hybrid RRF (Dense + Lexical)
    - **Orchestration:** LangGraph Self-Correcting Loop
    - **Memory:** Multi-turn Context Condensing
    """)
    st.markdown("---")
    st.info("💡 **Tip:** Ask follow-up questions like 'What moves should it run?' to test conversational memory!")

# 3. Session State Initialization (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome to the Regulation M-B War Room! What matchup, core, or EV spread are we analyzing today?",
            "meta": None
        }
    ]

# 4. Render Existing Chat History
st.title("⚔️ Competitive VGC Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🎮"):
        st.markdown(msg["content"])
        
        # If the assistant message has RAG metadata attached, show an expandable reasoning drawer
        if msg.get("meta"):
            with st.expander("🔍 View Agent Reasoning & Retrieved Context"):
                meta = msg["meta"]
                
                # Show if the query was rephrased
                if meta.get("standalone_query") and meta["standalone_query"] != meta.get("query"):
                    st.markdown(f"**💬 Rephrased Search Query:** *\"{meta['standalone_query']}\"*")
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                col1.metric("Grounded in Context", "Yes ✅" if meta.get("is_grounded") else "No ⚠️")
                col2.metric("Graph Retries", f"{meta.get('_retry_count', 0)}")
                col3.metric("Chunks Retrieved", f"{len(meta.get('retrieved_chunks', []))}")
                
                st.markdown("#### 📚 Retrieved Knowledge Chunks")
                for idx, chunk in enumerate(meta.get("retrieved_chunks", []), 1):
                    st.markdown(f"**{idx}. [{chunk.get('category', 'general').upper()}] {chunk.get('title')}**")
                    st.caption(chunk.get('text'))
                    st.divider()

# 5. Handle New User Input
if prompt := st.chat_input("Ex: How do I counter Urshifu-Rapid-Strike in Tailwind?"):
    # Pass past message history (excluding metadata fields) into run_query
    history_for_rag = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in st.session_state.messages
    ]

    # Display user prompt
    st.session_state.messages.append({"role": "user", "content": prompt, "meta": None})
    with st.chat_message("user", avatar="🎮"):
        st.markdown(prompt)

    # Execute LangGraph RAG workflow with live status spinner
    with st.chat_message("assistant", avatar="🤖"):
        with st.status("⚡ Running LangGraph Multi-Agent Pipeline...", expanded=True) as status:
            st.write("🧠 Analyzing conversation history & condensing query...")
            st.write("🔍 Querying Supabase hybrid index (HNSW + BM25)...")
            st.write("⚖️ Computing Reciprocal Rank Fusion (RRF)...")
            
            # Call our Python graph execution engine with history
            result = run_query(prompt, chat_history=history_for_rag)
            
            st.write("🤖 Groq LPU reranking chunks & drafting strategy...")
            st.write("🛡️ Validator checking for hallucinations...")
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # Extract draft answer and save state
        answer = result.get("draft_answer", "I couldn't generate an answer based on the current knowledge base.")
        st.markdown(answer)

        # Render reasoning expander for the new response
        with st.expander("🔍 View Agent Reasoning & Retrieved Context"):
            if result.get("standalone_query") and result["standalone_query"] != prompt:
                st.markdown(f"**💬 Rephrased Search Query:** *\"{result['standalone_query']}\"*")
                
            col1, col2, col3 = st.columns(3)
            col1.metric("Grounded in Context", "Yes ✅" if result.get("is_grounded") else "No ⚠️")
            col2.metric("Graph Retries", f"{result.get('_retry_count', 0)}")
            col3.metric("Chunks Retrieved", f"{len(result.get('retrieved_chunks', []))}")
            
            st.markdown("#### 📚 Retrieved Knowledge Chunks")
            for idx, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                st.markdown(f"**{idx}. [{chunk.get('category', 'general').upper()}] {chunk.get('title')}**")
                st.caption(chunk.get('text'))
                st.divider()

        # Save assistant response and metadata to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": result
        })