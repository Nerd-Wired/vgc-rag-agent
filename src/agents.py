"""
Agent node implementations using Groq LPU Inference.

Each agent is a plain function of (state) -> state, so they can be wired
into a LangGraph StateGraph in graph.py. Each one calls the Groq API
directly and stashes token usage into state["_<node>_meta"] so the
logging_utils.timed_node decorator can report cost per node.

Requires GROQ_API_KEY to be set in the environment or Streamlit secrets.
"""
import json
import os
import streamlit as st

from groq import Groq
from dotenv import load_dotenv

from src.logging_utils import timed_node
from src.tools import lookup_pokedex_fact
from src.retriever import retrieve

# Using Llama 3.3 70B for high-speed, high-accuracy reasoning and structured outputs
MODEL = "llama-3.3-70b-versatile"

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from environment variables or Streamlit secrets.")

client = Groq(api_key=GROQ_API_KEY)


def _call_groq(
    system: str, user: str, max_tokens: int = 500, json_mode: bool = False
) -> tuple[str, int, int]:
    """Helper to execute chat completions via Groq and return tokens used.

    json_mode=True sets response_format to force valid JSON output from the
    model, instead of just asking nicely in the prompt. This is what actually
    keeps json.loads() from throwing downstream — the prompt saying "output
    ONLY JSON" is a suggestion the model can still ignore under load; the
    response_format constraint is enforced by the API.
    """
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    text = resp.choices[0].message.content
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


@timed_node("retriever")
def retriever_node(state: dict) -> dict:
    """Retrieves candidate context chunks from Supabase RRF hybrid engine,
    rewrites follow-up queries using chat history, AND executes structured PokeAPI 
    tool calls if the user asks about stats, typing, abilities, or move learnsets."""
    raw_query = state["query"]
    chat_history = state.get("chat_history", [])
    
    total_in_tok = 0
    total_out_tok = 0
    
    # 1. Condense conversational follow-ups into a standalone search query
    search_query = raw_query
    if chat_history:
        history_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in chat_history[-4:]
        )
        condense_system = (
            "Given a chat history and the latest user follow-up question, "
            "rephrase the follow-up question into a standalone, self-contained search query. "
            "Do NOT answer the question, just return the rephrased query text."
        )
        condense_user = f"Chat History:\n{history_text}\n\nFollow-up Question: {raw_query}"
        search_query, in_tok, out_tok = _call_groq(condense_system, condense_user, max_tokens=100)
        search_query = search_query.strip()
        total_in_tok += in_tok
        total_out_tok += out_tok

    # 2. TOOL INTENT ROUTING: Ask Llama if we need a structured PokeAPI lookup
    tool_system = (
        "You are an intent parser for a Pokemon VGC tool. Check if the query asks about "
        "a specific Pokemon's Base Stats, Typing, Abilities, Speed tier, or whether it can learn a specific move. "
        "Output ONLY a JSON object: {\"needs_tool\": true/false, \"pokemon\": \"name or null\", \"move\": \"move name or null\"}."
    )
    tool_raw, in_tok, out_tok = _call_groq(
        tool_system, f"Query: {search_query}", max_tokens=100, json_mode=True
    )
    total_in_tok += in_tok
    total_out_tok += out_tok
    
    tool_chunks = []
    try:
        tool_intent = json.loads(tool_raw)
        if isinstance(tool_intent, dict) and tool_intent.get("needs_tool") and tool_intent.get("pokemon"):
            pkmn = str(tool_intent["pokemon"]).strip()
            mv = tool_intent.get("move")
            if mv:
                mv = str(mv).strip()
                if mv.lower() == "null":
                    mv = None
            
            if pkmn.lower() != "null":
                print(f"⚡ [TOOL CALL FIRED] Querying PokeAPI for: Pokemon={pkmn}, Move={mv}...")
                
                # Execute Python tool against live API
                fact_text = lookup_pokedex_fact(pkmn, move_to_check=mv)
                
                # Wrap as a high-priority structured chunk
                tool_chunks.append({
                    "id": f"pokeapi_tool_{pkmn}",
                    "category": "structured_fact",
                    "title": f"PokeAPI Authoritative Data: {pkmn.capitalize()}",
                    "text": fact_text
                })
    except Exception as e:
        print(f"⚠️ [TOOL ROUTING SKIP] Could not parse tool intent: {e}")

    # 3. Execute standard Supabase Hybrid RAG Search (HNSW + BM25)
    top_k = state.get("_retry_count", 0) * 3 + 5
    rag_chunks = retrieve(search_query, top_k=top_k)
    
    # Merge: Put authoritative tool facts at the very top of retrieved context!
    combined_chunks = tool_chunks + rag_chunks
    
    state["standalone_query"] = search_query
    state["retrieved_chunks"] = combined_chunks
    state["_retriever_meta"] = {
        "input_tokens": total_in_tok,
        "output_tokens": total_out_tok,
        "chunks_retrieved": len(combined_chunks), 
        "search_query": search_query,
        "tool_called": len(tool_chunks) > 0
    }
    return state


@timed_node("reranker")
def reranker_node(state: dict) -> dict:
    """Critic agent: re-scores retrieved chunks for relevance to the query,
    and flags whether retrieval confidence is high enough to proceed."""
    query = state.get("standalone_query", state["query"])
    chunks = state["retrieved_chunks"]

    chunk_list = "\n".join(f"[{c['id']}] {c['title']}: {c['text']}" for c in chunks)
    system = (
        "You are a retrieval critic for a Pokemon VGC assistant. Given a user "
        "question and a list of retrieved passages, output ONLY a JSON object: "
        '{"relevant_ids": [...], "confidence": 0-1, "reason": "..."}. '
        "relevant_ids should only include passages that directly help answer the "
        "question. confidence reflects whether the relevant passages are "
        "sufficient to answer fully."
    )
    user = f"Question: {query}\n\nPassages:\n{chunk_list}"

    raw, in_tok, out_tok = _call_groq(system, user, max_tokens=300, json_mode=True)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Reranker response is not a valid JSON object.")
    except Exception:
        # Fail conservative, not confident: keep everything so a real answer attempt can still happen
        parsed = {"relevant_ids": [c["id"] for c in chunks], "confidence": 0.0, "reason": "parse_failed"}

    kept = [c for c in chunks if c["id"] in parsed.get("relevant_ids", [])] or chunks

    state["reranked_chunks"] = kept
    state["retrieval_confidence"] = parsed.get("confidence", 0.5)
    state["_reranker_meta"] = {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "confidence": parsed.get("confidence", 0.5),
    }
    return state


@timed_node("answer")
def answer_node(state: dict) -> dict:
    """Drafts an answer grounded strictly in the reranked context chunks
    and ongoing conversation history."""
    query = state["query"]
    chunks = state.get("reranked_chunks", state.get("retrieved_chunks", []))
    chat_history = state.get("chat_history", [])

    context = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)

    # Format last few messages for conversational context
    history_str = ""
    if chat_history:
        history_str = "Conversation History:\n" + "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in chat_history[-4:]
        ) + "\n\n"

    system = (
        "You are a VGC (Pokemon competitive doubles) assistant. Answer the "
        "user's question using ONLY the provided context, which is a list of "
        "chunks each prefixed with a bracketed [chunk_id]. After any claim "
        "that comes from a specific chunk, cite it inline like [chunk_id]. "
        "If the context is insufficient, say so explicitly rather than "
        "guessing. Be concise and concrete — this is for a competitive "
        "player, not a general audience. Strategic questions (archetypes, "
        "counter-play, team building) deserve a fuller answer than a simple "
        "rules lookup; don't artificially truncate a multi-part explanation."
    )
    user = f"{history_str}Context:\n{context}\n\nLatest Question: {query}"

    answer, in_tok, out_tok = _call_groq(system, user, max_tokens=700)

    state["draft_answer"] = answer
    state["_answer_meta"] = {"input_tokens": in_tok, "output_tokens": out_tok}
    return state


@timed_node("validator")
def validator_node(state: dict) -> dict:
    """Checks the draft answer is grounded in context (no hallucinated claims).
    If not grounded, routes back to the retriever with a widened query."""
    chunks = state.get("reranked_chunks", state.get("retrieved_chunks", []))
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    answer = state["draft_answer"]

    system = (
        "You are a groundedness validator. Given context chunks (each "
        "prefixed with a bracketed [chunk_id]) and a draft answer, output "
        'ONLY JSON: {"grounded": true/false, "confidence": 0-1, '
        '"unsupported_claims": [...]}. A claim is unsupported if it is not '
        "directly stated or clearly implied by the context, or if it cites a "
        "[chunk_id] whose chunk does not actually support it."
    )
    user = f"Context:\n{context}\n\nDraft answer:\n{answer}"

    raw, in_tok, out_tok = _call_groq(system, user, max_tokens=250, json_mode=True)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Validator response is not a valid JSON object.")
    except Exception:
        # Fail CLOSED: if validator output fails parsing, treat as unverified and retry
        parsed = {
            "grounded": False,
            "confidence": 0.0,
            "unsupported_claims": ["validator_parse_failed"],
        }

    is_grounded = bool(parsed.get("grounded", False))
    state["is_grounded"] = is_grounded
    state["validation_confidence"] = parsed.get("confidence", 0.5)
    state["unsupported_claims"] = parsed.get("unsupported_claims", [])

    # Increment _retry_count INSIDE the node if validation fails to prevent recursion loops
    if not is_grounded:
        state["_retry_count"] = state.get("_retry_count", 0) + 1

    state["_validator_meta"] = {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "confidence": parsed.get("confidence", 0.5),
    }
    return state