"""
Agent node implementations.

Each agent is a plain function of (state) -> state, so they can be wired
into a LangGraph StateGraph in graph.py. Each one calls the Anthropic API
directly and stashes token usage into state["_<node>_meta"] so the
logging_utils.timed_node decorator can report cost per node.

Requires ANTHROPIC_API_KEY to be set in the environment.
"""
import json
import os

import anthropic

from src.logging_utils import timed_node

MODEL = "claude-sonnet-4-6"
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _call_claude(system: str, user: str, max_tokens: int = 500) -> tuple[str, int, int]:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


@timed_node("reranker")
def reranker_node(state: dict) -> dict:
    """Critic agent: re-scores retrieved chunks for relevance to the query,
    and flags whether retrieval confidence is high enough to proceed."""
    query = state["query"]
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

    raw, in_tok, out_tok = _call_claude(system, user, max_tokens=300)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"relevant_ids": [c["id"] for c in chunks], "confidence": 0.5, "reason": "parse_failed"}

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
    """Drafts an answer grounded strictly in the reranked context chunks."""
    query = state["query"]
    chunks = state["reranked_chunks"]
    context = "\n".join(f"- {c['text']}" for c in chunks)

    system = (
        "You are a VGC (Pokemon competitive doubles) assistant. Answer the "
        "user's question using ONLY the provided context. If the context is "
        "insufficient, say so explicitly rather than guessing. Be concise and "
        "concrete — this is for a competitive player, not a general audience."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"

    answer, in_tok, out_tok = _call_claude(system, user, max_tokens=400)

    state["draft_answer"] = answer
    state["_answer_meta"] = {"input_tokens": in_tok, "output_tokens": out_tok}
    return state


@timed_node("validator")
def validator_node(state: dict) -> dict:
    """Checks the draft answer is grounded in context (no hallucinated claims).
    If not grounded, routes back to the retriever with a widened query."""
    context = "\n".join(f"- {c['text']}" for c in state["reranked_chunks"])
    answer = state["draft_answer"]

    system = (
        "You are a groundedness validator. Given context and a draft answer, "
        'output ONLY JSON: {"grounded": true/false, "confidence": 0-1, '
        '"unsupported_claims": [...]}. A claim is unsupported if it is not '
        "directly stated or clearly implied by the context."
    )
    user = f"Context:\n{context}\n\nDraft answer:\n{answer}"

    raw, in_tok, out_tok = _call_claude(system, user, max_tokens=250)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"grounded": True, "confidence": 0.5, "unsupported_claims": []}

    state["is_grounded"] = parsed.get("grounded", True)
    state["validation_confidence"] = parsed.get("confidence", 0.5)
    state["unsupported_claims"] = parsed.get("unsupported_claims", [])
    state["_validator_meta"] = {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "confidence": parsed.get("confidence", 0.5),
    }
    return state
