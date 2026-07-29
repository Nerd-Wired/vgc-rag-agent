"""
LangGraph orchestration for the VGC RAG multi-agent system.

Flow:
    retriever -> reranker -> answer -> validator --grounded?--> END
                   ^                    |
                   |__ not grounded ____|   (retry, up to MAX_RETRIES)

The validator can send the run back to retrieval (with a widened top_k)
if the answer wasn't grounded in context — this is the "escalate rather
than hallucinate" guardrail called out in the JD's Quality & Governance
responsibilities.
"""
import uuid

from langgraph.graph import END, StateGraph

# Imported ALL four nodes cleanly from agents.py
from src.agents import answer_node, reranker_node, retriever_node, validator_node
from src.logging_utils import log_event

MAX_RETRIES = 2


def build_graph():
    """Builds and compiles the self-correcting RAG workflow graph."""
    
    def route_after_validation(state: dict) -> str:
        grounded = state.get("is_grounded", True)
        retries = state.get("_retry_count", 0)
        
        # If grounded OR we've hit max retries, end execution cleanly
        if grounded or retries >= MAX_RETRIES:
            return "end"
            
        return "retry"

    graph = StateGraph(dict)
    
    # Register our nodes using exact names
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("answer", answer_node)
    graph.add_node("validator", validator_node)

    # Define execution edge flow
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "answer")
    graph.add_edge("answer", "validator")
    
    # Conditional feedback loop
    graph.add_conditional_edges(
        "validator", 
        route_after_validation, 
        {"end": END, "retry": "retriever"}
    )

    return graph.compile()


def run_query(query: str, chat_history: list[dict] = None) -> dict:
    """Executes the compiled RAG workflow against a user query and history."""
    app = build_graph()
    run_id = str(uuid.uuid4())[:8]
    log_event("run_start", run_id, query=query)

    initial_state = {
        "query": query, 
        "chat_history": chat_history or [],
        "run_id": run_id, 
        "_retry_count": 0
    }

    final_state = app.invoke(initial_state)

    log_event(
        "run_end",
        run_id,
        grounded=final_state.get("is_grounded"),
        retries=final_state.get("_retry_count", 0),
    )
    return final_state


if __name__ == "__main__":
    test_q = "What should I run to beat a Trick Room team?"
    print(f"\n🚀 Launching RAG pipeline for: '{test_q}'...\n")
    result = run_query(test_q)
    
    print("--- ANSWER ---")
    print(result.get("draft_answer", "No answer generated."))
    print(f"\nGrounded: {result.get('is_grounded')} | Retries: {result.get('_retry_count')}")