"""
LangGraph orchestration for the VGC RAG multi-agent system.

Flow:
    retrieve -> rerank -> answer -> validate --grounded?--> END
                   ^                    |
                   |__ not grounded ____|   (retry, up to MAX_RETRIES)

The validator can send the run back to retrieval (with a widened top_k)
if the answer wasn't grounded in context — this is the "escalate rather
than hallucinate" guardrail called out in the JD's Quality & Governance
responsibilities.
"""
import uuid

from langgraph.graph import END, StateGraph

from src.agents import answer_node, reranker_node, validator_node
from src.logging_utils import log_event, timed_node
from src.retriever import HybridRetriever

MAX_RETRIES = 2


def build_graph(corpus_path: str = "data/corpus.json"):
    retriever = HybridRetriever(corpus_path=corpus_path)

    @timed_node("retrieve")
    def retrieve_node(state: dict) -> dict:
        top_k = state.get("_retry_count", 0) * 3 + 5  # widen search on retry
        chunks = retriever.retrieve(state["query"], top_k=top_k)
        state["retrieved_chunks"] = chunks
        state["_retrieve_meta"] = {}  # no LLM call, so no token cost
        return state

    def route_after_validation(state: dict) -> str:
        grounded = state.get("is_grounded", True)
        retries = state.get("_retry_count", 0)
        if grounded or retries >= MAX_RETRIES:
            return "end"
        state["_retry_count"] = retries + 1
        return "retry"

    graph = StateGraph(dict)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", reranker_node)
    graph.add_node("answer", answer_node)
    graph.add_node("validate", validator_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validation, {"end": END, "retry": "retrieve"}
    )

    return graph.compile()


def run_query(query: str, corpus_path: str = "data/corpus.json") -> dict:
    app = build_graph(corpus_path)
    run_id = str(uuid.uuid4())[:8]
    log_event("run_start", run_id, query=query)

    final_state = app.invoke({"query": query, "run_id": run_id, "_retry_count": 0})

    log_event(
        "run_end",
        run_id,
        grounded=final_state.get("is_grounded"),
        retries=final_state.get("_retry_count", 0),
    )
    return final_state


if __name__ == "__main__":
    result = run_query("What should I run to beat a Trick Room team?")
    print("\n--- ANSWER ---")
    print(result["draft_answer"])
    print(f"\nGrounded: {result['is_grounded']} | Retries: {result['_retry_count']}")
