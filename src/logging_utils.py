"""
Lightweight observability layer.

This is the piece most student RAG projects skip. Every agent node logs:
- latency (ms)
- input/output token counts and estimated cost
- a confidence/quality signal specific to that node

Logs are appended as JSON lines to logs/run_log.jsonl so the Streamlit
dashboard (dashboard/app.py) can read them without a database.
"""
import functools
import json
import time
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "run_log.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

# Groq pricing for llama-3.3-70b-versatile (the model actually called in
# src/agents.py), per 1M tokens as of mid-2026. Kept as constants so cost
# tracking is transparent and easy to tune — update if you change MODEL
# in src/agents.py or Groq revises pricing.
PRICE_PER_1M_INPUT = 0.59
PRICE_PER_1M_OUTPUT = 0.79


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * PRICE_PER_1M_INPUT + (
        output_tokens / 1_000_000
    ) * PRICE_PER_1M_OUTPUT


def log_event(node: str, run_id: str, **fields):
    record = {"node": node, "run_id": run_id, "ts": time.time(), **fields}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def timed_node(node_name: str):
    """Decorator for LangGraph node functions: measures latency and logs it.

    Expects the wrapped function to return a dict that may include
    'input_tokens', 'output_tokens', and 'confidence' — these get logged
    alongside latency if present.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            run_id = state.get("run_id", "unknown")
            start = time.perf_counter()
            result_state = fn(state, *args, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            node_meta = result_state.get(f"_{node_name}_meta", {})
            input_tok = node_meta.get("input_tokens", 0)
            output_tok = node_meta.get("output_tokens", 0)
            cost = estimate_cost(input_tok, output_tok)

            log_event(
                node_name,
                run_id,
                latency_ms=round(latency_ms, 1),
                input_tokens=input_tok,
                output_tokens=output_tok,
                cost_usd=round(cost, 6),
                confidence=node_meta.get("confidence"),
            )
            return result_state

        return wrapper

    return decorator