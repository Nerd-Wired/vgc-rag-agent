"""
Observability dashboard.

Reads logs/run_log.jsonl (written by src/logging_utils.py) and shows:
- latency per node, over time
- cumulative cost
- confidence distribution per node (reranker / validator)
- a live query box that runs the graph and shows the trace

Run: streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import run_query

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "run_log.jsonl"

st.set_page_config(page_title="VGC RAG Agent — Ops Dashboard", layout="wide")
st.title("VGC RAG Agent — Observability Dashboard")

tab_live, tab_metrics = st.tabs(["Live Query", "Metrics"])

with tab_live:
    query = st.text_input("Ask a VGC rules or strategy question:", "How do I beat a Rain team?")
    if st.button("Run"):
        with st.spinner("Running multi-agent pipeline..."):
            state = run_query(query)
        st.subheader("Answer")
        st.write(state["draft_answer"])
        col1, col2, col3 = st.columns(3)
        col1.metric("Grounded", "Yes" if state.get("is_grounded") else "No")
        col2.metric("Retries", state.get("_retry_count", 0))
        col3.metric("Retrieval confidence", f"{state.get('retrieval_confidence', 0):.2f}")

        with st.expander("Retrieved chunks used"):
            for c in state.get("reranked_chunks", []):
                st.markdown(f"**{c['title']}** — {c['text']}")

with tab_metrics:
    if not LOG_PATH.exists():
        st.info("No logs yet — run a query or the eval harness first.")
    else:
        records = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(records)
        df = df[df["node"].isin(["retrieve", "rerank", "answer", "validator"])] if "node" in df else df

        node_events = pd.DataFrame(
            [r for r in records if r["node"] in ("retrieve", "rerank", "answer", "validator")]
        )

        if not node_events.empty:
            st.subheader("Latency per node (ms)")
            st.bar_chart(node_events.groupby("node")["latency_ms"].mean())

            st.subheader("Cumulative cost (USD)")
            node_events["cost_usd"] = node_events["cost_usd"].fillna(0)
            node_events_sorted = node_events.sort_values("ts")
            node_events_sorted["cumulative_cost"] = node_events_sorted["cost_usd"].cumsum()
            st.line_chart(node_events_sorted.set_index("ts")["cumulative_cost"])

            st.subheader("Confidence over time (reranker + validator)")
            conf_events = node_events[node_events["confidence"].notna()]
            if not conf_events.empty:
                st.line_chart(conf_events.pivot_table(index="ts", columns="node", values="confidence"))

            total_cost = node_events["cost_usd"].sum()
            avg_latency = node_events.groupby("run_id")["latency_ms"].sum().mean()
            st.metric("Total spend so far", f"${total_cost:.4f}")
            st.metric("Avg total latency per run", f"{avg_latency:.0f} ms")
        else:
            st.info("No node-level events logged yet.")
