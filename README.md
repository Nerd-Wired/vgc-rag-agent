# VGC Rules & Strategy Assistant — Multi-Agent RAG with Observability

A retrieval-augmented, multi-agent system that answers competitive Pokemon
VGC questions (format rules, mechanics, matchup strategy) — grounded in a
curated knowledge base, orchestrated with LangGraph, and instrumented end
to end for cost, latency, and quality.

## Why this project

Most RAG demos stop at "retrieve chunks, call an LLM." The gap between
that and a production AI system is the part most tutorials skip:
validating outputs, measuring cost/latency per step, and having a defined
behavior when the model doesn't actually know the answer. This project
builds that layer explicitly, rather than assuming it away.

## Architecture

```
 query
   |
   v
[Retrieve] --hybrid BM25 + embeddings, fused via RRF--
   |
   v
[Rerank/Critic] --drops irrelevant chunks, scores retrieval confidence--
   |
   v
[Answer] --drafts response, grounded ONLY in kept context--
   |
   v
[Validate] --checks for unsupported claims--
   |
   +--grounded--> done
   |
   +--not grounded (up to 2 retries)--> back to [Retrieve] with wider top_k
```

Every node is logged: latency, input/output tokens, estimated cost, and a
confidence score where applicable (`src/logging_utils.py`).

## Why Reciprocal Rank Fusion for retrieval

Lexical (BM25) and semantic (embedding) similarity scores live on
different, incomparable scales. Naively adding them (e.g. `0.6 * bm25 +
0.4 * cosine_sim`) means whichever score happens to have a larger numeric
range silently dominates — a real bug I hit doing hybrid search on a
product catalog at work, where lexical bonuses were outweighing semantic
rank signal. RRF sidesteps this by fusing on **rank position** rather
than raw score, which is a more principled fix than tuning weights by
hand.

## Project structure

```
data/corpus.json       Seed knowledge base (format rules, mechanics, strategy)
src/retriever.py        Hybrid BM25 + embedding retriever with RRF fusion
src/agents.py           Reranker/Critic, Answer, and Validator agent nodes
src/graph.py            LangGraph orchestration + retry routing
src/logging_utils.py    Per-node latency/cost/confidence logging
eval/eval_set.json      20-question eval set (rules, strategy, deliberate out-of-scope)
eval/run_eval.py        Runs eval set, scores groundedness + correct-decline rate
dashboard/app.py        Streamlit dashboard: live query + cost/latency/confidence charts
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

Run a single query:
```bash
python -m src.graph
```

Run the eval suite:
```bash
python -m eval.run_eval
```

Launch the dashboard:
```bash
streamlit run dashboard/app.py
```

## Eval design

The eval set intentionally includes questions with no answer in the
corpus (e.g. "who won the most recent World Championships"). The metric
that matters most isn't just "did it answer" — it's whether the system
**correctly declines** rather than hallucinating a confident, wrong
answer. This is scored separately as the "correct-decline rate" in
`eval/run_eval.py`.

## Known limitations / next steps

- Knowledge base is a small seed corpus (~20 chunks) for demo purposes —
  swapping in a larger, scraped ruleset would be a drop-in change to
  `data/corpus.json`.
- No persistent vector DB; embeddings are recomputed in memory on
  startup. Fine at this scale, would move to a proper vector store
  (e.g. pgvector, since I've worked with it in production) at larger scale.
- Cost constants in `logging_utils.py` are approximate list prices — swap
  in actuals from your Anthropic Console usage page for precise tracking.
