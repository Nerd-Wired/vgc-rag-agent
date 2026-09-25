# VGC Rules & Strategy Assistant — Multi-Agent RAG with Observability

A retrieval-augmented, multi-agent system that answers competitive Pokemon
VGC questions (format rules, mechanics, matchup strategy) — grounded in a
curated knowledge base, orchestrated with LangGraph, backed by a persistent
hybrid vector store, and instrumented end to end for cost, latency, and
quality.

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
[Retriever] --condenses follow-ups, routes PokeAPI tool calls, hybrid
   |          BM25 + embedding search fused via RRF--
   v
[Reranker/Critic] --drops irrelevant chunks, scores retrieval confidence--
   |
   v
[Answer] --drafts response, grounded ONLY in kept context, inline [chunk_id] citations--
   |
   v
[Validator] --checks for unsupported claims--
   |
   +--grounded--> done
   |
   +--not grounded (up to 2 retries)--> back to [Retriever] with widened top_k
```

Every node is logged: latency, input/output tokens, estimated cost, and a
confidence score where applicable (`src/logging_utils.py`).

## Retrieval

- **Store:** Supabase Postgres with `pgvector` — persistent, not
  recomputed in memory. Dense search uses an HNSW index; lexical search
  uses a `tsvector`/GIN index. Both run inside a single SQL query via
  `WITH` CTEs, fused with **Reciprocal Rank Fusion (RRF)**.
- **Why RRF over weighted score blending:** lexical (BM25) and semantic
  (embedding) similarity scores live on different, incomparable scales.
  Naively adding them (e.g. `0.6 * bm25 + 0.4 * cosine_sim`) means
  whichever score happens to have a larger numeric range silently
  dominates — a real bug encountered doing hybrid search on a product
  catalog in production, where lexical bonuses were outweighing semantic
  rank signal. RRF sidesteps this by fusing on **rank position** rather
  than raw score.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim) — chosen over a
  general-purpose sentence embedder for better performance on short,
  jargon-heavy queries like "Sitrus Berry" or "Tera Type." BGE is trained
  asymmetrically: queries get an instruction prefix at query time
  (`src/retriever.py`); passages are embedded plain at ingest time
  (`scripts/ingest_corpus.py`). Mixing embeddings from two different
  models in the same table silently produces garbage similarity scores,
  so both must stay in sync.
- **Connection handling:** a pooled `psycopg` connection (`psycopg_pool`)
  with idle/lifetime recycling, sized for Supabase's free/starter tier
  limits — not a fresh connection per query.

## Tool use

Structured Pokémon facts (typing, base stats, abilities, move learnsets)
are answered from live PokeAPI lookups (`src/tools.py`) rather than the
LLM's parametric memory — an intent-classification call in the retriever
node decides when a query needs this, and the result is injected as a
high-priority context chunk alongside RAG retrieval. This eliminates a
whole class of hallucination on hard factual game data that a text
corpus alone can't guarantee.

## Multi-turn memory

Follow-up questions ("What moves should it run?") are condensed into a
standalone search query using recent chat history before retrieval runs,
so conversational context doesn't get lost between turns.

## Project structure

```
data/corpus.json          Seed knowledge base (format rules, mechanics, strategy)
scripts/ingest_corpus.py  Batch-embeds corpus.json and upserts into Postgres/pgvector
src/retriever.py          Hybrid BM25 + embedding retriever with RRF fusion (pgvector)
src/tools.py              Live PokeAPI lookups for structured factual data
src/agents.py             Retriever, Reranker/Critic, Answer, and Validator agent nodes
src/graph.py              LangGraph orchestration + retry routing
src/logging_utils.py      Per-node latency/cost/confidence logging (JSONL)
eval/eval_set.json        20-question eval set (rules, strategy, deliberate out-of-scope)
eval/run_eval.py          Runs eval set, scores groundedness + correct-decline rate
app.py                    Streamlit chat UI with live agent-reasoning drawer
dashboard/app.py          Streamlit ops dashboard: latency/cost/confidence charts
```

## Setup

```
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
export DATABASE_URL=your_supabase_postgres_connection_string
```

Ingest the seed corpus into Postgres (one-time, or after editing
`data/corpus.json`):

```
python -m scripts.ingest_corpus
```

Run a single query from the CLI:

```
python -m src.graph
```

Run the eval suite:

```
python -m eval.run_eval
```

Launch the chat UI:

```
streamlit run app.py
```

Launch the observability dashboard:

```
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
- Caching (embedding model, DB pool) currently uses Streamlit's
  `@st.cache_resource`, which couples `src/retriever.py` to the Streamlit
  runtime. Moving to a plain lazy singleton would let the retrieval and
  agent modules be reused from a non-Streamlit service (e.g. a FastAPI
  layer) without pulling in Streamlit as a dependency.
- Per-node logs are appended to a single JSONL file with no write
  locking — fine for single-user local runs, but concurrent requests
  from multiple users could interleave writes. Would move to SQLite or a
  proper logging sink before any multi-user deployment.
- Cost constants in `logging_utils.py` are approximate Groq list prices
  for `llama-3.3-70b-versatile` — swap in actuals from Groq's console for
  precise tracking.