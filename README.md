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
[Retrieve] --hybrid dense (pgvector HNSW) + lexical (Postgres GIN), fused via RRF--
   |         --also fires a PokeAPI tool call for structured stat/typing/learnset questions--
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

Inference runs on **Groq** (`llama-3.3-70b-versatile`) for low-latency
generation at every node. Retrieval runs against a **Supabase Postgres**
database with the `pgvector` extension enabled, storing both a `vector(384)`
embedding column (`all-MiniLM-L6-v2`, via `sentence-transformers`) and a
generated `tsvector` column for lexical search, combined at query time with
Reciprocal Rank Fusion (see `src/retriever.py`).

## Why Reciprocal Rank Fusion for retrieval

Lexical (BM25-style) and semantic (embedding) similarity scores live on
different, incomparable scales. Naively adding them (e.g. `0.6 * bm25 +
0.4 * cosine_sim`) means whichever score happens to have a larger numeric
range silently dominates — a real bug I hit doing hybrid search on a
product catalog at work, where lexical bonuses were outweighing semantic
rank signal. RRF sidesteps this by fusing on **rank position** rather
than raw score, which is a more principled fix than tuning weights by
hand. Here, the fusion itself runs as a single SQL query with two CTEs
(`dense_search`, `lexical_search`) rather than in Python, so ranking stays
close to the data.

## Project structure

```
data/corpus.json           Seed knowledge base (format rules, mechanics, strategy)
scripts/ingest_corpus.py   One-time batch embed + UPSERT of data/corpus.json into Supabase
src/retriever.py           Hybrid pgvector + lexical retriever with RRF fusion (Postgres)
src/tools.py               PokeAPI lookup tool for authoritative stats/typing/learnset facts
src/agents.py              Retriever, Reranker/Critic, Answer, and Validator agent nodes (Groq)
src/graph.py                LangGraph orchestration + retry routing
src/logging_utils.py        Per-node latency/cost/confidence logging
eval/eval_set.json          20-question eval set (rules, strategy, deliberate out-of-scope)
eval/run_eval.py            Runs eval set, scores groundedness + correct-decline rate
app.py                      Streamlit chat UI (conversational, multi-turn)
dashboard/app.py            Streamlit ops dashboard: live query + cost/latency/confidence charts
```

## Setup

Requires a Supabase (or any Postgres with the `pgvector` extension enabled)
database and a Groq API key.

```bash
pip install -r requirements.txt

# .env (or export directly)
DATABASE_URL=postgresql://...      # Supabase connection string, pgvector extension enabled
GROQ_API_KEY=your_groq_key_here
```

The `vgc_knowledge_base` table (with `embedding vector(384)` and a generated
`search_vector tsvector` column, plus HNSW and GIN indexes) is expected to
already exist — this repo assumes you've provisioned it directly in the
Supabase SQL editor rather than shipping a migration file.

Ingest the seed corpus into Supabase (embeds + UPSERTs every row in
`data/corpus.json`):
```bash
python -m scripts.ingest_corpus
```

Run a single query from the command line:
```bash
python -m src.graph
```

Launch the conversational chat UI:
```bash
streamlit run app.py
```

Run the eval suite:
```bash
python -m eval.run_eval
```

Launch the ops dashboard (latency/cost/confidence charts, reads
`logs/run_log.jsonl`):
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

## Answer quality notes

A few deliberate choices in `src/agents.py` and `src/retriever.py` that
affect answer quality directly:

- **All structured-output calls (tool-intent routing, reranker, validator)
  use Groq's `response_format={"type": "json_object"}`**, not just a prompt
  asking for JSON. The prompt-only approach fails silently under load; this
  makes malformed JSON an API-level guarantee against, not a hope.
- **The validator fails closed.** If its own JSON response fails to parse,
  it now reports `grounded: False` (routing back through the existing
  retry loop) instead of the old default of `grounded: True`. A validator
  that silently trusts unverified answers the moment it breaks defeats the
  point of having one.
- **Chunks are tagged with `[chunk_id]` in both the answer and validator
  prompts**, and the answer model is asked to cite the chunk id behind each
  claim. This lets the validator check citations against real chunks
  instead of comparing prose to prose, and makes a wrong answer traceable
  to the specific chunk that misled it.
- **Embeddings use `BAAI/bge-small-en-v1.5`** instead of general-purpose
  `all-MiniLM-L6-v2` — same 384 dimensions (no schema change), better at
  short jargon-heavy queries ("Sitrus Berry", "Sucker Punch", "Tera Type").
  BGE's asymmetric training means queries get an instruction prefix
  (`BGE_QUERY_PREFIX` in `src/retriever.py`) and passages don't (see
  `scripts/ingest_corpus.py`). **If you change this, you must re-run
  `python -m scripts.ingest_corpus`** — mixing embeddings from two
  different models in the same table produces meaningless similarity
  scores, since the vectors aren't comparable.
- `answer_node`'s `max_tokens` raised from 400 to 700, since strategic
  answers (archetypes, counter-play, team building) were likely getting
  cut off under the old limit.

## Known limitations / next steps

- Decline detection in `eval/run_eval.py` is currently a substring match
  against a fixed set of phrases (`INSUFFICIENCY_PHRASES`) — it can
  under-count correct declines if the model phrases a refusal differently
  than expected. A model-graded check would be more robust.
- `src/tools.py`'s PokeAPI learnset check doesn't filter by game
  version/generation, so it can report a move as learnable based on older
  games even if it isn't obtainable in the current format.
- Knowledge base is a curated seed corpus, not a full scrape — extending
  coverage is a matter of adding rows to `data/corpus.json` and re-running
  `scripts/ingest_corpus.py`.
- Cost constants in `logging_utils.py` are Groq's list price for
  `llama-3.3-70b-versatile` — update them if you change `MODEL` in
  `src/agents.py` or Groq revises pricing.
- `dashboard/app.py` filters logged events by node name `"retrieve"`/
  `"rerank"`, but the nodes are actually logged as `"retriever"`/
  `"reranker"` — those two charts currently show no data.