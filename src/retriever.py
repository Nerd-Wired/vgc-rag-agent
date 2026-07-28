"""
Hybrid retriever for the VGC RAG agent.

Combines BM25 (lexical) and dense embedding similarity (semantic) using
Reciprocal Rank Fusion (RRF) — fuses on rank position rather than raw
score, since BM25 and cosine similarity live on different, incomparable
scales and BM25 scores are query-dependent in a way cosine similarity
isn't.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    id: str
    category: str
    title: str
    text: str


def _tokenize(text: str) -> list[str]:
    """
    Domain-aware tokenization for Pokemon VGC.
    
    Preserves hyphenated names (e.g., 'Chien-Pao', 'Will-O-Wisp', 'Urshifu-R')
    and internal periods ('Sp. Atk') while stripping terminal sentence punctuation.
    """
    # Matches words with optional internal hyphens or periods (e.g., chien-pao, sp.atk, porygon-z)
    pattern = r"\b[a-z0-9]+(?:[-.'][a-z0-9]+)*\b"
    return re.findall(pattern, text.lower())


class HybridRetriever:
    def __init__(self, corpus_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.chunks = self._load_corpus(corpus_path)
        self._corpus_texts = [f"{c.title}. {c.text}" for c in self.chunks]

        # Lexical index
        tokenized = [_tokenize(t) for t in self._corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Dense index
        self.embedder = SentenceTransformer(embedding_model)
        self.embeddings = self.embedder.encode(
            self._corpus_texts, normalize_embeddings=True, show_progress_bar=False
        )

    @staticmethod
    def _load_corpus(corpus_path: str) -> list["Chunk"]:
        data = json.loads(Path(corpus_path).read_text())
        return [Chunk(**d) for d in data]

    def _bm25_ranked_ids(self, query: str, candidate_k: int = 20) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        # Slice top candidates immediately to prevent O(N) downstream sorting
        top_indices = np.argsort(scores)[::-1][:candidate_k]
        return [self.chunks[i].id for i in top_indices]

    def _dense_ranked_ids(self, query: str, candidate_k: int = 20) -> list[str]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q_emb
        # Slice top candidates immediately
        top_indices = np.argsort(sims)[::-1][:candidate_k]
        return [self.chunks[i].id for i in top_indices]

    @staticmethod
    def reciprocal_rank_fusion(
        bm25_ranked_ids: list[str],
        dense_ranked_ids: list[str],
        k: int = 60,
    ) -> dict[str, float]:
        """
        score(doc) = sum over rankers of 1 / (k + rank + 1)
        k=60 dampens the influence of any single ranker's #1 pick, so a
        document ranked decently by BOTH rankers beats one ranked #1 by
        only one.
        """
        fused_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ranked_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(dense_ranked_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return fused_scores

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        candidate_k: int = 20, 
        rrf_k: int = 60
    ) -> list[dict]:
        bm25_ids = self._bm25_ranked_ids(query, candidate_k=candidate_k)
        dense_ids = self._dense_ranked_ids(query, candidate_k=candidate_k)

        fused = self.reciprocal_rank_fusion(bm25_ids, dense_ids, k=rrf_k)
        ranked_ids = sorted(fused, key=lambda d: fused[d], reverse=True)

        id_to_chunk = {c.id: c for c in self.chunks}
        results = []
        for doc_id in ranked_ids[:top_k]:
            c = id_to_chunk[doc_id]
            results.append(
                {
                    "id": c.id,
                    "category": c.category,
                    "title": c.title,
                    "text": c.text,
                    "fused_score": round(fused[doc_id], 4),
                }
            )
        return results


if __name__ == "__main__":
    retriever = HybridRetriever(corpus_path="data/corpus.json")
    for r in retriever.retrieve("how do I beat a rain team", top_k=3):
        print(f"[{r['fused_score']}] {r['title']}")