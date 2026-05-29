"""
ingestion/hybrid_retriever.py  (v3)

Fixes:
  - Score normalization via sigmoid (no more -9.xx scores in output)
  - BM25/dense weights balanced at 0.5/0.5 (RTI fix — was over-weighting BM25)
  - Cross-encoder reranker retained
  - Section number exact-match boost retained
"""

import re
import sys
import math
from typing import Optional

from rank_bm25 import BM25Okapi
from loguru import logger

sys.path.append("..")
from config import TOP_K_RETRIEVAL, TOP_K_RERANK, COHERE_API_KEY

BM25_WEIGHT   = 0.5
VECTOR_WEIGHT = 0.5

# Legal acronym → act name mapping
# When query contains a known acronym, force act_filter to that act
# so BM25 generic word pollution (application, filing etc.) is bypassed
ACRONYM_TO_ACT = {
    "rti":   "Right to Information Act",
    "ipc":   "Indian Penal Code",
    "crpc":  "Code of Criminal Procedure",
    "pocso": "POCSO Act",
    "ibc":   "Insolvency and Bankruptcy Code",
    "fir":   "Code of Criminal Procedure",
}

# High-frequency generic words that pollute BM25 legal retrieval
STOPWORDS = {
    "application", "filing", "filed", "file", "how", "what", "when",
    "where", "which", "who", "please", "tell", "the", "an", "a",
    "to", "of", "in", "is", "are", "can", "do", "does", "for", "and",
    "or", "want", "know", "about", "under", "me", "my", "i",
}


def sigmoid(x: float) -> float:
    """Normalize any score to (0, 1). Safe for large negative/positive values."""
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:

    def __init__(self, vector_store, cohere_api_key: str = ""):
        self.vs             = vector_store
        self._bm25          = None
        self._bm25_corpus   = []
        self._bm25_metadata = []
        self._cohere_key    = cohere_api_key
        self._reranker      = None
        self._load_reranker()

    def _load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.success("Cross-encoder reranker loaded.")
        except Exception as e:
            logger.warning(f"Reranker not loaded: {e}")
            self._reranker = None

    def build_bm25_index(self, texts: list[str], metadatas: list[dict]) -> None:
        logger.info(f"Building BM25 index over {len(texts)} chunks...")
        self._bm25          = BM25Okapi([self._tokenise(t) for t in texts])
        self._bm25_corpus   = texts
        self._bm25_metadata = metadatas
        logger.success("BM25 index built.")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RERANK,
        act_filter: Optional[str] = None,
    ) -> list[dict]:

        # ── Query expansion: detect acronyms, auto-set act_filter ──────────
        expanded_query, auto_filter = self._expand_query(query)
        if auto_filter and not act_filter:
            act_filter = auto_filter
            logger.info(f"Auto act_filter set to: '{act_filter}' from query acronym")

        fetch_k = max(top_k * 4, TOP_K_RETRIEVAL)

        dense_results  = self.vs.query(expanded_query, top_k=fetch_k, act_filter=act_filter)
        sparse_results = self._bm25_retrieve(expanded_query, top_k=fetch_k, act_filter=act_filter)

        dense_results  = self._section_boost(query, dense_results)
        sparse_results = self._section_boost(query, sparse_results)

        fused = self._rrf_merge(dense_results, sparse_results)

        if self._reranker and len(fused) > top_k:
            fused = self._cross_encoder_rerank(query, fused, top_k)
        else:
            fused = fused[:top_k]

        for r in fused:
            raw = r.get("cross_encoder_score",
                  r.get("rrf_score",
                  r.get("score", 0.0)))
            r["normalized_score"] = round(sigmoid(float(raw)), 4)

        for i, r in enumerate(fused):
            r["final_rank"] = i + 1

        return fused

    def _expand_query(self, query: str) -> tuple[str, Optional[str]]:
        """
        Two things:
        1. Detect legal acronyms (RTI, IPC, CrPC) → return matching act name
           so retrieve() can set act_filter automatically.
        2. Remove generic stopwords from the BM25 query to reduce noise.
           Dense query keeps original text (embeddings handle context better).

        Returns: (expanded_query_for_bm25, act_filter_or_None)
        """
        tokens     = query.lower().split()
        act_filter = None

        # Detect acronym → act mapping
        for token in tokens:
            clean = re.sub(r"[^a-z]", "", token)
            if clean in ACRONYM_TO_ACT:
                act_filter = ACRONYM_TO_ACT[clean]
                # Expand: add full act name to query so dense retrieval also benefits
                full_name  = act_filter
                query      = f"{query} {full_name}"
                logger.debug(f"Query expanded: '{query}'")
                break

        # Remove generic stopwords for BM25 (keeps legal terms)
        legal_tokens = [t for t in tokens if t not in STOPWORDS]
        bm25_query   = " ".join(legal_tokens) if legal_tokens else query

        return bm25_query, act_filter

    # ──────────────────────────────────────────────────────────────────────

    def _bm25_retrieve(self, query, top_k, act_filter):
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(self._tokenise(query))
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k * 3]:
            if score == 0:
                break
            meta = self._bm25_metadata[idx]
            if act_filter and meta.get("act_name") != act_filter:
                continue
            results.append({
                "text":     self._bm25_corpus[idx],
                "metadata": meta,
                "score":    float(score),
                "source":   "bm25",
            })
            if len(results) >= top_k:
                break
        return results

    def _section_boost(self, query: str, results: list[dict]) -> list[dict]:
        mentions = re.findall(
            r"(?:section\s+)?(\d{1,4}[A-Z]?)\b", query, re.IGNORECASE
        )
        if not mentions:
            return results
        for r in results:
            sec = str(r["metadata"].get("section_num", ""))
            if sec and sec in mentions:
                r["score"] = r.get("score", 0) + 0.3
                r["source"] = r.get("source", "dense") + "+exact"
        return results

    def _cross_encoder_rerank(self, query, candidates, top_k):
        pairs  = [(query, r["text"][:400]) for r in candidates]
        scores = self._reranker.predict(pairs)
        for r, s in zip(candidates, scores):
            r["cross_encoder_score"] = float(s)
        reranked = sorted(
            candidates,
            key=lambda x: x["cross_encoder_score"],
            reverse=True
        )
        return reranked[:top_k]

    @staticmethod
    def _rrf_merge(dense, sparse, k=60):
        rrf: dict[str, float] = {}
        rmap: dict[str, dict] = {}

        def key(r): return r["text"][:120]

        for rank, r in enumerate(dense):
            k_ = key(r)
            rrf[k_]  = rrf.get(k_, 0) + VECTOR_WEIGHT / (k + rank + 1)
            rmap[k_] = {**r, "source": "dense"}

        for rank, r in enumerate(sparse):
            k_ = key(r)
            rrf[k_]  = rrf.get(k_, 0) + BM25_WEIGHT / (k + rank + 1)
            if k_ not in rmap:
                rmap[k_] = {**r, "source": "sparse"}
            else:
                rmap[k_]["source"] = "hybrid"

        sorted_keys = sorted(rrf, key=rrf.__getitem__, reverse=True)
        merged = []
        for k_ in sorted_keys:
            r = rmap[k_]
            r["rrf_score"] = round(rrf[k_], 6)
            merged.append(r)
        return merged

    @staticmethod
    def _tokenise(text):
        tokens = re.findall(
            r"[a-zA-Z0-9\u0900-\u097F]+(?:[-][a-zA-Z0-9]+)*", text
        )
        return [t.lower() for t in tokens if len(t) > 1]