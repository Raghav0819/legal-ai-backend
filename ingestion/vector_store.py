"""
ingestion/vector_store.py  (v2)

Changes from v1:
  - Better embedding model: BAAI/bge-large-en-v1.5 (optional upgrade)
  - Cross-encoder reranker added (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - BM25 weight increased to 0.6 (legal search needs exact term matching)
  - metadata keys updated for new chunker (section_num, section_title)
  - collection wiped + rebuilt when re-ingesting (--reset flag in ingest.py)
"""

import sys
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger

sys.path.append("..")
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBED_MODEL_NAME,
    TOP_K_RETRIEVAL,
)


class LegalVectorStore:

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB collection '{CHROMA_COLLECTION}' ready. "
            f"Current size: {self.collection.count()} chunks"
        )

    # ── Ingestion ──────────────────────────────────────────────────────────

    def reset_collection(self):
        """Wipe and recreate the collection. Use when re-ingesting from scratch."""
        self.chroma_client.delete_collection(CHROMA_COLLECTION)
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection reset. All existing chunks deleted.")

    def add_nodes(
        self,
        nodes: list[TextNode],
        batch_size: int = 64,
        skip_existing: bool = True,
    ) -> int:
        if not nodes:
            logger.warning("add_nodes called with empty list.")
            return 0

        ids       = [self._node_id(n) for n in nodes]
        texts     = [n.text for n in nodes]
        metadatas = [self._clean_metadata(n.metadata) for n in nodes]

        if skip_existing:
            existing_ids = set(self.collection.get(ids=ids)["ids"])
            new_mask  = [id_ not in existing_ids for id_ in ids]
            ids       = [x for x, k in zip(ids,       new_mask) if k]
            texts     = [x for x, k in zip(texts,     new_mask) if k]
            metadatas = [x for x, k in zip(metadatas, new_mask) if k]
            nodes     = [x for x, k in zip(nodes,     new_mask) if k]

            if not ids:
                logger.info("All chunks already in DB.")
                return 0
            logger.info(f"Adding {len(ids)} new chunks")

        added = 0
        for i in range(0, len(nodes), batch_size):
            b_nodes = nodes[i : i + batch_size]
            b_ids   = ids[i : i + batch_size]
            b_texts = texts[i : i + batch_size]
            b_meta  = metadatas[i : i + batch_size]

            embeddings = self.embed_model.get_text_embedding_batch(
                b_texts, show_progress=True,
            )
            self.collection.upsert(
                ids=b_ids,
                embeddings=embeddings,
                documents=b_texts,
                metadatas=b_meta,
            )
            added += len(b_ids)
            logger.info(f"  Stored batch {i//batch_size + 1} ({added}/{len(ids)} chunks)")

        logger.success(f"Ingestion complete. Total chunks in DB: {self.collection.count()}")
        return added

    # ── Retrieval ──────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = TOP_K_RETRIEVAL,
        act_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> list[dict]:
        where_clause = self._build_where(act_filter, section_filter)

        query_embedding = self.embed_model.get_query_embedding(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self.collection.count(), 1)),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text":     doc,
                "metadata": meta,
                "distance": dist,
                "score":    1 - dist,
                "rank":     rank + 1,
            }
            for rank, (doc, meta, dist) in enumerate(
                zip(docs, metadatas, distances)
            )
        ]

    def collection_stats(self) -> dict:
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0, "unique_acts": 0, "acts": []}

        sample = self.collection.get(
            limit=min(count, 2000),
            include=["metadatas"]
        )
        acts = set(m.get("act_name", "unknown") for m in sample["metadatas"])
        return {
            "total_chunks": count,
            "unique_acts":  len(acts),
            "acts":         sorted(acts),
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _node_id(node: TextNode) -> str:
        # Include act + section + index so identical short texts get unique IDs
        act    = node.metadata.get("act_name", "")
        sec    = node.metadata.get("section_num", "")
        idx    = str(node.metadata.get("chunk_index", 0))
        unique = f"{act}|{sec}|{idx}|{node.text}"
        return hashlib.sha256(unique.encode()).hexdigest()[:32]

    @staticmethod
    def _clean_metadata(meta: dict) -> dict:
        cleaned = {}
        for k, v in meta.items():
            if v is None:
                cleaned[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            else:
                cleaned[k] = str(v)
        return cleaned

    @staticmethod
    def _build_where(
        act_filter: Optional[str],
        section_filter: Optional[str],
    ) -> Optional[dict]:
        clauses = []
        if act_filter:
            clauses.append({"act_name": {"$eq": act_filter}})
        if section_filter:
            clauses.append({"section_num": {"$eq": section_filter}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}