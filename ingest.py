"""
ingest.py — Main ingestion script (v2)

Usage:
    python ingest.py                  # ingest all PDFs
    python ingest.py --reset          # wipe ChromaDB and re-ingest ← USE THIS NOW
    python ingest.py --preview        # preview chunks without storing
    python ingest.py --stats          # show DB stats only
"""

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
logger.remove()
logger.add(sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO")
logger.add("logs/ingest.log", rotation="10 MB", level="DEBUG")

from config import PDF_DIR, LEGAL_ACTS
from ingestion.pdf_parser import parse_all_pdfs
from ingestion.chunker import LegalChunker, preview_chunks
from ingestion.vector_store import LegalVectorStore
from ingestion.hybrid_retriever import HybridRetriever


def run_ingestion(act_prefix=None, reset=False):
    start = time.time()

    # Parse PDFs
    acts = LEGAL_ACTS
    if act_prefix:
        acts = [a for a in LEGAL_ACTS if a[0].startswith(act_prefix)]

    pages = parse_all_pdfs(PDF_DIR, acts)
    if not pages:
        logger.error(f"No pages parsed. Put PDFs in: {PDF_DIR}")
        sys.exit(1)

    # Chunk — now section-based
    chunker = LegalChunker()
    nodes   = chunker.chunk(pages)
    logger.info(f"Generated {len(nodes)} section-based chunks")

    # Store
    store = LegalVectorStore()

    if reset:
        store.reset_collection()
        logger.warning("ChromaDB wiped. Re-ingesting all chunks...")

    added = store.add_nodes(nodes)
    stats = store.collection_stats()
    logger.success(
        f"ChromaDB: {stats['total_chunks']} chunks | "
        f"{stats['unique_acts']} acts: {', '.join(stats['acts'])}"
    )

    # BM25
    all_data  = store.collection.get(
        limit=stats["total_chunks"],
        include=["documents", "metadatas"],
    )
    retriever = HybridRetriever(store)
    retriever.build_bm25_index(all_data["documents"], all_data["metadatas"])

    # Smoke test
    logger.info("Running smoke test queries...")
    test_queries = [
        ("What is the punishment for murder?",     "Indian Penal Code"),
        ("How to file an RTI application?",        "Right to Information Act"),
        ("Rights of arrested person bail",         "Code of Criminal Procedure"),
        ("Consumer complaint refund defective",    "Consumer Protection Act"),
        ("महिला के खिलाफ घरेलू हिंसा",            "Domestic Violence Act"),
    ]

    print(f"\n{'='*65}")
    print("SMOKE TEST — Expected section numbers shown")
    print("="*65)
    for query, expected_act in test_queries:
        results = retriever.retrieve(query, top_k=2)
        print(f"\nQ: {query}")
        print(f"   Expected act: {expected_act}")
        for r in results:
            m   = r["metadata"]
            sec = m.get("section_num", "?")
            ttl = m.get("section_title", "")[:40]
            act = m.get("act_name", "?")
            src = r.get("source", "?")
            scr = r.get("cross_encoder_score", r.get("rrf_score", 0))
            hit = "✅" if expected_act.lower() in act.lower() else "❌"
            print(f"   {hit} [{src}] {act} § {sec} — {ttl} (score: {scr:.4f})")
    print("="*65)

    logger.success(f"Done in {time.time()-start:.1f}s")
    return retriever


def show_stats():
    store = LegalVectorStore()
    stats = store.collection_stats()
    print(f"\nCollection: '{CHROMA_COLLECTION}'")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  Unique acts  : {stats['unique_acts']}")
    for act in stats.get("acts", []):
        print(f"    - {act}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",   action="store_true", help="Wipe DB and re-ingest")
    parser.add_argument("--preview", action="store_true", help="Preview chunks without storing")
    parser.add_argument("--act",     type=str, default=None)
    parser.add_argument("--stats",   action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        sys.exit(0)

    if args.preview:
        pages = parse_all_pdfs(PDF_DIR, LEGAL_ACTS)
        nodes = LegalChunker().chunk(pages)
        preview_chunks(nodes, n=5)
        sys.exit(0)

    run_ingestion(act_prefix=args.act, reset=args.reset)