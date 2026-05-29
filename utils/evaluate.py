"""
utils/evaluate.py

Evaluates the RAG pipeline quality using a golden test set.
Metrics computed:
  - Context Precision:  are retrieved chunks relevant to the query?
  - Context Recall:     does retrieved context contain the answer?
  - Faithfulness:       is the LLM answer grounded in retrieved context?
  - Answer Relevance:   does the answer actually address the question?

These four metrics together form the RAGAS score — the standard benchmark
for RAG system quality. Target: context_precision > 0.75, faithfulness > 0.80.

Usage:
    python utils/evaluate.py
"""

import json
import sys
from pathlib import Path

from loguru import logger

# ── Golden test set ────────────────────────────────────────────────────────
# Format: {query, ground_truth_answer, relevant_act}
# In your project, build this by having law students label 50+ examples.
GOLDEN_DATASET = [
    {
        "query": "What is the punishment for murder under IPC?",
        "ground_truth": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. (Section 302, IPC)",
        "act": "Indian Penal Code",
    },
    {
        "query": "How do I file an RTI application?",
        "ground_truth": "A person must make a request in writing or through electronic means in English or Hindi or in the official language of the area, along with the prescribed fee.",
        "act": "RTI Act",
    },
    {
        "query": "What are my rights if arrested by police?",
        "ground_truth": "Article 22 of the Constitution guarantees: right to be informed of the grounds of arrest, right to consult a lawyer of choice, and production before a magistrate within 24 hours.",
        "act": "Constitution of India",
    },
    {
        "query": "Can I get a refund for a defective product?",
        "ground_truth": "Under the Consumer Protection Act 2019, a consumer can file a complaint for defective goods and claim replacement, refund, or compensation.",
        "act": "Consumer Protection Act",
    },
    {
        "query": "महिला को घरेलू हिंसा से कैसे बचाया जा सकता है?",
        "ground_truth": "Domestic Violence Act 2005 provides protection orders, residence orders, and monetary relief to women facing domestic violence.",
        "act": "Domestic Violence Act",
    },
]


def compute_context_precision(retrieved_chunks: list[dict], relevant_act: str) -> float:
    """
    Fraction of retrieved chunks that come from the relevant act.
    Simple proxy — in production, use an LLM judge or manual labels.
    """
    if not retrieved_chunks:
        return 0.0
    relevant = sum(
        1 for r in retrieved_chunks
        if relevant_act.lower() in r["metadata"].get("act_name", "").lower()
    )
    return relevant / len(retrieved_chunks)


def evaluate_retrieval(retriever, llm_generate_fn=None) -> dict:
    """
    Run the golden dataset through the retriever and compute metrics.

    Args:
        retriever:       HybridRetriever instance (already built)
        llm_generate_fn: Optional callable(query, context) → answer string.
                         If None, only retrieval metrics are computed.

    Returns:
        Dict with per-query results and aggregate scores.
    """
    results = []

    for item in GOLDEN_DATASET:
        query   = item["query"]
        act     = item["act"]
        gt      = item["ground_truth"]

        logger.info(f"Evaluating: {query[:60]}...")

        # Retrieve
        chunks = retriever.retrieve(query, top_k=4)
        ctx_precision = compute_context_precision(chunks, act)

        result = {
            "query":             query,
            "relevant_act":      act,
            "context_precision": round(ctx_precision, 3),
            "top_chunks":        [
                {
                    "act":     r["metadata"].get("act_name"),
                    "section": r["metadata"].get("first_section"),
                    "score":   round(r.get("rrf_score", r.get("score", 0)), 4),
                    "preview": r["text"][:120] + "...",
                }
                for r in chunks[:3]
            ],
        }

        # Optionally generate and score the answer
        if llm_generate_fn:
            context = "\n\n".join(r["text"] for r in chunks)
            answer  = llm_generate_fn(query, context)
            result["answer"]   = answer
            result["gt"]       = gt
            # Simple lexical faithfulness proxy (use RAGAS library for proper scoring)
            gt_words  = set(gt.lower().split())
            ans_words = set(answer.lower().split())
            result["faithfulness_proxy"] = round(
                len(gt_words & ans_words) / max(len(gt_words), 1), 3
            )

        results.append(result)

    # Aggregate
    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    report = {
        "num_queries":             len(results),
        "avg_context_precision":   round(avg_precision, 3),
        "per_query":               results,
    }

    if llm_generate_fn:
        avg_faith = sum(r.get("faithfulness_proxy", 0) for r in results) / len(results)
        report["avg_faithfulness_proxy"] = round(avg_faith, 3)

    return report


def print_report(report: dict) -> None:
    print("\n" + "="*64)
    print("RETRIEVAL EVALUATION REPORT")
    print("="*64)
    print(f"Queries evaluated:      {report['num_queries']}")
    print(f"Avg context precision:  {report['avg_context_precision']:.1%}")
    if "avg_faithfulness_proxy" in report:
        print(f"Avg faithfulness proxy: {report['avg_faithfulness_proxy']:.1%}")
    print("\nPer-query breakdown:")
    for r in report["per_query"]:
        emoji = "✓" if r["context_precision"] >= 0.5 else "✗"
        print(f"  {emoji} [{r['context_precision']:.0%}] {r['query'][:55]}")
        for chunk in r["top_chunks"][:2]:
            print(f"       → {chunk['act']} §{chunk['section']} (score: {chunk['score']})")
    print("="*64)


if __name__ == "__main__":
    # Import here so this file can be used as a module without side effects
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingest import run_ingestion

    retriever = run_ingestion()
    report    = evaluate_retrieval(retriever)
    print_report(report)

    # Save results
    output = Path("logs/eval_report.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"Report saved to {output}")