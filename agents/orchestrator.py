"""
agents/orchestrator.py

Builds and runs the LangGraph state machine that connects all agent nodes.

Graph Flow:

START
  ↓
intent_node
  ↓
translate_node
  ↓
retrieve_node
  ↓
reason_node
  ├── draft → draft_node
  └── normal → format_node
              ↓
            END
"""

import sys
import functools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from loguru import logger

# ─────────────────────────────────────────────────────────────
# Agent nodes
# ─────────────────────────────────────────────────────────────

from agents.nodes import (
    AgentState,
    intent_node,
    translate_node,
    retrieve_node,
    reason_node,
    draft_node,
    format_node,
)

# ─────────────────────────────────────────────────────────────
# Retrieval system
# ─────────────────────────────────────────────────────────────

from ingestion.vector_store import LegalVectorStore
from ingestion.hybrid_retriever import HybridRetriever

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

from config import (
    validate_config,
    COHERE_API_KEY,
)

# ─────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────


class LegalAidOrchestrator:
    """
    Main entry point for the legal aid chatbot.

    Responsibilities:
    - Load vector DB
    - Build BM25 index
    - Load reranker
    - Build LangGraph pipeline
    - Execute queries end-to-end
    """

    def __init__(self):

        # Validate config
        validate_config()

        logger.info("Initialising retriever...")

        # ─────────────────────────────────────────────
        # Vector store
        # ─────────────────────────────────────────────

        self._store = LegalVectorStore()

        # ─────────────────────────────────────────────
        # Hybrid retriever
        # ─────────────────────────────────────────────

        self._retriever = HybridRetriever(
            self._store,
            cohere_api_key=COHERE_API_KEY,
        )

        # ─────────────────────────────────────────────
        # Load corpus stats
        # ─────────────────────────────────────────────

        stats = self._store.collection_stats()

        if stats["total_chunks"] == 0:
            raise RuntimeError(
                "ChromaDB is empty. Run `python ingest.py --reset` first."
            )

        logger.info(
            f"Loading BM25 index over {stats['total_chunks']} chunks..."
        )

        # ─────────────────────────────────────────────
        # Load all chunks
        # ─────────────────────────────────────────────

        all_data = self._store.collection.get(
            limit=stats["total_chunks"],
            include=["documents", "metadatas"],
        )

        # ─────────────────────────────────────────────
        # Build BM25
        # ─────────────────────────────────────────────

        self._retriever.build_bm25_index(
            texts=all_data["documents"],
            metadatas=all_data["metadatas"],
        )

        # ─────────────────────────────────────────────
        # Build LangGraph
        # ─────────────────────────────────────────────

        self._graph = self._build_graph()

        logger.success("LegalAidOrchestrator ready.")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def run(
        self,
        user_query: str,
        session_id: str = "default",
        act_hint: str | None = None,
    ) -> dict:
        """
        Process a user query end-to-end.
        """

        initial_state: AgentState = {
            "user_query": user_query,
            "session_id": session_id,

            "intent": "",

            "act_filter": act_hint,

            "english_query": "",

            "source_lang": "en",

            "language_name": "English",

            "chunks": [],

            "context": "",

            "answer": "",

            "draft_document": None,

            "final_response": {},

            "error": None,
        }

        logger.info(
            f"[{session_id}] Query: {user_query[:80]}"
        )

        final_state = self._graph.invoke(initial_state)

        return final_state["final_response"]

    # ─────────────────────────────────────────────────────────
    # LangGraph builder
    # ─────────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.
        """

        graph = StateGraph(AgentState)

        # ─────────────────────────────────────────────
        # Add nodes
        # IMPORTANT:
        # Node names MUST NOT clash with state keys.
        # ─────────────────────────────────────────────

        graph.add_node(
            "intent_node",
            intent_node
        )

        graph.add_node(
            "translate_node",
            translate_node
        )

        graph.add_node(
            "retrieve_node",
            functools.partial(
                retrieve_node,
                retriever=self._retriever,
            ),
        )

        graph.add_node(
            "reason_node",
            reason_node
        )

        graph.add_node(
            "draft_node",
            draft_node
        )

        graph.add_node(
            "format_node",
            format_node
        )

        # ─────────────────────────────────────────────
        # Entry point
        # ─────────────────────────────────────────────

        graph.set_entry_point("intent_node")

        # ─────────────────────────────────────────────
        # Linear flow
        # ─────────────────────────────────────────────

        graph.add_edge(
            "intent_node",
            "translate_node"
        )

        graph.add_edge(
            "translate_node",
            "retrieve_node"
        )

        graph.add_edge(
            "retrieve_node",
            "reason_node"
        )

        # ─────────────────────────────────────────────
        # Conditional routing
        # ─────────────────────────────────────────────

        graph.add_conditional_edges(
            "reason_node",
            self._should_draft,
            {
                "draft": "draft_node",
                "format": "format_node",
            },
        )

        # ─────────────────────────────────────────────
        # Final flow
        # ─────────────────────────────────────────────

        graph.add_edge(
            "draft_node",
            "format_node"
        )

        graph.add_edge(
            "format_node",
            END
        )

        return graph.compile()

    # ─────────────────────────────────────────────────────────
    # Conditional router
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _should_draft(state: AgentState) -> str:
        """
        Route to draft_node only if:
            intent == "draft"
        """

        return (
            "draft"
            if state.get("intent") == "draft"
            else "format"
        )


# ─────────────────────────────────────────────────────────────
# Singleton instance for FastAPI
# ─────────────────────────────────────────────────────────────

_orchestrator_instance: LegalAidOrchestrator | None = None


def get_orchestrator() -> LegalAidOrchestrator:
    """
    Returns singleton orchestrator instance.
    """

    global _orchestrator_instance

    if _orchestrator_instance is None:
        _orchestrator_instance = LegalAidOrchestrator()

    return _orchestrator_instance