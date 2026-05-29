"""
agents/orchestrator.py
"""

import sys
import functools

from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).parent.parent)
)

from langgraph.graph import (
    StateGraph,
    END,
)

from loguru import logger

# ─────────────────────────────────────────────
# Agent nodes
# ─────────────────────────────────────────────

from agents.nodes import (
    AgentState,
    intent_node,
    translate_node,
    retrieve_node,
    reason_node,
    draft_node,
    format_node,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

from config import (
    validate_config,
    COHERE_API_KEY,
)

# ─────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────

class LegalAidOrchestrator:

    def __init__(self):

        validate_config()

        logger.info(
            "Initialising orchestrator..."
        )

        # LAZY LOADED
        self._store = None

        self._retriever = None

        self._graph = None

        logger.success(
            "LegalAidOrchestrator ready."
        )

    # ─────────────────────────────────────────
    # Lazy Load Retriever
    # ─────────────────────────────────────────

    def _load_retriever(self):

        if self._retriever is not None:
            return

        logger.info(
            "Lazy loading retriever..."
        )

        # IMPORT HERE
        # prevents heavy startup RAM usage

        from ingestion.vector_store import (
            LegalVectorStore
        )

        from ingestion.hybrid_retriever import (
            HybridRetriever
        )

        # ─────────────────────────────────────
        # Vector Store
        # ─────────────────────────────────────

        self._store = LegalVectorStore()

        # ─────────────────────────────────────
        # Hybrid Retriever
        # ─────────────────────────────────────

        self._retriever = HybridRetriever(
            self._store,
            cohere_api_key=COHERE_API_KEY,
        )

        # ─────────────────────────────────────
        # Collection Stats
        # ─────────────────────────────────────

        stats = self._store.collection_stats()

        if stats["total_chunks"] == 0:

            raise RuntimeError(
                "ChromaDB is empty. "
                "Run ingest first."
            )

        logger.info(
            f"Building BM25 index over "
            f"{stats['total_chunks']} chunks..."
        )

        # ─────────────────────────────────────
        # Load all chunks
        # ─────────────────────────────────────

        all_data = self._store.collection.get(

            limit=stats["total_chunks"],

            include=[
                "documents",
                "metadatas",
            ],
        )

        # ─────────────────────────────────────
        # Build BM25
        # ─────────────────────────────────────

        self._retriever.build_bm25_index(

            texts=all_data["documents"],

            metadatas=all_data["metadatas"],
        )

        logger.success(
            "Retriever loaded successfully."
        )

    # ─────────────────────────────────────────
    # Lazy Load Graph
    # ─────────────────────────────────────────

    def _load_graph(self):

        if self._graph is not None:
            return

        self._load_retriever()

        logger.info(
            "Building LangGraph..."
        )

        graph = StateGraph(
            AgentState
        )

        # ─────────────────────────────────────
        # Nodes
        # ─────────────────────────────────────

        graph.add_node(
            "intent_node",
            intent_node,
        )

        graph.add_node(
            "translate_node",
            translate_node,
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
            reason_node,
        )

        graph.add_node(
            "draft_node",
            draft_node,
        )

        graph.add_node(
            "format_node",
            format_node,
        )

        # ─────────────────────────────────────
        # Flow
        # ─────────────────────────────────────

        graph.set_entry_point(
            "intent_node"
        )

        graph.add_edge(
            "intent_node",
            "translate_node",
        )

        graph.add_edge(
            "translate_node",
            "retrieve_node",
        )

        graph.add_edge(
            "retrieve_node",
            "reason_node",
        )

        graph.add_conditional_edges(

            "reason_node",

            self._should_draft,

            {
                "draft":
                    "draft_node",

                "format":
                    "format_node",
            },
        )

        graph.add_edge(
            "draft_node",
            "format_node",
        )

        graph.add_edge(
            "format_node",
            END,
        )

        self._graph = graph.compile()

        logger.success(
            "LangGraph ready."
        )

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def run(

        self,

        user_query: str,

        session_id: str = "default",

        act_hint: str | None = None,
    ) -> dict:

        # LAZY LOAD EVERYTHING
        self._load_graph()

        initial_state: AgentState = {

            "user_query":
                user_query,

            "session_id":
                session_id,

            "intent":
                "",

            "act_filter":
                act_hint,

            "english_query":
                "",

            "source_lang":
                "en",

            "language_name":
                "English",

            "chunks":
                [],

            "context":
                "",

            "answer":
                "",

            "draft_document":
                None,

            "final_response":
                {},

            "error":
                None,
        }

        logger.info(
            f"[{session_id}] "
            f"Query: {user_query[:80]}"
        )

        final_state = self._graph.invoke(
            initial_state
        )

        return final_state[
            "final_response"
        ]

    # ─────────────────────────────────────────
    # Conditional Router
    # ─────────────────────────────────────────

    @staticmethod
    def _should_draft(
        state: AgentState
    ) -> str:

        return (

            "draft"

            if state.get("intent")
            == "draft"

            else "format"
        )


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────

_orchestrator_instance = None


def get_orchestrator():

    global _orchestrator_instance

    if _orchestrator_instance is None:

        _orchestrator_instance = (
            LegalAidOrchestrator()
        )

    return _orchestrator_instance
