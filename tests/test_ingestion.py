"""
tests/test_ingestion.py

Tests for the Legal RAG ingestion pipeline.
Uses synthetic legal text so tests run without actual PDFs.

Run with:  pytest tests/ -v
"""

import pytest
import tempfile
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow imports from parent
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Synthetic legal text fixtures ─────────────────────────────────────────

SAMPLE_IPC_TEXT = """
CHAPTER XVI

OF OFFENCES AFFECTING THE HUMAN BODY

Section 299. Culpable homicide.—Whoever causes death by doing an act with the intention
of causing death, or with the intention of causing such bodily injury as is likely to
cause death, or with the knowledge that he is likely by such act to cause death, commits
the offence of culpable homicide.

Section 300. Murder.—Except in the cases hereinafter excepted, culpable homicide is
murder, if the act by which the death is caused is done with the intention of causing
death, or if it is done with the intention of causing such bodily injury as the offender
knows to be likely to cause the death of the person to whom the harm is caused.

Section 302. Punishment for murder.—Whoever commits murder shall be punished with death,
or imprisonment for life, and shall also be liable to fine.
"""

SAMPLE_RTI_TEXT = """
Chapter II
RIGHT TO INFORMATION AND OBLIGATIONS OF PUBLIC AUTHORITIES

Section 3. Right to information.—Subject to the provisions of this Act, all citizens
shall have the right to information.

Section 4. Obligations of public authorities.—Every public authority shall—
(a) maintain all its records duly catalogued and indexed;
(b) publish within one hundred and twenty days from the enactment of this Act,
all relevant facts while formulating important policies.

Section 6. Request for obtaining information.—A person, who desires to obtain any
information under this Act, shall make a request in writing or through electronic
means in English or Hindi or in the official language of the area.
"""

SAMPLE_HINDI_QUERY = "हत्या की सजा क्या है"  # "What is the punishment for murder"


# ── Tests ──────────────────────────────────────────────────────────────────

class TestPDFParser:
    """Tests for ingestion/pdf_parser.py"""

    def test_section_detection_ipc(self):
        from ingestion.pdf_parser import LegalPDFParser
        parser = LegalPDFParser()
        section, stype = parser._detect_section(SAMPLE_IPC_TEXT)
        assert section == "299", f"Expected '299', got '{section}'"
        assert stype == "section"

    def test_chapter_detection(self):
        import re
        from ingestion.pdf_parser import CHAPTER_RE
        match = CHAPTER_RE.search(SAMPLE_IPC_TEXT)
        assert match is not None, "Should detect CHAPTER XVI"
        assert "CHAPTER XVI" in match.group(0)

    def test_article_detection_constitution(self):
        from ingestion.pdf_parser import LegalPDFParser
        parser = LegalPDFParser()
        text = "Article 21. Protection of life and personal liberty."
        section, stype = parser._detect_section(text)
        assert section == "21"
        assert stype == "article"

    def test_skips_short_pages(self):
        """Pages with < min_chars should be skipped."""
        from ingestion.pdf_parser import LegalPDFParser
        # We can't test real PDF parsing without a file, but we can
        # test the threshold logic
        parser = LegalPDFParser(min_chars_per_page=100)
        assert parser.min_chars == 100

    def test_missing_pdf_raises(self):
        from ingestion.pdf_parser import LegalPDFParser
        parser = LegalPDFParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.pdf", "Test Act", 2024)


class TestChunker:
    """Tests for ingestion/chunker.py"""

    def _make_pages(self, text: str, act: str, year: int, chapter: str = "Chapter I"):
        """Helper: create fake ParsedPage objects."""
        from ingestion.pdf_parser import ParsedPage
        return [ParsedPage(
            page_num=1,
            text=text,
            act_name=act,
            year=year,
            chapter=chapter,
            section_num="1",
        )]

    def test_chunk_produces_nodes(self):
        from ingestion.chunker import LegalChunker
        pages   = self._make_pages(SAMPLE_IPC_TEXT, "Indian Penal Code", 1860)
        chunker = LegalChunker(chunk_size=128, chunk_overlap=16)
        nodes   = chunker.chunk(pages)
        assert len(nodes) > 0, "Should produce at least one chunk"

    def test_metadata_propagated(self):
        from ingestion.chunker import LegalChunker
        pages  = self._make_pages(SAMPLE_IPC_TEXT, "Indian Penal Code", 1860, "Chapter XVI")
        nodes  = LegalChunker(chunk_size=128, chunk_overlap=16).chunk(pages)
        for node in nodes:
            assert node.metadata["act_name"] == "Indian Penal Code"
            assert node.metadata["year"] == 1860
            assert "chunk_context" in node.metadata

    def test_context_prefix_prepended(self):
        from ingestion.chunker import LegalChunker
        pages = self._make_pages(SAMPLE_IPC_TEXT, "Indian Penal Code", 1860, "Chapter XVI")
        nodes = LegalChunker(chunk_size=512, chunk_overlap=64).chunk(pages)
        # Every chunk should start with the act name
        for node in nodes:
            assert "Indian Penal Code" in node.text, (
                f"Context prefix missing from chunk: {node.text[:100]}"
            )

    def test_two_acts_chunked_separately(self):
        from ingestion.chunker import LegalChunker
        from ingestion.pdf_parser import ParsedPage
        pages = [
            ParsedPage(1, SAMPLE_IPC_TEXT, "Indian Penal Code", 1860, "Chapter XVI", "299"),
            ParsedPage(1, SAMPLE_RTI_TEXT, "RTI Act",           2005, "Chapter II",  "3"),
        ]
        nodes = LegalChunker(chunk_size=128, chunk_overlap=16).chunk(pages)
        act_names = {n.metadata["act_name"] for n in nodes}
        assert "Indian Penal Code" in act_names
        assert "RTI Act" in act_names


class TestVectorStore:
    """Tests for ingestion/vector_store.py — uses a temp ChromaDB directory."""

    @pytest.fixture
    def tmp_store(self, tmp_path, monkeypatch):
        """LegalVectorStore pointing at a temp dir, with a tiny embed model."""
        monkeypatch.setattr("config.CHROMA_DIR", tmp_path)
        monkeypatch.setattr(
            "config.EMBED_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",  # tiny model for tests
        )
        from ingestion.vector_store import LegalVectorStore
        return LegalVectorStore()

    def test_store_initialises(self, tmp_store):
        assert tmp_store.collection is not None
        assert tmp_store.collection_stats()["total_chunks"] == 0

    def test_add_and_query(self, tmp_store):
        from ingestion.chunker import LegalChunker
        from ingestion.pdf_parser import ParsedPage
        pages = [ParsedPage(1, SAMPLE_IPC_TEXT, "Indian Penal Code", 1860, "Chapter XVI", "299")]
        nodes = LegalChunker(chunk_size=256, chunk_overlap=32).chunk(pages)
        added = tmp_store.add_nodes(nodes)
        assert added > 0

        results = tmp_store.query("punishment for murder", top_k=3)
        assert len(results) > 0
        assert "text"     in results[0]
        assert "metadata" in results[0]
        assert "score"    in results[0]

    def test_idempotent_upsert(self, tmp_store):
        """Running ingest twice should not double-count chunks."""
        from ingestion.chunker import LegalChunker
        from ingestion.pdf_parser import ParsedPage
        pages = [ParsedPage(1, SAMPLE_IPC_TEXT, "Indian Penal Code", 1860)]
        nodes = LegalChunker(chunk_size=512, chunk_overlap=64).chunk(pages)

        added_first  = tmp_store.add_nodes(nodes)
        added_second = tmp_store.add_nodes(nodes)  # re-run same data
        assert added_second == 0, "Second run should add 0 new chunks"
        assert tmp_store.collection_stats()["total_chunks"] == added_first

    def test_act_filter(self, tmp_store):
        from ingestion.chunker import LegalChunker
        from ingestion.pdf_parser import ParsedPage
        pages = [
            ParsedPage(1, SAMPLE_IPC_TEXT, "Indian Penal Code", 1860, "Chapter XVI", "299"),
            ParsedPage(2, SAMPLE_RTI_TEXT, "RTI Act",           2005, "Chapter II",  "3"),
        ]
        nodes = LegalChunker(chunk_size=128, chunk_overlap=16).chunk(pages)
        tmp_store.add_nodes(nodes)

        results = tmp_store.query("right to information", top_k=5, act_filter="RTI Act")
        for r in results:
            assert r["metadata"]["act_name"] == "RTI Act", (
                f"act_filter failed: got '{r['metadata']['act_name']}'"
            )


class TestHybridRetriever:
    """Tests for ingestion/hybrid_retriever.py"""

    def test_tokeniser(self):
        from ingestion.hybrid_retriever import HybridRetriever
        tokens = HybridRetriever._tokenise("Section 498A of IPC deals with dowry-harassment")
        assert "498a" in tokens, "Should preserve alphanumeric legal tokens"
        assert "dowry-harassment" in tokens or "dowry" in tokens

    def test_rrf_merge_deduplication(self):
        from ingestion.hybrid_retriever import HybridRetriever
        text = "Section 302 IPC punishment for murder"
        dense  = [{"text": text, "metadata": {}, "score": 0.9}]
        sparse = [{"text": text, "metadata": {}, "score": 5.1}]
        merged = HybridRetriever._rrf_merge(dense, sparse)
        # Same text in both lists → should appear only once
        assert len(merged) == 1
        assert merged[0]["source"] == "hybrid"

    def test_bm25_retrieval(self):
        from ingestion.hybrid_retriever import HybridRetriever
        vs = MagicMock()
        vs.query.return_value = []
        retriever = HybridRetriever(vs)
        texts = [SAMPLE_IPC_TEXT, SAMPLE_RTI_TEXT]
        metas = [{"act_name": "IPC"}, {"act_name": "RTI Act"}]
        retriever.build_bm25_index(texts, metas)

        results = retriever._bm25_retrieve("murder punishment", top_k=2, act_filter=None)
        assert len(results) > 0
        assert any("murder" in r["text"].lower() for r in results)