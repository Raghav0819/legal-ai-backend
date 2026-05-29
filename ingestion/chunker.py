"""
ingestion/chunker.py  (v3)

Consumes ParsedSection objects from pdf_parser.py (v3).
Each section → one TextNode, with correct section_num in metadata.
Very long sections are sub-split on sub-clauses.
Token fallback handles acts with no section structure.
"""

import re
import sys
from pathlib import Path

from llama_index.core.schema import TextNode
from loguru import logger

sys.path.append("..")
from config import CHUNK_SIZE

MIN_CHARS    = 60
MAX_CHARS    = 3000   # sub-split sections longer than this


class LegalChunker:

    def chunk(self, items) -> list[TextNode]:
        """
        items: list of ParsedSection or ParsedPage (fallback)
        Returns list of TextNode ready for embedding.
        """
        # Separate sections from fallback pages
        from ingestion.pdf_parser import ParsedSection, ParsedPage
        sections  = [x for x in items if isinstance(x, ParsedSection)]
        fallbacks = [x for x in items if isinstance(x, ParsedPage)]

        nodes = []

        # ── Section-based chunks (main path) ──────────────────────────────
        # Group by act so we can log per-act counts
        from collections import defaultdict
        act_groups: dict[str, list] = defaultdict(list)
        for sec in sections:
            act_groups[sec.act_name].append(sec)

        for act_name, act_sections in act_groups.items():
            act_nodes = []
            for sec in act_sections:
                act_nodes.extend(self._section_to_nodes(sec))
            logger.info(f"  {act_name}: {len(act_nodes)} section chunks")
            nodes.extend(act_nodes)

        # ── Token fallback (for acts with no section structure) ────────────
        for page in fallbacks:
            fb_nodes = self._token_fallback(page)
            logger.info(f"  {page.act_name}: {len(fb_nodes)} fallback chunks")
            nodes.extend(fb_nodes)

        logger.success(f"Generated {len(nodes)} section-based chunks total")
        return nodes

    # ──────────────────────────────────────────────────────────────────────

    def _section_to_nodes(self, sec) -> list[TextNode]:
        """Convert one ParsedSection → one or more TextNodes."""
        if len(sec.text.strip()) < MIN_CHARS:
            return []

        # Build context prefix (prepended to text for richer embeddings)
        parts = [sec.act_name]
        if sec.chapter:
            parts.append(sec.chapter)
        if sec.section_num:
            parts.append(f"Section {sec.section_num}")
        if sec.section_title:
            parts.append(sec.section_title)
        context_prefix = ", ".join(parts)

        # Sub-split if section is very long
        text_chunks = self._maybe_subsplit(sec.text)

        nodes = []
        for i, chunk_text in enumerate(text_chunks):
            full_text = f"{context_prefix}\n\n{chunk_text}"
            node = TextNode(
                text=full_text,
                metadata={
                    "act_name":      sec.act_name,
                    "year":          sec.year,
                    "chapter":       sec.chapter,
                    "section_num":   sec.section_num,    # ← correct number now
                    "section_title": sec.section_title,
                    "first_section": sec.section_num,    # legacy key
                    "page_num":      sec.page_num,
                    "chunk_index":   i,
                },
                excluded_embed_metadata_keys=["page_num", "chunk_index"],
            )
            nodes.append(node)
        return nodes

    def _maybe_subsplit(self, text: str) -> list[str]:
        """Sub-split very long sections on sub-clause markers."""
        if len(text) <= MAX_CHARS:
            return [text]

        # Try sub-clause split: (a), (b), (i), (ii)
        sub_re = re.compile(r"(?m)^\s*\([a-z]{1,3}\)\s")
        parts  = sub_re.split(text)

        if len(parts) <= 1:
            # Hard-split at paragraph boundaries
            chunks, buf = [], ""
            for line in text.split("\n"):
                if len(buf) + len(line) > MAX_CHARS and buf:
                    chunks.append(buf.strip())
                    buf = ""
                buf += line + "\n"
            if buf.strip():
                chunks.append(buf.strip())
            return chunks

        markers = sub_re.findall(text)
        result  = [parts[0]]
        for marker, part in zip(markers, parts[1:]):
            result.append(marker + part)
        return [r for r in result if len(r.strip()) >= MIN_CHARS]

    def _token_fallback(self, page) -> list[TextNode]:
        """Token-based chunking for acts without section structure."""
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=64)
        doc = Document(
            text=page.text,
            metadata={
                "act_name":      page.act_name,
                "year":          page.year,
                "chapter":       getattr(page, "chapter", ""),
                "section_num":   "",
                "section_title": "",
                "first_section": "",
                "page_num":      page.page_num,
                "chunk_index":   0,
            }
        )
        nodes = splitter.get_nodes_from_documents([doc])
        for n in nodes:
            n.text = f"{page.act_name}:\n\n{n.text}"
        return nodes


def preview_chunks(nodes: list[TextNode], n: int = 5) -> None:
    print(f"\n{'='*65}")
    print(f"CHUNK PREVIEW — {n} of {len(nodes)} total")
    print("="*65)
    for node in nodes[:n]:
        m = node.metadata
        print(f"\nAct:     {m.get('act_name')} ({m.get('year')})")
        print(f"Chapter: {m.get('chapter','')}")
        print(f"Section: {m.get('section_num','')} — {m.get('section_title','')}")
        print(f"Text ({len(node.text)} chars):\n{node.text[:350]}...")
        print("-"*65)