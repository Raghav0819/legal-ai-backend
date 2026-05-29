"""
ingestion/pdf_parser.py  (v3)

Complete rewrite — parses at SECTION level, not page level.

Instead of returning one ParsedPage per PDF page, this returns one
ParsedSection per legal section. Section numbers are extracted at
parse time with multiple regex patterns covering all 8 Indian legal PDFs.

This fixes the root cause of "§ ?" — section numbers are now extracted
from the actual section heading line, not inferred later by the chunker.
"""

import re
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import fitz  # PyMuPDF
from loguru import logger


# ── Section heading patterns (ordered: most specific first) ───────────────
#
# Indian legal PDFs have 3 common formats:
#
# FORMAT A (IPC, CrPC):  "302. Punishment for murder.—"
#                         bare number + period + title + em-dash
#
# FORMAT B (RTI, Consumer, POCSO, DV):
#                         "Section 3. Right to information.—"
#                         explicit "Section" keyword
#
# FORMAT C (Constitution): "21. Protection of life and personal liberty."
#                           number + period + title (no em-dash)
#
# FORMAT D (IBC):          "7. Initiation of corporate insolvency..."
#                           same as A but with longer titles

SECTION_PATTERNS = [
    # Format B — "Section 3. Title" or "Section 3A. Title"
    re.compile(
        r"(?m)^[Ss]ection\s+(\d{1,4}[A-Z]?)\s*[.\-–]\s*(.{3,120}?)(?:\s*[\.—–]|$)",
    ),
    # Format A/D — "302. Punishment for murder.—" at line start
    re.compile(
        r"(?m)^(\d{1,4}[A-Z]?)\.\s+([A-Z][a-zA-Z ,\-']{3,100}?)(?:\s*\.—|\s*—|\.?\s*$)",
    ),
    # Format C — Constitution articles "21. Protection of life"
    re.compile(
        r"(?m)^(\d{1,4}[A-Z]?)\.\s+([A-Z][a-zA-Z ,\-'()]{5,120}?)\.\s*$",
    ),
]

CHAPTER_RE = re.compile(
    r"(?m)^(?:CHAPTER|Chapter)\s+([IVXLCDM\d]+[A-Z]?)[\s\-—]*([^\n]{0,80})?$"
)

# Pages before this are typically TOC / preamble — skip them
SKIP_PAGES_THRESHOLD = 3
# Minimum characters for a section body to be kept
MIN_SECTION_CHARS = 60


@dataclass
class ParsedSection:
    """One legal section extracted from a PDF."""
    section_num:   str          # "302", "21A", "6"
    section_title: str          # "Punishment for murder"
    text:          str          # full section text including heading
    act_name:      str
    year:          int
    chapter:       str          # "CHAPTER XVI"
    page_num:      int          # page where section starts


class LegalPDFParser:
    """
    Parses Indian legal PDFs directly into ParsedSection objects.
    Section numbers and titles are extracted from actual heading lines.
    """

    def parse(
        self,
        pdf_path: str | Path,
        act_name: str,
        year: int,
        skip_pages: int = SKIP_PAGES_THRESHOLD,
    ) -> list[ParsedSection]:

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc         = fitz.open(str(pdf_path))
        full_text   = ""
        page_map    = []   # list of (char_offset, page_num)

        for i, page in enumerate(doc):
            if i < skip_pages:
                continue
            t = page.get_text("text")
            page_map.append((len(full_text), i + 1))
            full_text += t + "\n"

        doc.close()

        sections = self._split_into_sections(full_text, page_map, act_name, year)

        if sections:
            logger.success(f"Parsed {len(sections)} sections from {pdf_path.name}")
        else:
            logger.warning(
                f"Parsed 0 sections from {pdf_path.name} — "
                f"will fall back to token chunking"
            )
        return sections

    # ──────────────────────────────────────────────────────────────────────

    def _split_into_sections(
        self,
        text: str,
        page_map: list[tuple],
        act_name: str,
        year: int,
    ) -> list[ParsedSection]:
        """Find all section boundaries and split text there."""

        # Try each pattern; use the one with the most matches
        best_matches = []
        best_pattern = None
        for pat in SECTION_PATTERNS:
            matches = list(pat.finditer(text))
            if len(matches) > len(best_matches):
                best_matches = matches
                best_pattern = pat

        if len(best_matches) < 3:
            return []   # not enough structure found — caller will token-fallback

        sections      = []
        current_chap  = ""

        for i, match in enumerate(best_matches):
            start = match.start()
            end   = best_matches[i + 1].start() if i + 1 < len(best_matches) else len(text)

            body = text[start:end].strip()
            if len(body) < MIN_SECTION_CHARS:
                continue

            # Extract section number and title from regex groups
            g         = match.groups()
            sec_num   = g[0].strip() if g[0] else ""
            sec_title = g[1].strip() if len(g) > 1 and g[1] else ""

            # Clean title — remove trailing punctuation / em-dashes
            sec_title = re.sub(r'[\s\.—–\-]+$', '', sec_title).strip()
            # Remove leading "—" that sometimes attaches
            sec_title = sec_title.lstrip('—–').strip()

            # Look for chapter heading in the 500 chars before this section
            pre = text[max(0, start - 500): start]
            chap_m = CHAPTER_RE.search(pre)
            if chap_m:
                chap_num   = chap_m.group(1)
                chap_title = (chap_m.group(2) or "").strip()
                current_chap = f"CHAPTER {chap_num}" + (f" — {chap_title}" if chap_title else "")

            page_num = self._find_page(start, page_map)

            sections.append(ParsedSection(
                section_num=sec_num,
                section_title=sec_title,
                text=body,
                act_name=act_name,
                year=year,
                chapter=current_chap,
                page_num=page_num,
            ))

        return sections

    @staticmethod
    def _find_page(char_offset: int, page_map: list[tuple]) -> int:
        """Binary search: find which page a character offset falls on."""
        page_num = 1
        for offset, pg in page_map:
            if offset <= char_offset:
                page_num = pg
            else:
                break
        return page_num


# ── ParsedPage shim — keeps chunker.py compatible ─────────────────────────
# chunker.py expects objects with .act_name, .year, .text, .page_num
# ParsedSection already has all these fields, so no shim needed.

@dataclass
class ParsedPage:
    """Legacy shim — only used by token fallback in chunker.py."""
    page_num:      int
    text:          str
    act_name:      str
    year:          int
    chapter:       str = ""
    section_num:   str = ""
    section_title: str = ""


def parse_all_pdfs(
    pdf_dir: Path,
    legal_acts: list[tuple],
) -> list[ParsedSection]:
    """
    Parse all PDFs. Returns ParsedSection objects (one per legal section).
    Falls back to page-level ParsedPage for acts with no section structure.
    """
    parser    = LegalPDFParser()
    all_items = []

    for prefix, act_name, year in legal_acts:
        matches = list(pdf_dir.glob(f"{prefix}*.pdf"))
        if not matches:
            logger.warning(f"No PDF found for '{act_name}' (prefix='{prefix}'). Skipping.")
            continue

        for pdf_path in matches:
            sections = parser.parse(pdf_path, act_name=act_name, year=year)
            if sections:
                all_items.extend(sections)
            else:
                # Fallback: return full text as one ParsedPage
                # chunker.py token_fallback will handle it
                doc  = fitz.open(str(pdf_path))
                text = "\n".join(p.get_text("text") for p in doc)
                doc.close()
                all_items.append(ParsedPage(
                    page_num=1,
                    text=text,
                    act_name=act_name,
                    year=year,
                ))
                logger.warning(
                    f"{act_name}: no sections found in PDF — "
                    f"falling back to token chunking"
                )

    logger.info(f"Total sections parsed across all acts: {len(all_items)}")
    return all_items