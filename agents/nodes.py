"""
agents/nodes.py

Each function here is one NODE in the LangGraph state machine.
The orchestrator (orchestrator.py) connects these nodes with edges.

Flow:
  user_query
      ↓
  [intent_node]      → classifies what the user wants
      ↓
  [translate_node]   → detects language, translates to English
      ↓
  [retrieve_node]    → hybrid search on ChromaDB
      ↓
  [reason_node]      → LLM reads context, generates answer
      ↓
  [draft_node]       → (optional) generates RTI / complaint document
      ↓
  [format_node]      → final response with citations
"""

import sys
from pathlib import Path
from typing import TypedDict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from utils.llm import generate_legal_response
from utils.translator import translate_query_to_english
from config import TOP_K_RERANK, SUPPORTED_LANGUAGES

# ── State schema ───────────────────────────────────────────────────────────
# This dict is passed between every node. Each node reads what it needs
# and writes its output back into the same dict.

class AgentState(TypedDict):
    # Set by caller
    user_query:      str
    session_id:      str

    # Set by intent_node
    intent:          str        # "explain" | "draft" | "locate" | "general"
    act_filter:      Optional[str]  # e.g. "Indian Penal Code"

    # Set by translate_node
    english_query:   str
    source_lang:     str        # "hi", "en", "mr" etc.
    language_name:   str        # "Hindi", "English" etc.

    # Set by retrieve_node
    chunks:          list[dict]
    context:         str        # chunks joined as one string for the LLM

    # Set by reason_node
    answer:          str

    # Set by draft_node (optional)
    draft_document:  Optional[str]

    # Set by format_node
    final_response:  dict       # {answer, citations, draft, language}

    # Error handling
    error:           Optional[str]


# ── INTENT DETECTION NODE ──────────────────────────────────────────────────

INTENT_EXAMPLES = """
User: "What is the punishment for theft?"          → intent: explain
User: "How do I file an RTI application?"          → intent: explain
User: "Write me an RTI application for road works" → intent: draft
User: "Draft a consumer complaint against Amazon"  → intent: draft
User: "Where is the nearest district court?"       → intent: locate
User: "Hello, how are you?"                        → intent: general
User: "चोरी की सजा क्या है?"                      → intent: explain
User: "मुझे RTI आवेदन लिखना है"                  → intent: draft
"""

def intent_node(state: AgentState) -> AgentState:
    """
    Classifies user intent into one of 4 categories.
    Also tries to detect which Act is relevant (for ChromaDB filtering).

    Uses keyword matching first (fast, free), falls back to LLM if unclear.
    """
    query = state["user_query"].lower()

    # ── Fast keyword-based intent detection ───────────────────────────────
    draft_keywords   = ["draft", "write", "लिखना", "लिखें", "application",
                        "complaint", "notice", "आवेदन", "शिकायत"]
    locate_keywords  = ["where", "nearest", "court", "address", "location",
                        "कहाँ", "नजदीक"]
    general_keywords = ["hello", "hi", "thanks", "who are you", "नमस्ते"]

    if any(k in query for k in general_keywords):
        intent = "general"
    elif any(k in query for k in draft_keywords):
        intent = "draft"
    elif any(k in query for k in locate_keywords):
        intent = "locate"
    else:
        intent = "explain"    # default: explain a law

    # ── Act filter detection (for ChromaDB metadata filtering) ────────────
    act_filter = None
    act_keywords = {
        "Indian Penal Code":          ["ipc", "murder", "theft", "assault", "rape",
                                       "section 302", "section 420", "हत्या", "चोरी"],
        "Right to Information Act":   ["rti", "right to information", "सूचना का अधिकार"],
        "Consumer Protection Act":    ["consumer", "refund", "product", "complaint",
                                       "उपभोक्ता"],
        "Constitution of India":      ["fundamental rights", "article 21", "constitution",
                                       "मौलिक अधिकार", "संविधान"],
        "Domestic Violence Act":      ["domestic violence", "घरेलू हिंसा", "wife",
                                       "husband", "marital"],
        "POCSO Act":                  ["pocso", "child", "minor", "sexual offence"],
        "Code of Criminal Procedure": ["crpc", "arrest", "bail", "fir", "गिरफ्तारी",
                                       "जमानत"],
    }
    for act, keywords in act_keywords.items():
        if any(k in query for k in keywords):
            act_filter = act
            break

    logger.info(f"Intent: {intent} | Act filter: {act_filter}")
    return {**state, "intent": intent, "act_filter": act_filter}


# ── TRANSLATION NODE ───────────────────────────────────────────────────────

def translate_node(state: AgentState) -> AgentState:
    """
    Detects the language of the user query and translates to English
    for retrieval. Stores original language so the LLM responds correctly.

    THIS IS WHERE your translator code lives.
    """
    user_query = state["user_query"]

    # ── Your translator code — exactly here ───────────────────────────────
    english_query, source_lang = translate_query_to_english(user_query)
    # ─────────────────────────────────────────────────────────────────────

    language_name = SUPPORTED_LANGUAGES.get(source_lang, "English")

    logger.info(f"Language: {source_lang} ({language_name})")
    logger.info(f"Original:   {user_query}")
    logger.info(f"Translated: {english_query}")

    return {
        **state,
        "english_query": english_query,
        "source_lang":   source_lang,
        "language_name": language_name,
    }


# ── RETRIEVAL NODE ─────────────────────────────────────────────────────────

def retrieve_node(state: AgentState, retriever) -> AgentState:
    """
    Runs hybrid retrieval (BM25 + ChromaDB) using the English query.
    Applies act_filter if intent_node detected a specific act.

    THIS IS WHERE your retriever.retrieve() call lives.
    """
    # ── Your retriever code — exactly here ────────────────────────────────
    chunks = retriever.retrieve(
        query=state["english_query"],
        top_k=TOP_K_RERANK,
        act_filter=state.get("act_filter"),   # None = search all acts
    )
    # ─────────────────────────────────────────────────────────────────────

    if not chunks:
        logger.warning("No chunks retrieved — query may be out of scope")
        return {
            **state,
            "chunks":  [],
            "context": "",
            "error":   "No relevant legal information found for this query.",
        }

    # Join chunks into one context string for the LLM
    context_parts = []
    for i, chunk in enumerate(chunks):
        act     = chunk["metadata"].get("act_name", "Unknown Act")
        section = chunk["metadata"].get("first_section", "")
        sec_ref = f"Section {section}" if section else ""
        header  = f"[Source {i+1}: {act} {sec_ref}]".strip()
        context_parts.append(f"{header}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    logger.info(f"Retrieved {len(chunks)} chunks from: "
                f"{set(c['metadata'].get('act_name','?') for c in chunks)}")

    return {**state, "chunks": chunks, "context": context}


# ── REASONING NODE ─────────────────────────────────────────────────────────

def reason_node(state: AgentState) -> AgentState:
    """
    Sends the retrieved context + user query to the LLM.
    LLM generates a cited, language-appropriate legal explanation.

    THIS IS WHERE your system_prompt code lives.
    """
    if state.get("error") or not state.get("context"):
        # No context retrieved — give a graceful fallback
        fallback = {
            "hi": "मुझे इस प्रश्न के लिए प्रासंगिक कानूनी जानकारी नहीं मिली। कृपया किसी योग्य वकील से परामर्श करें।",
            "en": "I couldn't find relevant legal information for this query. Please consult a qualified lawyer.",
        }
        answer = fallback.get(state.get("source_lang", "en"), fallback["en"])
        return {**state, "answer": answer}

    # ── Your system_prompt code — exactly here ────────────────────────────
    answer = generate_legal_response(
        query=state["user_query"],          # original (Hindi or English)
        context=state["context"],
        source_lang=state.get("source_lang", "en"),
        conversation_id=state["session_id"],
    )
    # ─────────────────────────────────────────────────────────────────────

    logger.info(f"Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}


# ── DRAFT NODE (optional) ──────────────────────────────────────────────────

DRAFT_TEMPLATES = {
    "rti": """
RTI APPLICATION

To,
The Public Information Officer,
[Department Name],
[Office Address]

Subject: Application under Right to Information Act, 2005

Sir/Madam,

I, [Applicant Name], resident of [Address], hereby request the following
information under Section 6 of the Right to Information Act, 2005:

Information Requested:
{information_requested}

I am ready to pay the prescribed fee. Please provide the information within
30 days as per Section 7(1) of the RTI Act.

Yours faithfully,
[Name]
[Address]
[Date]
    """,

    "consumer_complaint": """
CONSUMER COMPLAINT

To,
The District Consumer Disputes Redressal Commission,
[District Name]

Complainant: [Your Name], [Address], [Phone]
Opposite Party: {company_name}, [Their Address]

COMPLAINT UNDER CONSUMER PROTECTION ACT, 2019

Facts:
{facts}

Relief Sought:
1. Refund of Rs. {amount}/-
2. Compensation for mental harassment
3. Cost of litigation

Declaration: I declare that the facts stated are true to the best of my knowledge.

Date: [Date]
Signature: [Name]
    """,
}

def draft_node(state: AgentState) -> AgentState:
    """
    Generates a legal document draft if intent == 'draft'.
    Skipped entirely for 'explain', 'locate', 'general' intents.
    """
    if state.get("intent") != "draft":
        return {**state, "draft_document": None}

    # Ask the LLM to generate the document using the context
    from utils.llm import get_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm()
    draft_prompt = f"""Based on the user's request and the legal context provided,
generate a professional legal document draft.

User request: {state['user_query']}

Legal context:
{state['context']}

Generate a complete, ready-to-use document with [PLACEHOLDER] for information
the user needs to fill in. Use formal legal language appropriate for Indian courts."""

    response = llm.invoke([HumanMessage(content=draft_prompt)])
    draft = response.content

    logger.info("Draft document generated")
    return {**state, "draft_document": draft}



# ── FORMAT NODE ────────────────────────────────────────────────────────────

def format_node(state: AgentState) -> AgentState:
    """
    Assembles the final response dict that the FastAPI endpoint returns.
    Extracts rich citation metadata for frontend rendering.
    """

    citations = []

    for i, chunk in enumerate(
        state.get("chunks", [])
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        # ─────────────────────────────────────
        # Score
        # ─────────────────────────────────────

        raw_score = (

            chunk.get(
                "rrf_score"
            )

            or chunk.get(
                "score"
            )

            or 0.85
        )

        try:

            score = round(
                float(raw_score),
                4
            )

        except Exception:

            score = 0.85

        # ─────────────────────────────────────
        # Citation object
        # ─────────────────────────────────────

        citation = {

            "rank":
                i + 1,

            "act":
                metadata.get(
                    "act_name",
                    "Unknown Act"
                ),

            "year":
                metadata.get(
                    "year",
                    ""
                ),

            "chapter":
                metadata.get(
                    "chapter",
                    ""
                ),

            "section":
                metadata.get(
                    "first_section",
                    ""
                ),

            "page_range":
                f"{metadata.get('page_start','')}"
                f"–"
                f"{metadata.get('page_end','')}",

            # 🔥 frontend confidence bar
            "score":
                score,

            # 🔥 expandable preview
            "preview":
                chunk.get(
                    "text",
                    ""
                )[:500],

            # 🔥 future support
            "source_id":
                metadata.get(
                    "source_id",
                    f"src_{i+1}"
                ),
        }

        citations.append(
            citation
        )

    # ─────────────────────────────────────────
    # Final response
    # ─────────────────────────────────────────

    final_response = {

        "answer":
            state.get(
                "answer",
                ""
            ),

        "language":
            state.get(
                "language_name",
                "English",
            ),

        "source_lang":
            state.get(
                "source_lang",
                "en",
            ),

        "intent":
            state.get(
                "intent",
                "explain",
            ),

        "citations":
            citations,

        "draft":
            state.get(
                "draft_document"
            ),

        "error":
            state.get(
                "error"
            ),

        "session_id":
            state.get(
                "session_id"
            ),

        "context":
            state.get(
                "context"
            ),
    }

    return {

        **state,

        "final_response":
            final_response
    }
