"""
utils/llm.py

LLM wrapper for:
- Groq
- Gemini

Includes:
- Conversational memory
- Legal grounding
- Multilingual responses
- Draft generation support
"""

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)

from langchain_groq import ChatGroq

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)

from config import (
    LLM_PROVIDER,

    GROQ_API_KEY,
    GEMINI_API_KEY,

    GROQ_MODEL,
    GEMINI_MODEL,

    LLM_TEMPERATURE,
)

from memory.memory_manager import (
    build_memory_context,
)

# ─────────────────────────────────────────────
# Select LLM Provider
# ─────────────────────────────────────────────

if LLM_PROVIDER == "groq":

    llm = ChatGroq(
        api_key=GROQ_API_KEY,

        model_name=GROQ_MODEL,

        temperature=LLM_TEMPERATURE,

        streaming=True,
    )

elif LLM_PROVIDER == "gemini":

    llm = ChatGoogleGenerativeAI(
        google_api_key=GEMINI_API_KEY,

        model=GEMINI_MODEL,

        temperature=LLM_TEMPERATURE,
    )

else:

    raise ValueError(
        f"Unsupported LLM provider: {LLM_PROVIDER}"
    )

# ─────────────────────────────────────────────
# Main legal response generator
# ─────────────────────────────────────────────

def generate_legal_response(
    query: str,

    context: str,

    language: str = "English",

    source_lang: str = "en",

    conversation_id: str | None = None,
):
    """
    Generate grounded legal response.

    Features:
    - Conversational memory
    - Context-aware reasoning
    - Legal grounding
    - Multilingual output
    """

    # ─────────────────────────────────────────
    # Fetch memory context
    # ─────────────────────────────────────────

    memory_context = ""

    if conversation_id:

        memory_context = build_memory_context(
            conversation_id
        )

    # ─────────────────────────────────────────
    # System prompt
    # ─────────────────────────────────────────

    system_prompt = f"""
You are an expert Indian legal aid assistant.

You help Indian citizens understand:
- IPC
- CrPC
- Constitution
- RTI
- Consumer Protection
- Domestic Violence laws
- POCSO
- IBC

IMPORTANT RULES:
1. Use ONLY provided legal context
2. Never hallucinate laws
3. Explain in simple language
4. Mention legal sections when available
5. Be concise but accurate
6. If unsure, say information is insufficient
7. Use conversational memory when relevant
8. Maintain continuity in follow-up questions

Respond in:
{language}
"""

    # ─────────────────────────────────────────
    # User prompt
    # ─────────────────────────────────────────

    user_prompt = f"""
CONVERSATION MEMORY:
{memory_context}

LEGAL CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    # ─────────────────────────────────────────
    # Build messages
    # ─────────────────────────────────────────

    messages = [

        SystemMessage(
            content=system_prompt
        ),

        HumanMessage(
            content=user_prompt
        ),
    ]

    # ─────────────────────────────────────────
    # Invoke LLM
    # ─────────────────────────────────────────

    response = llm.invoke(messages)

    return response.content

# ─────────────────────────────────────────────
# Draft generation
# ─────────────────────────────────────────────

def generate_legal_draft(
    query: str,

    context: str,

    language: str = "English",
):
    """
    Generate legal drafts:
    - RTI
    - Complaints
    - Applications
    - Notices
    """

    system_prompt = f"""
You are an expert Indian legal drafting assistant.

Generate professional legal drafts for Indian citizens.

IMPORTANT:
- Use formal legal language
- Keep formatting clean
- Use placeholders where information is missing
- Make drafts legally structured
- Respond in {language}
"""

    user_prompt = f"""
LEGAL CONTEXT:
{context}

DRAFT REQUEST:
{query}

GENERATE COMPLETE DRAFT:
"""

    messages = [

        SystemMessage(
            content=system_prompt
        ),

        HumanMessage(
            content=user_prompt
        ),
    ]

    response = llm.invoke(messages)

    return response.content

def stream_legal_response(
    query: str,

    context: str,

    language: str = "English",

    conversation_id: str | None = None,
):
    """
    Streams legal response token-by-token.
    """

    memory_context = ""

    if conversation_id:

        memory_context = build_memory_context(
            conversation_id
        )

    system_prompt = f"""
You are an Indian legal assistant.

Use ONLY provided legal context.

Respond in {language}.
"""

    user_prompt = f"""
CONVERSATION MEMORY:
{memory_context}

LEGAL CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    messages = [

        SystemMessage(content=system_prompt),

        HumanMessage(content=user_prompt),
    ]

    for chunk in llm.stream(messages):

        if chunk.content:

            yield chunk.content

# ─────────────────────────────────────────────
# Raw LLM Access
# ─────────────────────────────────────────────

def get_llm():

    """
    Returns raw LLM instance for
    LangGraph draft nodes.
    """

    return llm

