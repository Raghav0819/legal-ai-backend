"""
main.py
Production-Ready FastAPI Backend
Indian Legal Aid Chatbot
"""

import sys

from pathlib import Path

from contextlib import asynccontextmanager

sys.path.insert(
    0,
    str(Path(__file__).parent)
)

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    StreamingResponse,
)

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────

from sqlalchemy.orm import Session

from db.session import get_db

from db.models.conversation import (
    Conversation,
)

from db.models.message import (
    Message,
)

# ─────────────────────────────────────────────
# Pydantic
# ─────────────────────────────────────────────

from pydantic import (
    BaseModel,
    Field,
)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

from loguru import logger

# ─────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────

from auth.firebase_auth import (
    verify_firebase_token,
)

from agents.orchestrator import (
    get_orchestrator,
    LegalAidOrchestrator,
)

from memory.memory_manager import (
    save_message,
)

from utils.llm import (
    stream_legal_response,
)

import config

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

ALLOWED_ORIGINS = getattr(
    config,
    "ALLOWED_ORIGINS",
    ["*"],
)

# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting Legal AI Backend..."
    )

    try:

        get_orchestrator()

        logger.success(
            "Orchestrator loaded."
        )

    except Exception as e:

        logger.exception(e)

        raise e

    yield

    logger.info(
        "Shutting down..."
    )

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(

    title=
        "Indian Legal Aid Chatbot API",

    version="6.0.0",

    lifespan=lifespan,
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────

app.add_middleware(

    CORSMiddleware,

    allow_origins=
        ALLOWED_ORIGINS,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
    )

    conversation_id: int

    act_hint: str | None = None


class CitationOut(BaseModel):

    rank: int

    act: str

    year: int | str

    section: str

    chapter: str

    score: float


class ChatResponse(BaseModel):

    answer: str

    language: str

    intent: str

    citations: list[CitationOut]

    draft: str | None

    error: str | None

    session_id: str

# ─────────────────────────────────────────────
# Dependency
# ─────────────────────────────────────────────

def get_bot() -> LegalAidOrchestrator:

    return get_orchestrator()

# ─────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────

@app.get("/")
def root():

    return {

        "message":
            "Legal AI Backend Running",

        "version":
            "6.0.0",
    }

# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
def health():

    return {

        "status": "ok"
    }

# ─────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────

@app.get("/stats")
def stats(

    bot:
    LegalAidOrchestrator =
        Depends(get_bot)
):

    stats = bot._store.collection_stats()

    return {

        "total_chunks":
            stats.get(
                "total_chunks",
                0,
            ),

        "acts":
            stats.get(
                "acts",
                [],
            ),
    }

# ─────────────────────────────────────────────
# Create Conversation
# ─────────────────────────────────────────────

@app.post("/conversations")
def create_conversation(

    current_user = Depends(
        verify_firebase_token
    ),

    db: Session =
        Depends(get_db),
):

    conversation = Conversation(

        firebase_uid=
            current_user["uid"],

        title="New Chat",
    )

    db.add(conversation)

    db.commit()

    db.refresh(conversation)

    return {

        "conversation_id":
            conversation.id
    }

# ─────────────────────────────────────────────
# Get User Conversations
# ─────────────────────────────────────────────

@app.get("/conversations")
def get_conversations(

    current_user = Depends(
        verify_firebase_token
    ),

    db: Session =
        Depends(get_db),
):

    conversations = (

        db.query(Conversation)

        .filter(

            Conversation.firebase_uid ==
            current_user["uid"]

        )

        .order_by(
            Conversation.created_at.desc()
        )

        .all()
    )

    return [

        {
            "id":
                convo.id,

            "title":
                convo.title,

            "created_at":
                convo.created_at,
        }

        for convo in conversations
    ]

# ─────────────────────────────────────────────
# Get Messages
# ─────────────────────────────────────────────

@app.get(
    "/conversations/{conversation_id}/messages"
)
def get_messages(

    conversation_id: int,

    current_user = Depends(
        verify_firebase_token
    ),

    db: Session =
        Depends(get_db),
):

    conversation = (

        db.query(Conversation)

        .filter(

            Conversation.id ==
            conversation_id,

            Conversation.firebase_uid ==
            current_user["uid"]

        )

        .first()
    )

    if not conversation:

        raise HTTPException(

            status_code=404,

            detail=
                "Conversation not found",
        )

    messages = (

        db.query(Message)

        .filter(

            Message.conversation_id ==
            conversation_id

        )

        .order_by(
            Message.created_at.asc()
        )

        .all()
    )

    return [

        {
            "id":
                msg.id,

            "role":
                msg.role,

            "content":
                msg.content,

            "created_at":
                msg.created_at,
        }

        for msg in messages
    ]

# ─────────────────────────────────────────────
# Standard Chat
# ─────────────────────────────────────────────

@app.post(
    "/chat",

    response_model=
        ChatResponse,
)
def chat(

    req: ChatRequest,

    current_user = Depends(
        verify_firebase_token
    ),

    db: Session =
        Depends(get_db),

    bot:
    LegalAidOrchestrator =
        Depends(get_bot),
):

    conversation = (

        db.query(Conversation)

        .filter(

            Conversation.id ==
            req.conversation_id,

            Conversation.firebase_uid ==
            current_user["uid"]

        )

        .first()
    )

    if not conversation:

        raise HTTPException(

            status_code=404,

            detail=
                "Conversation not found",
        )

    # Save user message

    db.add(

        Message(

            conversation_id=
                req.conversation_id,

            role="user",

            content=req.query,
        )
    )

    db.commit()

    save_message(

        conversation_id=
            str(req.conversation_id),

        role="user",

        content=req.query,
    )

    # Update title if first message

    if conversation.title == "New Chat":

        conversation.title = (
            req.query[:40]
        )

        db.commit()

    # Run orchestrator

    result = bot.run(

        user_query=req.query,

        session_id=
            str(req.conversation_id),

        act_hint=req.act_hint,
    )

    answer = result.get(
        "answer",
        ""
    )

    # Save assistant message

    db.add(

        Message(

            conversation_id=
                req.conversation_id,

            role="assistant",

            content=answer,
        )
    )

    db.commit()

    save_message(

        conversation_id=
            str(req.conversation_id),

        role="assistant",

        content=answer,
    )

    return ChatResponse(

        answer=answer,

        language=result.get(
            "language",
            "English",
        ),

        intent=result.get(
            "intent",
            "explain",
        ),

        citations=[
            CitationOut(**c)

            for c in result.get(
                "citations",
                []
            )
        ],

        draft=result.get("draft"),

        error=result.get("error"),

        session_id=
            str(req.conversation_id),
    )

# ─────────────────────────────────────────────
# Streaming Chat
# ─────────────────────────────────────────────

@app.post("/chat/stream")
def stream_chat(

    req: ChatRequest,

    current_user = Depends(
        verify_firebase_token
    ),

    db: Session =
        Depends(get_db),

    bot:
    LegalAidOrchestrator =
        Depends(get_bot),
):

    conversation = (

        db.query(Conversation)

        .filter(

            Conversation.id ==
            req.conversation_id,

            Conversation.firebase_uid ==
            current_user["uid"]

        )

        .first()
    )

    if not conversation:

        raise HTTPException(

            status_code=404,

            detail=
                "Conversation not found",
        )

    # Save user message

    db.add(

        Message(

            conversation_id=
                req.conversation_id,

            role="user",

            content=req.query,
        )
    )

    db.commit()

    save_message(

        conversation_id=
            str(req.conversation_id),

        role="user",

        content=req.query,
    )

    # Update title

    if conversation.title == "New Chat":

        conversation.title = (
            req.query[:40]
        )

        db.commit()

    # Run retrieval

    result = bot.run(

        user_query=req.query,

        session_id=
            str(req.conversation_id),

        act_hint=req.act_hint,
    )

    answer = result.get(
        "answer",
        ""
    )

    # Stream tokens

    def generate():

        full_response = ""

        for token in answer.split():

            token += " "

            full_response += token

            yield token

        # Save assistant response

        db.add(

            Message(

                conversation_id=
                    req.conversation_id,

                role="assistant",

                content=full_response,
            )
        )

        db.commit()

        save_message(

            conversation_id=
                str(req.conversation_id),

            role="assistant",

            content=full_response,
        )

    return StreamingResponse(

        generate(),

        media_type=
            "text/plain",
    )

# ─────────────────────────────────────────────
# Draft Endpoint
# ─────────────────────────────────────────────

@app.post("/draft")
def draft_document(

    req: ChatRequest,

    current_user = Depends(
        verify_firebase_token
    ),

    db: Session =
        Depends(get_db),

    bot:
    LegalAidOrchestrator =
        Depends(get_bot),
):

    forced_query = (
        f"Draft: {req.query}"
    )

    result = bot.run(

        user_query=
            forced_query,

        session_id=
            str(req.conversation_id),

        act_hint=
            req.act_hint,
    )

    return {

        "draft":
            result.get("draft"),

        "answer":
            result.get("answer"),
    }

# ─────────────────────────────────────────────
# Run Local
# ─────────────────────────────────────────────

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=8000,

        reload=True,
    )

