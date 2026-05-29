"""
main.py
Production FastAPI Backend
"""

import sys

from pathlib import Path

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
)

from memory.memory_manager import (
    save_message,
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
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(

    title=
        "Indian Legal Aid Chatbot API",

    version="7.0.0",
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
# Root
# ─────────────────────────────────────────────

@app.get("/")
def root():

    return {

        "message":
            "Legal AI Backend Running",

        "version":
            "7.0.0",
    }

# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
def health():

    return {

        "status":
            "ok"
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
# Get Conversations
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
):

    # LAZY LOAD BOT ONLY HERE
    bot = get_orchestrator()

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

    # Run AI

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

        # Save AI response

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
