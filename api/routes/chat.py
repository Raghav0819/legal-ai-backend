from fastapi import (
    APIRouter,
    Depends,
)

from sqlalchemy.orm import Session

from pydantic import BaseModel

from db.session import get_db

from db.models.conversation import Conversation

from db.models.message import Message

from api.dependencies import get_current_user

router = APIRouter()

# ─────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────

class CreateConversationRequest(BaseModel):

    title: str = "New Chat"

# ─────────────────────────────────────────────
# Create conversation
# ─────────────────────────────────────────────

@router.post("/conversation/create")
def create_conversation(
    req: CreateConversationRequest,

    current_user: User = Depends(get_current_user),

    db: Session = Depends(get_db),
):

    conversation = Conversation(
        user_id=current_user.id,
        title=req.title,
    )

    db.add(conversation)

    db.commit()

    db.refresh(conversation)

    return {
        "success": True,
        "conversation_id": conversation.id,
        "title": conversation.title,
    }

# ─────────────────────────────────────────────
# Get all conversations
# ─────────────────────────────────────────────

@router.get("/conversation/all")
def get_conversations(
    current_user: User = Depends(get_current_user),

    db: Session = Depends(get_db),
):

    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id
        )
        .order_by(Conversation.created_at.desc())
        .all()
    )

    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at,
        }
        for c in conversations
    ]

# ─────────────────────────────────────────────
# Get messages
# ─────────────────────────────────────────────

@router.get("/conversation/{conversation_id}")
def get_messages(
    conversation_id: int,

    current_user: User = Depends(get_current_user),

    db: Session = Depends(get_db),
):

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if not conversation:

        return {
            "success": False,
            "message": "Conversation not found",
        }

    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at.asc())
        .all()
    )

    return {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in messages
        ]
    }