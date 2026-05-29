from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)

from sqlalchemy.orm import relationship

from datetime import datetime

from db.base import Base


class Message(Base):

    __tablename__ = "messages"

    # ─────────────────────────────────────────
    # Primary Key
    # ─────────────────────────────────────────

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # ─────────────────────────────────────────
    # Conversation Foreign Key
    # ─────────────────────────────────────────

    conversation_id = Column(
        Integer,

        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),

        nullable=False,
    )

    # ─────────────────────────────────────────
    # Message Role
    # ─────────────────────────────────────────

    role = Column(
        String,
        nullable=False,
    )

    # ─────────────────────────────────────────
    # Message Content
    # ─────────────────────────────────────────

    content = Column(
        Text,
        nullable=False,
    )

    # ─────────────────────────────────────────
    # Timestamp
    # ─────────────────────────────────────────

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    # ─────────────────────────────────────────
    # Relationship
    # ─────────────────────────────────────────

    conversation = relationship(
        "Conversation",

        back_populates="messages",
    )