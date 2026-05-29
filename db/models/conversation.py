from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
)

from sqlalchemy.orm import relationship

from datetime import datetime

from db.base import Base


class Conversation(Base):

    __tablename__ = "conversations"

    # ─────────────────────────────────────────
    # Primary Key
    # ─────────────────────────────────────────

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # ─────────────────────────────────────────
    # Firebase User ID
    # ─────────────────────────────────────────

    firebase_uid = Column(
        String,
        nullable=False,
        index=True,
    )

    # ─────────────────────────────────────────
    # Conversation Metadata
    # ─────────────────────────────────────────

    title = Column(
        String,
        default="New Conversation",
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    # ─────────────────────────────────────────
    # Relationship
    # ─────────────────────────────────────────

    messages = relationship(
        "Message",

        back_populates="conversation",

        cascade="all, delete-orphan",
    )