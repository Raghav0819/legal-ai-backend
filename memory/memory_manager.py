"""
memory_manager.py

Handles conversational memory using Redis.
"""

import json

from memory.redis_client import redis_client

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MAX_MEMORY_MESSAGES = 10

# ─────────────────────────────────────────────
# Save message
# ─────────────────────────────────────────────

def save_message(
    conversation_id: str,

    role: str,

    content: str,
):
    """
    Save message into Redis memory.
    """

    key = f"chat_memory:{conversation_id}"

    message = {
        "role": role,
        "content": content,
    }

    redis_client.rpush(
        key,
        json.dumps(message),
    )

    # Keep only latest N messages
    redis_client.ltrim(
        key,
        -MAX_MEMORY_MESSAGES,
        -1,
    )

# ─────────────────────────────────────────────
# Fetch memory
# ─────────────────────────────────────────────

def get_memory(
    conversation_id: str,
):
    """
    Fetch recent messages from Redis.
    """

    key = f"chat_memory:{conversation_id}"

    messages = redis_client.lrange(
        key,
        0,
        -1,
    )

    return [
        json.loads(m)
        for m in messages
    ]

# ─────────────────────────────────────────────
# Build memory context
# ─────────────────────────────────────────────

def build_memory_context(
    conversation_id: str,
):
    """
    Convert memory into LLM-friendly text.
    """

    messages = get_memory(conversation_id)

    if not messages:
        return ""

    context = "\n".join(
        [
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
        ]
    )

    return context