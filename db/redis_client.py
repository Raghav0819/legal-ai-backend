import os

from upstash_redis import Redis

# ─────────────────────────────────────────────
# Upstash Redis Client
# ─────────────────────────────────────────────

redis_client = Redis(

    url=os.getenv(
        "UPSTASH_REDIS_REST_URL"
    ),

    token=os.getenv(
        "UPSTASH_REDIS_REST_TOKEN"
    ),
)
