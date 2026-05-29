"""
config.py
Central configuration for the Indian Legal Aid Chatbot.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────────────────────

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Base paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdfs"

CHROMA_DIR = DATA_DIR / "chroma_db"

LOG_DIR = BASE_DIR / "logs"

# Create directories automatically
for d in [DATA_DIR, PDF_DIR, CHROMA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Debug
# ─────────────────────────────────────────────────────────────

DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000"
).split(",")

# ─────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/legalaid_db"
)

# ─────────────────────────────────────────────────────────────
# Redis
# ─────────────────────────────────────────────────────────────

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)

# ─────────────────────────────────────────────────────────────
# JWT / Security
# ─────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "super_secret_key"
)

JWT_ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_DAYS = 7

# ─────────────────────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────────────────────

EMBED_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBED_DIMENSION = 384

# ─────────────────────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────────────────────

CHROMA_COLLECTION = "indian_legal_corpus"

CHROMA_HOST = os.getenv(
    "CHROMA_HOST",
    "localhost"
)

CHROMA_PORT = int(
    os.getenv("CHROMA_PORT", 8000)
)

# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────

CHUNK_SIZE = 512

CHUNK_OVERLAP = 64

# ─────────────────────────────────────────────────────────────
# Legal Acts
# ─────────────────────────────────────────────────────────────

LEGAL_ACTS = [
    ("ipc",               "Indian Penal Code",              1860),

    ("crpc",              "Code of Criminal Procedure",     1973),

    ("constitution",      "Constitution of India",          1950),

    ("rti",               "Right to Information Act",       2005),

    ("consumer",          "Consumer Protection Act",        2019),

    ("pocso",             "POCSO Act",                      2012),

    ("domestic_violence", "Domestic Violence Act",          2005),

    ("ibc",               "Insolvency and Bankruptcy Code", 2016),
]

# ─────────────────────────────────────────────────────────────
# Terms that should never be translated
# ─────────────────────────────────────────────────────────────

PRESERVE_TERMS = {
    "FIR",
    "IPC",
    "CrPC",
    "RTI",
    "PIL",
    "HC",
    "SC",
    "SLP",
    "Writ",
    "Habeas Corpus",
    "Mandamus",
    "Section",
    "Article",
    "Schedule",
    "POCSO",
    "IBC",
    "NCLT",
    "NCLAT",
}

# ─────────────────────────────────────────────────────────────
# Retrieval settings
# ─────────────────────────────────────────────────────────────

TOP_K_RETRIEVAL = 8

TOP_K_RERANK = 4

BM25_WEIGHT = 0.3

VECTOR_WEIGHT = 0.7

# ─────────────────────────────────────────────────────────────
# LLM Provider
# ─────────────────────────────────────────────────────────────

LLM_PROVIDER = "groq"

# ─────────────────────────────────────────────────────────────
# Groq models
# ─────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────
# Gemini models
# ─────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-1.5-pro"

# ─────────────────────────────────────────────────────────────
# Generation settings
# ─────────────────────────────────────────────────────────────

LLM_TEMPERATURE = 0.1

LLM_MAX_TOKENS = 1024

# ─────────────────────────────────────────────────────────────
# Supported languages
# ─────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
}

DEFAULT_LANGUAGE = "en"

# ─────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

COHERE_API_KEY = os.getenv(
    "COHERE_API_KEY",
    ""
)

# ─────────────────────────────────────────────────────────────
# Twilio
# ─────────────────────────────────────────────────────────────

TWILIO_ACCOUNT_SID = os.getenv(
    "TWILIO_ACCOUNT_SID",
    ""
)

TWILIO_AUTH_TOKEN = os.getenv(
    "TWILIO_AUTH_TOKEN",
    ""
)

TWILIO_WHATSAPP_NUMBER = os.getenv(
    "TWILIO_WHATSAPP_NUMBER",
    ""
)

# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

def validate_config() -> None:
    """
    Validate required environment variables.
    """

    # ─────────────────────────────────────────
    # Validate LLM provider
    # ─────────────────────────────────────────

    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:

        raise EnvironmentError(
            "GROQ_API_KEY missing in .env"
        )

    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:

        raise EnvironmentError(
            "GEMINI_API_KEY missing in .env"
        )

    # ─────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────

    if COHERE_API_KEY:
        print("✓ Cohere reranker enabled")

    else:
        print(
            "ℹ Cohere API key not set — using default ranking"
        )

    if DEBUG:
        print("✓ DEBUG mode enabled")

    print(f"✓ Using LLM Provider: {LLM_PROVIDER}")

    print(f"✓ Chroma Collection: {CHROMA_COLLECTION}")

    print(f"✓ Database: {DATABASE_URL}")