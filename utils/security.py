from datetime import datetime, timedelta

from jose import jwt
from passlib.context import CryptContext

from config import (
    SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_DAYS,
)

# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

# ─────────────────────────────────────────────
# Hash password
# ─────────────────────────────────────────────

def hash_password(password: str) -> str:

    return pwd_context.hash(password)

# ─────────────────────────────────────────────
# Verify password
# ─────────────────────────────────────────────

def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:

    return pwd_context.verify(
        plain_password,
        hashed_password,
    )

# ─────────────────────────────────────────────
# Create JWT token
# ─────────────────────────────────────────────

def create_access_token(data: dict):

    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        days=ACCESS_TOKEN_EXPIRE_DAYS
    )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    return encoded_jwt