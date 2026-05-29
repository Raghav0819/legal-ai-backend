from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from jose import jwt, JWTError

from sqlalchemy.orm import Session

from config import (
    SECRET_KEY,
    JWT_ALGORITHM,
)

from db.session import get_db

# ─────────────────────────────────────────────
# Bearer token scheme
# ─────────────────────────────────────────────

security = HTTPBearer()

# ─────────────────────────────────────────────
# Get current user
# ─────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):

    token = credentials.credentials

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )

        user_id = payload.get("user_id")

        if user_id is None:

            raise HTTPException(
                status_code=401,
                detail="Invalid token",
            )

    except JWTError:

        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="User not found",
        )

    return user