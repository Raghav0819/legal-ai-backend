import os
import json

import firebase_admin

from firebase_admin import (
    credentials,
    auth,
)

from fastapi import (
    Depends,
    HTTPException,
)

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

# ─────────────────────────────────────────────
# Firebase Initialization
# ─────────────────────────────────────────────

firebase_json = os.getenv(
    "FIREBASE_CREDENTIALS"
)

if not firebase_json:

    raise ValueError(
        "FIREBASE_CREDENTIALS not found"
    )

firebase_credentials = json.loads(
    firebase_json
)

cred = credentials.Certificate(
    firebase_credentials
)

if not firebase_admin._apps:

    firebase_admin.initialize_app(
        cred
    )

# ─────────────────────────────────────────────
# Bearer Token Scheme
# ─────────────────────────────────────────────

security = HTTPBearer()

# ─────────────────────────────────────────────
# Verify Firebase Token
# ─────────────────────────────────────────────

def verify_firebase_token(

    credentials:
    HTTPAuthorizationCredentials = Depends(
        security
    )
):

    token = credentials.credentials

    try:

        decoded_token = (
            auth.verify_id_token(
                token
            )
        )

        return {

            "uid":
                decoded_token["uid"],

            "email":
                decoded_token.get(
                    "email"
                ),
        }

    except Exception as e:

        raise HTTPException(

            status_code=401,

            detail=
                f"Invalid Firebase token: {str(e)}"
        )
