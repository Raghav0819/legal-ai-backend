import os

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
# Firebase Initialization (Production Safe)
# ─────────────────────────────────────────────

firebase_credentials = {

    "type":
        "service_account",

    "project_id":
        os.getenv(
            "FIREBASE_PROJECT_ID"
        ),

    "private_key":
    (
        os.getenv(
            "FIREBASE_PRIVATE_KEY"
        ).replace(
            "\\n",
            "\n"
        )
        ),

    "client_email":
        os.getenv(
            "FIREBASE_CLIENT_EMAIL"
        ),

    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),

    "client_id": os.getenv("FIREBASE_CLIENT_ID"),

    "auth_uri":
        os.getenv("FIREBASE_AUTH_URI"),

    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),

    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),

    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),

    "universe_domain":        os.getenv(
            "FIREBASE_UNIVERSE_DOMAIN"
        ),
}

cred = credentials.Certificate(
    firebase_credentials
)

# Prevent duplicate initialization
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
