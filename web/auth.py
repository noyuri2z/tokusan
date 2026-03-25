"""Authentication utilities: password hashing, session cookies, FastAPI dependencies."""

import os
from typing import Optional

import bcrypt
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

# Cookie signing
SECRET_KEY = os.environ.get("TOKUSAN_SECRET_KEY", "tokusan-dev-secret-change-in-production")
COOKIE_NAME = "auth_token"
MAX_AGE = 86400  # 24 hours

_serializer = URLSafeTimedSerializer(SECRET_KEY)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_session_token(user_id: int, username: str) -> str:
    """Create a signed session token."""
    return _serializer.dumps({"user_id": user_id, "username": username})


def verify_session_token(token: str) -> Optional[dict]:
    """Verify and decode a session token. Returns None if invalid or expired."""
    try:
        data = _serializer.loads(token, max_age=MAX_AGE)
        return data
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request: Request) -> Optional[dict]:
    """Extract the current user from the auth cookie. Returns None if not authenticated."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    return verify_session_token(token)


def require_auth(request: Request) -> dict:
    """FastAPI dependency that requires authentication.

    For regular requests, redirects to /login.
    For HTMX requests, returns HX-Redirect header.
    Returns user dict if authenticated.
    """
    user = get_current_user(request)
    if user is not None:
        return user

    # Not authenticated — redirect to login
    if request.headers.get("HX-Request"):
        # HTMX requests need HX-Redirect header
        response = HTMLResponse(status_code=200)
        response.headers["HX-Redirect"] = "/login"
        raise _AuthRedirectException(response)

    raise _AuthRedirectException(RedirectResponse("/login", status_code=302))


class _AuthRedirectException(Exception):
    """Internal exception to carry a redirect response out of a dependency."""

    def __init__(self, response):
        self.response = response
