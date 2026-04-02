"""Session and state management for the web application."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tokusan import JapaneseTextClassifier, TrainingResult


@dataclass
class SessionState:
    """Holds the state for a single user session."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    classifier: Optional[JapaneseTextClassifier] = None
    training_result: Optional[TrainingResult] = None
    training_data: Optional[Any] = None  # pandas DataFrame
    class_names: list = field(default_factory=list)
    classifier_type: str = "logistic_regression"
    current_sample: Optional[str] = None
    project_name: Optional[str] = None


class AppState:
    """Global application state with session management."""

    def __init__(self, max_sessions: int = 100, session_ttl: int = 3600):
        """Initialize application state with session limits."""
        self.sessions: Dict[str, SessionState] = {}
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl

    def get_or_create_session(self, session_key: str) -> SessionState:
        """Get existing session or create a new one, keyed by user ID."""
        self._cleanup_expired()

        if session_key in self.sessions:
            # Refresh session timestamp
            self.sessions[session_key].created_at = time.time()
            return self.sessions[session_key]

        # Create new session
        session = SessionState(session_id=session_key)
        self.sessions[session_key] = session
        return session

    def _cleanup_expired(self):
        """Remove expired sessions to prevent memory leaks."""
        now = time.time()
        expired = [
            sid
            for sid, s in self.sessions.items()
            if now - s.created_at > self.session_ttl
        ]
        for sid in expired:
            del self.sessions[sid]

        # Also remove oldest sessions if over limit
        if len(self.sessions) > self.max_sessions:
            sorted_sessions = sorted(
                self.sessions.items(), key=lambda x: x[1].created_at
            )
            for sid, _ in sorted_sessions[: len(self.sessions) - self.max_sessions]:
                del self.sessions[sid]


# Global instance
app_state = AppState()
