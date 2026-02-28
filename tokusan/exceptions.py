"""Custom exceptions for the tokusan package."""


class TokusanError(Exception):
    """Base exception for all tokusan-related errors."""
    pass


class TokenizerError(TokusanError):
    """Raised when tokenization fails or is misconfigured."""
    pass


class ExplanationError(TokusanError):
    """Raised when explanation generation fails."""
    pass


class AIInterpretationError(TokusanError):
    """Raised when AI-powered interpretation fails."""
    pass
