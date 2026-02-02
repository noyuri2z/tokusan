"""
Custom exceptions for the tokusan package.

This module defines exception classes used throughout the tokusan library
to provide clear error messages for various failure conditions.
"""


class TokusanError(Exception):
    """
    Base exception class for all tokusan-related errors.

    This exception serves as the parent class for more specific exceptions
    and can be used to catch any error raised by the tokusan library.

    Example:
        >>> try:
        ...     explainer.explain_instance(text, classifier_fn)
        ... except TokusanError as e:
        ...     print(f"Tokusan error: {e}")
    """
    pass


class TokenizerError(TokusanError):
    """
    Raised when tokenization fails or is misconfigured.

    This exception is raised when:
    - The Japanese tokenizer (Sudachi) is not installed but required
    - Tokenization produces unexpected results
    - Custom tokenizer functions fail
    """
    pass


class ExplanationError(TokusanError):
    """
    Raised when explanation generation fails.

    This exception is raised when:
    - Invalid label index is requested
    - Explanation mode is incorrectly specified
    - Required explanation data is missing
    """
    pass
