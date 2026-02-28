"""Japanese text splitting utilities for LIME explanations."""

from typing import List
from .tokenizers import _SUDACHI_TOKENIZER, _SUDACHI_MODE, has_sudachi


def active_japanese_tokenizer() -> str:
    """Return 'sudachi' if SudachiPy is available, otherwise 'fallback'."""
    return 'sudachi' if has_sudachi() else 'fallback'


def split(text: str) -> List[str]:
    """Split Japanese text into tokens using Sudachi or character-level fallback."""
    sudachi_not_ready = (
        not has_sudachi() or
        _SUDACHI_TOKENIZER is None or
        _SUDACHI_MODE is None
    )

    if sudachi_not_ready:
        return [char for char in text if not char.isspace()]

    morphemes = _SUDACHI_TOKENIZER.tokenize(text, _SUDACHI_MODE)
    return [morpheme.surface() for morpheme in morphemes if not morpheme.surface().isspace()]


__all__ = ["split", "active_japanese_tokenizer"]
