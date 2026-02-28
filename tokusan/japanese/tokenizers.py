"""Japanese tokenizer integration using SudachiPy."""

import importlib.util

_sudachi_available = importlib.util.find_spec("sudachipy") is not None

_SUDACHI_TOKENIZER = None
_SUDACHI_MODE = None

if _sudachi_available:
    from sudachipy import tokenizer as _sudachi_tokenizer  # type: ignore
    from sudachipy import dictionary as _sudachi_dictionary  # type: ignore

    _SUDACHI_TOKENIZER = _sudachi_dictionary.Dictionary().create()

    # SplitMode.A = shortest morphological units
    _SUDACHI_MODE = _sudachi_tokenizer.Tokenizer.SplitMode.A


def has_sudachi() -> bool:
    """Return True if SudachiPy tokenizer is initialized."""
    return _SUDACHI_TOKENIZER is not None


__all__ = ["_SUDACHI_TOKENIZER", "_SUDACHI_MODE", "has_sudachi"]
