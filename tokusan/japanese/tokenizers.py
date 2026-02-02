"""
Japanese tokenizer integration using SudachiPy.

This module provides tokenization functionality for Japanese text using the
SudachiPy library. SudachiPy is preferred over MeCab because:
- It includes a built-in dictionary (sudachidict_core)
- No external configuration is required
- It has efficient memory usage and high-speed processing

The module initializes a singleton tokenizer instance on import to avoid
repeated initialization overhead during text processing.

Installation:
    pip install sudachipy sudachidict_core

Example:
    >>> from tokusan.japanese.tokenizers import has_sudachi
    >>> if has_sudachi():
    ...     print("SudachiPy is available for Japanese tokenization")
"""

import importlib.util

# Check if SudachiPy is available using importlib
_sudachi_available = importlib.util.find_spec("sudachipy") is not None

# Initialize tokenizer variables
_SUDACHI_TOKENIZER = None
_SUDACHI_MODE = None

if _sudachi_available:
    from sudachipy import tokenizer as _sudachi_tokenizer  # type: ignore
    from sudachipy import dictionary as _sudachi_dictionary  # type: ignore

    # Create a singleton dictionary instance
    # This is initialized once on module import for performance
    _SUDACHI_TOKENIZER = _sudachi_dictionary.Dictionary().create()

    # Use SplitMode.C (most granular splitting) for detailed analysis
    # SplitMode options:
    #   - A: Short unit (similar to unidic-cwj short unit)
    #   - B: Middle unit (default, balanced)
    #   - C: Long unit (named entities kept together)
    _SUDACHI_MODE = _sudachi_tokenizer.Tokenizer.SplitMode.C


def has_sudachi() -> bool:
    """
    Check if SudachiPy tokenizer is successfully initialized.

    Returns:
        bool: True if Sudachi tokenizer is available and ready to use,
              False if SudachiPy is not installed or failed to initialize.

    Example:
        >>> from tokusan.japanese.tokenizers import has_sudachi
        >>> if has_sudachi():
        ...     print("Japanese tokenization with Sudachi is available")
        ... else:
        ...     print("Install sudachipy for full Japanese support")
    """
    return _SUDACHI_TOKENIZER is not None


__all__ = ["_SUDACHI_TOKENIZER", "_SUDACHI_MODE", "has_sudachi"]
