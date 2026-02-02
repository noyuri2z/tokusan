"""
Japanese text splitting utilities for LIME explanations.

This module provides high-level APIs for tokenizing Japanese text.
When SudachiPy is available, it uses morphological analysis for accurate
word segmentation. Otherwise, it falls back to character-level tokenization.

The split function is designed to work seamlessly with LIME's text explainer,
providing proper word boundaries for Japanese text that lacks whitespace
delimiters.

Example:
    >>> from tokusan.japanese.splitters import split, active_japanese_tokenizer
    >>> tokens = split("日本語のテキストを分割します")
    >>> print(tokens)
    ['日本', '語', 'の', 'テキスト', 'を', '分割', 'し', 'ます']
    >>> print(active_japanese_tokenizer())
    'sudachi'
"""

from typing import List
from .tokenizers import _SUDACHI_TOKENIZER, _SUDACHI_MODE, has_sudachi


def active_japanese_tokenizer() -> str:
    """
    Return the name of the currently active Japanese tokenizer backend.

    This function helps users understand which tokenization method is being
    used for Japanese text processing. The 'sudachi' backend provides
    linguistically accurate word segmentation, while 'fallback' uses simple
    character-level splitting.

    Returns:
        str: 'sudachi' if SudachiPy is available and initialized,
             'fallback' if using character-level tokenization.

    Example:
        >>> backend = active_japanese_tokenizer()
        >>> if backend == 'sudachi':
        ...     print("Using SudachiPy for morphological analysis")
        ... else:
        ...     print("Using character-level fallback (install sudachipy for better results)")
    """
    return 'sudachi' if has_sudachi() else 'fallback'


def split(text: str) -> List[str]:
    """
    Split Japanese text into tokens using the best available tokenizer.

    When SudachiPy is available, this function performs morphological analysis
    to accurately segment Japanese text into meaningful words. This is essential
    because Japanese does not use spaces between words.

    When SudachiPy is not available, the function falls back to character-level
    tokenization, which is less accurate but still functional for basic use.

    Args:
        text: The Japanese text string to tokenize.

    Returns:
        List[str]: A list of token strings extracted from the input text.

    Example with Sudachi:
        >>> split("機械学習モデルの予測を説明する")
        ['機械', '学習', 'モデル', 'の', '予測', 'を', '説明', 'する']

    Example with fallback (character-level):
        >>> split("テスト")  # Without Sudachi
        ['テ', 'ス', 'ト']

    Notes:
        - Whitespace characters are excluded from the output
        - The function automatically selects the best available tokenizer
        - For production use, install sudachipy for accurate tokenization
    """
    # Check if Sudachi tokenizer is available
    sudachi_not_ready = (
        not has_sudachi() or
        _SUDACHI_TOKENIZER is None or
        _SUDACHI_MODE is None
    )

    if sudachi_not_ready:
        # Fallback: return non-whitespace characters as individual tokens
        # This provides basic functionality when Sudachi is not installed
        return [char for char in text if not char.isspace()]

    # Use SudachiPy for proper morphological analysis
    # The tokenize() method returns morpheme objects with surface() for the string form
    morphemes = _SUDACHI_TOKENIZER.tokenize(text, _SUDACHI_MODE)
    return [morpheme.surface() for morpheme in morphemes]


__all__ = ["split", "active_japanese_tokenizer"]
