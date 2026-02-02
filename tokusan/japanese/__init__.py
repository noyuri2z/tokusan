"""
Japanese language processing utilities for tokusan.

This subpackage provides Japanese-specific text processing capabilities,
including tokenization using SudachiPy for accurate word segmentation.

Japanese text presents unique challenges for NLP because:
1. Words are not separated by spaces (unlike English)
2. Multiple writing systems are used (hiragana, katakana, kanji)
3. Context-dependent meaning requires morphological analysis

This module addresses these challenges by integrating SudachiPy,
a high-performance Japanese tokenizer with built-in dictionaries.

Usage:
    >>> from tokusan.japanese import splitter, active_japanese_tokenizer
    >>> tokens = splitter("日本語テキストの解析")
    >>> print(tokens)
    ['日本', '語', 'テキスト', 'の', '解析']
    >>> print(active_japanese_tokenizer())
    'sudachi'

Components:
    splitter: Function to split Japanese text into tokens
    active_japanese_tokenizer: Function to check which tokenizer backend is active
"""

from .splitters import split as splitter, active_japanese_tokenizer

__all__ = [
    "splitter",
    "active_japanese_tokenizer",
]
