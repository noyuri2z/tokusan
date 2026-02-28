"""Tokusan: Japanese-friendly LIME explanations for text classification."""

__version__ = "0.1.0"
__author__ = "Noyu Ritsuji"

# Main classifier class
from .classifier import JapaneseTextClassifier

# Result classes
from .results import TrainingResult, PredictionResult, ExplanationResult

# Explainer class
from .explainer import TextExplainer

# Explanation class
from .explanation import Explanation, DomainMapper

# English explanation functions
from .explainer import (
    generate_sentence_for_feature,
    summarize_lime_explanation,
    print_lime_narrative,
)

# Japanese explanation functions
from .explainer import (
    generate_sentence_for_feature_jp,
    summarize_lime_explanation_jp,
    print_lime_narrative_jp,
)

# Exceptions
from .exceptions import TokusanError, TokenizerError, ExplanationError, AIInterpretationError

# AI interpretation (optional, requires GEMINI_API_KEY)
from .ai_interpreter import GeminiInterpreter, is_ai_available

# Japanese tokenizer utilities
from .japanese import splitter as japanese_splitter
from .japanese import active_japanese_tokenizer

__all__ = [
    "__version__",
    "__author__",
    "JapaneseTextClassifier",
    "TrainingResult",
    "PredictionResult",
    "ExplanationResult",
    "TextExplainer",
    "Explanation",
    "DomainMapper",
    "generate_sentence_for_feature",
    "summarize_lime_explanation",
    "print_lime_narrative",
    "generate_sentence_for_feature_jp",
    "summarize_lime_explanation_jp",
    "print_lime_narrative_jp",
    "TokusanError",
    "TokenizerError",
    "ExplanationError",
    "AIInterpretationError",
    "GeminiInterpreter",
    "is_ai_available",
    "japanese_splitter",
    "active_japanese_tokenizer",
]
