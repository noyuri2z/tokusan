"""
Tokusan: Japanese-friendly LIME explanations for text classification.

Tokusan extends LIME (Local Interpretable Model-Agnostic Explanations)
with robust support for Japanese text processing and plain language
explanation generation in both English and Japanese.

Key Features:
    - End-to-end Japanese text classification with JapaneseTextClassifier
    - Automatic Japanese tokenization using SudachiPy
    - Plain language explanations in English and Japanese
    - Model-agnostic: works with any text classifier
    - FastAPI/htmx compatible result classes
    - Model persistence (save/load trained models)

Quick Start (End-to-End Classifier):
    >>> from tokusan import JapaneseTextClassifier
    >>>
    >>> # Create and train classifier
    >>> clf = JapaneseTextClassifier(class_names=['Fake', 'Real'])
    >>> result = clf.train(texts, labels)
    >>> print(result.summary())
    >>>
    >>> # Predict with explanation
    >>> pred = clf.predict("ニュースのテキスト", explain=True)
    >>> print(pred.summary_jp)  # Japanese explanation
    >>> print(pred.to_dict())   # JSON-serializable for API
    >>>
    >>> # Save and load model
    >>> clf.save('model.pkl')
    >>> clf = JapaneseTextClassifier.load('model.pkl')

Quick Start (Custom Model):
    >>> from tokusan import TextExplainer
    >>>
    >>> # Create explainer for Japanese text
    >>> explainer = TextExplainer(
    ...     class_names=['フェイク', '本物'],
    ...     lang='jp'
    ... )
    >>>
    >>> # Explain a prediction
    >>> exp = explainer.explain_instance(
    ...     text="このニュースは信頼できる内容です",
    ...     classifier_fn=model.predict_proba
    ... )
    >>>
    >>> # Get word importance as list
    >>> print(exp.as_list(label=1))
    >>>
    >>> # Get Japanese plain language summary
    >>> from tokusan import print_lime_narrative_jp
    >>> print_lime_narrative_jp(exp)

Installation Requirements:
    For Japanese text processing, install SudachiPy:
        pip install sudachipy sudachidict_core

Classes:
    JapaneseTextClassifier: End-to-end classifier with training and explanation.
    TextExplainer: Low-level class for explaining any text classifier.
    Explanation: Container for explanation results.
    TrainingResult: Result from model training.
    PredictionResult: Result from prediction with optional explanation.
    ExplanationResult: Structured LIME explanation with summaries.

Functions:
    generate_sentence_for_feature: Generate English sentence for one word.
    generate_sentence_for_feature_jp: Generate Japanese sentence for one word.
    summarize_lime_explanation: Create English summary of explanation.
    summarize_lime_explanation_jp: Create Japanese summary of explanation.
    print_lime_narrative: Print formatted English explanation.
    print_lime_narrative_jp: Print formatted Japanese explanation.
"""

__version__ = "0.1.0"
__author__ = "Noyu Ritsuji"

# Main classifier class (end-to-end)
from .classifier import JapaneseTextClassifier

# Result classes for structured output
from .results import TrainingResult, PredictionResult, ExplanationResult

# Explainer class (for custom models)
from .explainer import TextExplainer

# Explanation class for type hints and direct use
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

# Japanese tokenizer utilities (for advanced users)
from .japanese import splitter as japanese_splitter
from .japanese import active_japanese_tokenizer

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Main classifier (end-to-end)
    "JapaneseTextClassifier",
    # Result classes
    "TrainingResult",
    "PredictionResult",
    "ExplanationResult",
    # Explainer classes (for custom models)
    "TextExplainer",
    "Explanation",
    "DomainMapper",
    # English explanation functions
    "generate_sentence_for_feature",
    "summarize_lime_explanation",
    "print_lime_narrative",
    # Japanese explanation functions
    "generate_sentence_for_feature_jp",
    "summarize_lime_explanation_jp",
    "print_lime_narrative_jp",
    # Exceptions
    "TokusanError",
    "TokenizerError",
    "ExplanationError",
    "AIInterpretationError",
    # AI interpretation
    "GeminiInterpreter",
    "is_ai_available",
    # Japanese utilities
    "japanese_splitter",
    "active_japanese_tokenizer",
]
