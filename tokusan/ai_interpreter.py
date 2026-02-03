"""
AI-powered interpretation using Google Gemini.

This module provides the GeminiInterpreter class for generating deep,
interpretive explanations of text classification results in Japanese.

Example:
    >>> import os
    >>> os.environ['GEMINI_API_KEY'] = 'your_api_key'
    >>>
    >>> from tokusan.ai_interpreter import GeminiInterpreter
    >>> interpreter = GeminiInterpreter()
    >>> interpretation = interpreter.interpret(
    ...     text="ニュース記事のテキスト",
    ...     predicted_class="Fake",
    ...     probabilities={"Real": 0.12, "Fake": 0.88},
    ...     word_weights=[("年", 0.037), ("2020", 0.021)],
    ...     class_names=["Real", "Fake"]
    ... )
    >>> print(interpretation)
"""

import os
from typing import Dict, List, Optional, Tuple

from .exceptions import AIInterpretationError


def _check_gemini_available() -> bool:
    """Check if google-generativeai package is available."""
    try:
        import google.generativeai
        return True
    except ImportError:
        return False


class GeminiInterpreter:
    """
    Generate AI-powered interpretations using Google Gemini.

    This class uses the Gemini API to analyze classification results
    and generate human-readable explanations in Japanese that explain
    WHY certain words influenced the classification.

    Attributes:
        model_name: Name of the Gemini model to use.

    Example:
        >>> interpreter = GeminiInterpreter()
        >>> result = interpreter.interpret(
        ...     text="記事のテキスト",
        ...     predicted_class="Fake",
        ...     probabilities={"Real": 0.1, "Fake": 0.9},
        ...     word_weights=[("年", 0.05)],
        ...     class_names=["Real", "Fake"]
        ... )
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Gemini interpreter.

        Args:
            model_name: Gemini model to use. Default is 'gemini-1.5-flash'
                       which is fast and works well with the free tier.

        Raises:
            AIInterpretationError: If GEMINI_API_KEY is not set or
                                  google-generativeai is not installed.
        """
        # Check if package is available
        if not _check_gemini_available():
            raise AIInterpretationError(
                "google-generativeai package is not installed. "
                "Install it with: pip install google-generativeai"
            )

        # Get API key from environment
        self.api_key = os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise AIInterpretationError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it with your Gemini API key."
            )

        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily initialize the Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def _build_prompt(
        self,
        text: str,
        predicted_class: str,
        probabilities: Dict[str, float],
        word_weights: List[Tuple[str, float]],
        class_names: List[str],
    ) -> str:
        """
        Build the prompt for Gemini.

        Args:
            text: The original input text that was classified.
            predicted_class: The predicted class name.
            probabilities: Dict mapping class names to probabilities.
            word_weights: List of (word, weight) tuples from LIME.
            class_names: List of all class names.

        Returns:
            str: The formatted prompt for Gemini.
        """
        # Format probabilities
        prob_str = ", ".join(
            f"{name}: {prob:.1%}" for name, prob in probabilities.items()
        )

        # Format word weights
        weight_lines = []
        for word, weight in word_weights[:15]:  # Limit to top 15 words
            sign = "+" if weight > 0 else ""
            direction = "支持" if weight > 0 else "反対"
            weight_lines.append(f"- 「{word}」: {sign}{weight:.3f} ({predicted_class}を{direction})")

        weights_str = "\n".join(weight_lines)

        # Truncate text if too long
        max_text_len = 1000
        display_text = text[:max_text_len]
        if len(text) > max_text_len:
            display_text += "...(省略)"

        prompt = f"""あなたは日本語のテキスト分類結果を解釈する専門家です。

以下の分類結果を分析し、なぜこのテキストが「{predicted_class}」と判定されたのかを
日本語で分かりやすく説明してください。

## 入力テキスト
{display_text}

## 分類結果
- 予測クラス: {predicted_class}
- クラス確率: {prob_str}
- 分類対象クラス: {', '.join(class_names)}

## 重要な単語と重み
正の重みはそのクラスへの分類を支持し、負の重みは反対クラスを支持します。
{weights_str}

## 指示
1. まず、予測結果と確信度を1文で簡潔に述べてください
2. 重みの高い単語（正負両方）を分析し、なぜそれらの単語がこの分類に影響したのかを解釈してください
3. 単語の組み合わせや文脈も考慮して、総合的な判定理由を説明してください
4. 専門用語を避け、一般の人にも分かりやすい日本語で説明してください
5. 箇条書きを使用して読みやすくしてください

日本語で回答してください。"""

        return prompt

    def interpret(
        self,
        text: str,
        predicted_class: str,
        probabilities: Dict[str, float],
        word_weights: List[Tuple[str, float]],
        class_names: List[str],
    ) -> str:
        """
        Generate a Japanese interpretation of the classification result.

        This method sends the classification data to Gemini and returns
        a human-readable explanation of why the text was classified
        the way it was.

        Args:
            text: The original input text that was classified.
            predicted_class: The predicted class name (e.g., "Fake").
            probabilities: Dict mapping class names to probabilities.
            word_weights: List of (word, weight) tuples from LIME explanation.
            class_names: List of all class names.

        Returns:
            str: Japanese interpretation explaining the classification.

        Raises:
            AIInterpretationError: If the API call fails.

        Example:
            >>> interpretation = interpreter.interpret(
            ...     text="政府が新政策を発表",
            ...     predicted_class="Real",
            ...     probabilities={"Real": 0.85, "Fake": 0.15},
            ...     word_weights=[("政府", 0.12), ("発表", 0.08)],
            ...     class_names=["Real", "Fake"]
            ... )
        """
        try:
            model = self._get_model()
            prompt = self._build_prompt(
                text=text,
                predicted_class=predicted_class,
                probabilities=probabilities,
                word_weights=word_weights,
                class_names=class_names,
            )

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            raise AIInterpretationError(
                f"Failed to generate AI interpretation: {e}"
            ) from e


def is_ai_available() -> bool:
    """
    Check if AI interpretation is available.

    Returns True if:
    1. google-generativeai package is installed
    2. GEMINI_API_KEY environment variable is set

    Returns:
        bool: True if AI interpretation can be used.

    Example:
        >>> if is_ai_available():
        ...     interpreter = GeminiInterpreter()
        ... else:
        ...     print("AI not available, using template summaries")
    """
    if not _check_gemini_available():
        return False
    return os.environ.get('GEMINI_API_KEY') is not None
