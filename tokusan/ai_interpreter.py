"""AI-powered interpretation using Google Gemini."""

import importlib.util
import os
from typing import Dict, List, Optional, Tuple

from .exceptions import AIInterpretationError


def _check_gemini_available() -> bool:
    """Check if google-genai package is available."""
    return importlib.util.find_spec("google.genai") is not None


class GeminiInterpreter:
    """Generate AI-powered interpretations of classification results using Google Gemini."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize the Gemini interpreter with the given model name."""
        if not _check_gemini_available():
            raise AIInterpretationError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )

        self.api_key = os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise AIInterpretationError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_prompt(
        self,
        text: str,
        predicted_class: str,
        probabilities: Dict[str, float],
        word_weights: List[Tuple[str, float]],
        class_names: List[str],
    ) -> str:
        """Build the prompt for Gemini."""
        prob_str = ", ".join(
            f"{name}: {prob:.1%}" for name, prob in probabilities.items()
        )

        weight_lines = []
        for word, weight in word_weights[:15]:
            sign = "+" if weight > 0 else ""
            direction = "支持" if weight > 0 else "反対"
            weight_lines.append(f"- 「{word}」: {sign}{weight:.3f} ({predicted_class}を{direction})")

        weights_str = "\n".join(weight_lines)

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
2. 重みの高い単語（正負両方）を分析し、なぜそれらの単語がこの分類に影響したのかを解釈してください。データセットの特徴なども踏まえて説明してください。
3. 単語の組み合わせや文脈も考慮して、総合的な判定理由を説明してください。
4. 専門用語を避け、一般の人にも分かりやすい日本語で説明してください
5. 段落分けで読みやすくしてください（マークダウン記号や*、#、-などの記号は使用しないでください）

日本語で回答してください。
"""

        return prompt

    def interpret(
        self,
        text: str,
        predicted_class: str,
        probabilities: Dict[str, float],
        word_weights: List[Tuple[str, float]],
        class_names: List[str],
    ) -> str:
        """Generate a Japanese interpretation of the classification result."""
        client = self._get_client()
        prompt = self._build_prompt(
            text=text,
            predicted_class=predicted_class,
            probabilities=probabilities,
            word_weights=word_weights,
            class_names=class_names,
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text


def is_ai_available() -> bool:
    """Return True if google-genai is installed and GEMINI_API_KEY is set."""
    if not _check_gemini_available():
        return False
    return os.environ.get('GEMINI_API_KEY') is not None
