"""Result classes for training, explanation, and prediction output."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import html
import os

if TYPE_CHECKING:
    from .ai_interpreter import GeminiInterpreter


@dataclass
class TrainingResult:
    """Training metrics including accuracy and per-class scores."""

    accuracy: float
    classification_report: Dict
    train_size: int
    test_size: int
    class_names: List[str]

    def summary(self) -> str:
        """Generate a human-readable English summary of training results."""
        lines = [
            f"Training completed successfully.",
            f"",
            f"Dataset:",
            f"  - Training samples: {self.train_size:,}",
            f"  - Test samples: {self.test_size:,}",
            f"",
            f"Performance:",
            f"  - Accuracy: {self.accuracy:.2%}",
            f"",
            f"Per-class metrics:",
        ]

        for class_name in self.class_names:
            if class_name in self.classification_report:
                metrics = self.classification_report[class_name]
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                lines.append(
                    f"  - {class_name}: precision={precision:.2f}, "
                    f"recall={recall:.2f}, f1={f1:.2f}"
                )

        if self.accuracy < 0.5:
            lines.append("")
            lines.append("正解率が50%以下です。モデルを変えるかデータセットに問題がないかを確認してください")

        return "\n".join(lines)

    def summary_jp(self) -> str:
        """Generate a Japanese summary of training results."""
        lines = [
            f"学習が完了しました。",
            f"",
            f"データセット:",
            f"  - 学習データ数: {self.train_size:,}",
            f"  - テストデータ数: {self.test_size:,}",
            f"",
            f"性能:",
            f"  - 正確度: {self.accuracy:.2%}",
            f"",
            f"クラス別メトリクス:",
        ]

        for class_name in self.class_names:
            if class_name in self.classification_report:
                metrics = self.classification_report[class_name]
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1-score', 0)
                lines.append(
                    f"  - {class_name}: 適合率={precision:.2f}, "
                    f"再現率={recall:.2f}, F1={f1:.2f}"
                )

        if self.accuracy < 0.5:
            lines.append("")
            lines.append("正解率が50%以下です。モデルを変えるかデータセットに問題がないかを確認してください")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "accuracy": self.accuracy,
            "classification_report": self.classification_report,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "class_names": self.class_names,
        }

    def to_html(self) -> str:
        """Generate an HTML fragment for htmx partial updates."""
        rows = []
        for class_name in self.class_names:
            if class_name in self.classification_report:
                metrics = self.classification_report[class_name]
                rows.append(f"""
                    <tr>
                        <td>{html.escape(class_name)}</td>
                        <td>{metrics.get('precision', 0):.2f}</td>
                        <td>{metrics.get('recall', 0):.2f}</td>
                        <td>{metrics.get('f1-score', 0):.2f}</td>
                    </tr>
                """)

        return f"""
        <div class="training-result">
            <h3>Training Results</h3>
            <p><strong>Accuracy:</strong> {self.accuracy:.2%}</p>
            <p><strong>Training samples:</strong> {self.train_size:,}</p>
            <p><strong>Test samples:</strong> {self.test_size:,}</p>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """


@dataclass
class ExplanationResult:
    """LIME explanation with word weights and bilingual summaries."""

    word_weights: List[Tuple[str, float]]
    class_name: str
    class_names: List[str]
    probability: float
    probabilities: Dict[str, float]
    sentences_jp: List[str] = field(default_factory=list)
    sentences_en: List[str] = field(default_factory=list)
    original_text: str = ""

    @property
    def top_positive_words(self) -> List[Tuple[str, float]]:
        """Words that increase the probability of this class."""
        return [(w, wt) for w, wt in self.word_weights if wt > 0]

    @property
    def top_negative_words(self) -> List[Tuple[str, float]]:
        """Words that decrease the probability of this class."""
        return [(w, wt) for w, wt in self.word_weights if wt < 0]

    @property
    def summary_jp(self) -> str:
        """Japanese explanation sentences joined by newlines."""
        return "\n".join(self.sentences_jp)

    @property
    def summary_en(self) -> str:
        """English explanation sentences joined by newlines."""
        return "\n".join(self.sentences_en)

    def get_ai_interpretation(self, fallback_to_template: bool = False) -> str:
        """Get AI-powered interpretation using Gemini, with optional template fallback."""
        from .ai_interpreter import GeminiInterpreter, is_ai_available
        from .exceptions import AIInterpretationError

        if not is_ai_available():
            if fallback_to_template:
                return self.summary_jp
            raise AIInterpretationError(
                "AI interpretation is not available. "
                "Set GEMINI_API_KEY and install google-genai."
            )

        interpreter = GeminiInterpreter()
        return interpreter.interpret(
            text=self.original_text,
            predicted_class=self.class_name,
            probabilities=self.probabilities,
            word_weights=self.word_weights,
            class_names=self.class_names,
        )

    def to_dict(self) -> Dict:
        """Convert to a JSON-serializable dictionary."""
        result = {
            "word_weights": [
                {"word": w, "weight": wt} for w, wt in self.word_weights
            ],
            "class_name": self.class_name,
            "class_names": self.class_names,
            "probability": self.probability,
            "probabilities": self.probabilities,
            "summary_jp": self.summary_jp,
            "summary_en": self.summary_en,
            "top_positive_words": [
                {"word": w, "weight": wt} for w, wt in self.top_positive_words
            ],
            "top_negative_words": [
                {"word": w, "weight": wt} for w, wt in self.top_negative_words
            ],
        }
        if self.original_text:
            result["original_text"] = self.original_text
        return result

    def to_html(self, lang: str = "jp") -> str:
        """Generate an HTML fragment for htmx partial updates."""
        summary = self.summary_jp if lang == "jp" else self.summary_en

        word_bars = []
        max_weight = max(abs(wt) for _, wt in self.word_weights) if self.word_weights else 1
        for word, weight in self.word_weights[:10]:
            normalized = abs(weight) / max_weight * 100
            color = "green" if weight > 0 else "red"
            word_bars.append(f"""
                <div class="word-bar">
                    <span class="word">{html.escape(word)}</span>
                    <div class="bar" style="width: {normalized}%; background-color: {color};"></div>
                    <span class="weight">{weight:+.3f}</span>
                </div>
            """)

        return f"""
        <div class="explanation-result">
            <h4>Explanation</h4>
            <div class="summary">
                <p>{html.escape(summary)}</p>
            </div>
            <div class="word-weights">
                {''.join(word_bars)}
            </div>
        </div>
        """


@dataclass
class PredictionResult:
    """Prediction output with probabilities and optional LIME explanation."""

    text: str
    predicted_class: str
    predicted_label: int
    probabilities: Dict[str, float]
    class_names: List[str]
    explanation: Optional[ExplanationResult] = None
    use_ai: Optional[bool] = None
    fallback_to_template: bool = True
    theme: Optional[str] = None

    @property
    def confidence(self) -> float:
        """Probability of the predicted class."""
        return self.probabilities.get(self.predicted_class, 0.0)

    @property
    def summary_jp(self) -> str:
        """Japanese summary, using AI interpretation when available."""
        should_use_ai = self.use_ai
        if should_use_ai is None:
            should_use_ai = bool(os.environ.get('GEMINI_API_KEY'))

        if should_use_ai and self.explanation:
            return self._generate_ai_summary_jp()

        return self._template_summary_jp()

    def _template_summary_jp(self) -> str:
        """Generate template-based Japanese summary."""
        lines = [
            f"予測結果: {self.predicted_class} ({self.confidence:.1%}の確率)",
        ]

        prob_parts = [f"{name}: {prob:.1%}" for name, prob in self.probabilities.items()]
        lines.append(f"クラス確率: {', '.join(prob_parts)}")

        if self.explanation:
            lines.append("")
            lines.append("説明:")
            lines.extend(self.explanation.sentences_jp)

        return "\n".join(lines)

    def _generate_ai_summary_jp(self) -> str:
        """Generate AI-powered Japanese summary using Gemini, with optional template fallback."""
        from .ai_interpreter import GeminiInterpreter, is_ai_available
        from .exceptions import AIInterpretationError

        if not is_ai_available():
            if self.fallback_to_template:
                return self._template_summary_jp()
            raise AIInterpretationError(
                "AI interpretation is not available. "
                "Set GEMINI_API_KEY and install google-genai."
            )

        interpreter = GeminiInterpreter()
        return interpreter.interpret(
            text=self.text,
            predicted_class=self.predicted_class,
            probabilities=self.probabilities,
            word_weights=self.explanation.word_weights if self.explanation else [],
            class_names=self.class_names,
            theme=self.theme,
        )

    @property
    def summary_en(self) -> str:
        """English summary of the prediction and explanation."""
        lines = [
            f"Prediction: {self.predicted_class} ({self.confidence:.1%} confidence)",
        ]

        prob_parts = [f"{name}: {prob:.1%}" for name, prob in self.probabilities.items()]
        lines.append(f"Class probabilities: {', '.join(prob_parts)}")

        if self.explanation:
            lines.append("")
            lines.append("Explanation:")
            lines.extend(self.explanation.sentences_en)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to a JSON-serializable dictionary."""
        result = {
            "text": self.text,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "probabilities": self.probabilities,
            "class_names": self.class_names,
            "confidence": self.confidence,
            "summary_jp": self.summary_jp,
            "summary_en": self.summary_en,
        }

        if self.explanation:
            result["explanation"] = self.explanation.to_dict()

        return result

    def to_html(self, lang: str = "jp") -> str:
        """Generate an HTML fragment for htmx partial updates."""
        prob_bars = []
        for class_name, prob in self.probabilities.items():
            is_predicted = class_name == self.predicted_class
            bar_class = "predicted" if is_predicted else ""
            prob_bars.append(f"""
                <div class="prob-bar {bar_class}">
                    <span class="class-name">{html.escape(class_name)}</span>
                    <div class="bar" style="width: {prob * 100}%;"></div>
                    <span class="prob-value">{prob:.1%}</span>
                </div>
            """)

        explanation_html = ""
        if self.explanation:
            explanation_html = self.explanation.to_html(lang=lang)

        return f"""
        <div class="prediction-result">
            <h3>Prediction Result</h3>
            <div class="prediction-header">
                <span class="predicted-class">{html.escape(self.predicted_class)}</span>
                <span class="confidence">{self.confidence:.1%}</span>
            </div>
            <div class="probabilities">
                {''.join(prob_bars)}
            </div>
            {explanation_html}
        </div>
        """
