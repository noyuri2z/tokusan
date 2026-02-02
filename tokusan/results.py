"""
Result classes for Japanese text classification.

This module provides structured result classes for training and prediction
operations, designed for easy integration with FastAPI and htmx.

Classes:
    TrainingResult: Contains training metrics and summary.
    ExplanationResult: Contains LIME explanation with Japanese/English summaries.
    PredictionResult: Contains prediction with probabilities and optional explanation.

Example:
    >>> from tokusan import JapaneseTextClassifier
    >>> clf = JapaneseTextClassifier(class_names=['Fake', 'Real'])
    >>> result = clf.predict("テスト文章", explain=True)
    >>> print(result.summary_jp)
    >>> print(result.to_dict())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import html


@dataclass
class TrainingResult:
    """
    Result from model training.

    Contains metrics and statistics from the training process,
    with methods for human-readable summaries and serialization.

    Attributes:
        accuracy: Overall accuracy on the test set.
        classification_report: Dict with precision, recall, f1-score per class.
        train_size: Number of samples in training set.
        test_size: Number of samples in test set.
        class_names: List of class names.

    Example:
        >>> result = classifier.train(texts, labels)
        >>> print(result.accuracy)
        0.85
        >>> print(result.summary())
        Training completed successfully...
    """

    accuracy: float
    classification_report: Dict
    train_size: int
    test_size: int
    class_names: List[str]

    def summary(self) -> str:
        """
        Generate a human-readable summary of training results.

        Returns:
            str: Formatted summary with accuracy and per-class metrics.
        """
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

        return "\n".join(lines)

    def summary_jp(self) -> str:
        """
        Generate a Japanese summary of training results.

        Returns:
            str: Formatted summary in Japanese.
        """
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

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """
        Convert to a JSON-serializable dictionary.

        Returns:
            Dict: All training result data.
        """
        return {
            "accuracy": self.accuracy,
            "classification_report": self.classification_report,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "class_names": self.class_names,
        }

    def to_html(self) -> str:
        """
        Generate an HTML fragment for htmx partial updates.

        Returns:
            str: HTML representation of training results.
        """
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
    """
    Structured LIME explanation with summaries.

    Contains word importance weights and pre-generated summaries
    in both Japanese and English.

    Attributes:
        word_weights: List of (word, weight) tuples sorted by |weight|.
        class_name: Name of the explained class.
        class_names: List of all class names.
        probability: Probability of the explained class.
        probabilities: Dict mapping class names to probabilities.
        sentences_jp: Japanese summary sentences.
        sentences_en: English summary sentences.

    Example:
        >>> exp = result.explanation
        >>> print(exp.sentences_jp[0])
        このインスタンスは0.881対0.119でFakeと分類されました...
    """

    word_weights: List[Tuple[str, float]]
    class_name: str
    class_names: List[str]
    probability: float
    probabilities: Dict[str, float]
    sentences_jp: List[str] = field(default_factory=list)
    sentences_en: List[str] = field(default_factory=list)

    @property
    def top_positive_words(self) -> List[Tuple[str, float]]:
        """
        Get words that increase the probability of this class.

        Returns:
            List of (word, weight) tuples with positive weights.
        """
        return [(w, wt) for w, wt in self.word_weights if wt > 0]

    @property
    def top_negative_words(self) -> List[Tuple[str, float]]:
        """
        Get words that decrease the probability of this class.

        Returns:
            List of (word, weight) tuples with negative weights.
        """
        return [(w, wt) for w, wt in self.word_weights if wt < 0]

    @property
    def summary_jp(self) -> str:
        """
        Get the Japanese summary as a single string.

        Returns:
            str: Japanese explanation sentences joined by newlines.
        """
        return "\n".join(self.sentences_jp)

    @property
    def summary_en(self) -> str:
        """
        Get the English summary as a single string.

        Returns:
            str: English explanation sentences joined by newlines.
        """
        return "\n".join(self.sentences_en)

    def to_dict(self) -> Dict:
        """
        Convert to a JSON-serializable dictionary.

        Returns:
            Dict: All explanation data.
        """
        return {
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

    def to_html(self, lang: str = "jp") -> str:
        """
        Generate an HTML fragment for htmx partial updates.

        Args:
            lang: Language for summary ("jp" or "en").

        Returns:
            str: HTML representation of explanation.
        """
        summary = self.summary_jp if lang == "jp" else self.summary_en

        # Build word weight bars
        word_bars = []
        max_weight = max(abs(wt) for _, wt in self.word_weights) if self.word_weights else 1
        for word, weight in self.word_weights[:10]:  # Top 10
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
    """
    Result from a single prediction.

    Contains the prediction, probabilities, and optional LIME explanation
    with methods for summaries and serialization.

    Attributes:
        text: The input text that was classified.
        predicted_class: Name of the predicted class.
        predicted_label: Integer label of the predicted class.
        probabilities: Dict mapping class names to probabilities.
        class_names: List of all class names.
        explanation: Optional ExplanationResult with word importance.

    Example:
        >>> result = classifier.predict("ニュースのテキスト", explain=True)
        >>> print(result.predicted_class)
        'Fake'
        >>> print(result.summary_jp)
        このインスタンスは...
    """

    text: str
    predicted_class: str
    predicted_label: int
    probabilities: Dict[str, float]
    class_names: List[str]
    explanation: Optional[ExplanationResult] = None

    @property
    def confidence(self) -> float:
        """
        Get the confidence (probability) of the predicted class.

        Returns:
            float: Probability of the predicted class.
        """
        return self.probabilities.get(self.predicted_class, 0.0)

    @property
    def summary_jp(self) -> str:
        """
        Get a Japanese summary of the prediction.

        Returns:
            str: Japanese text summarizing prediction and explanation.
        """
        lines = [
            f"予測結果: {self.predicted_class} ({self.confidence:.1%}の確率)",
        ]

        # Add probability breakdown
        prob_parts = [f"{name}: {prob:.1%}" for name, prob in self.probabilities.items()]
        lines.append(f"クラス確率: {', '.join(prob_parts)}")

        # Add explanation if available
        if self.explanation:
            lines.append("")
            lines.append("説明:")
            lines.extend(self.explanation.sentences_jp)

        return "\n".join(lines)

    @property
    def summary_en(self) -> str:
        """
        Get an English summary of the prediction.

        Returns:
            str: English text summarizing prediction and explanation.
        """
        lines = [
            f"Prediction: {self.predicted_class} ({self.confidence:.1%} confidence)",
        ]

        # Add probability breakdown
        prob_parts = [f"{name}: {prob:.1%}" for name, prob in self.probabilities.items()]
        lines.append(f"Class probabilities: {', '.join(prob_parts)}")

        # Add explanation if available
        if self.explanation:
            lines.append("")
            lines.append("Explanation:")
            lines.extend(self.explanation.sentences_en)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """
        Convert to a JSON-serializable dictionary.

        Returns:
            Dict: All prediction data including explanation if present.
        """
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
        """
        Generate an HTML fragment for htmx partial updates.

        Args:
            lang: Language for display ("jp" or "en").

        Returns:
            str: HTML representation of prediction result.
        """
        summary = self.summary_jp if lang == "jp" else self.summary_en

        # Build probability bars
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
