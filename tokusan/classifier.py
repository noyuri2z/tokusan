"""Japanese text classifier with LIME explanations."""

import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .explainer import (
    TextExplainer,
    summarize_lime_explanation,
    summarize_lime_explanation_jp,
)
from .japanese import splitter as japanese_splitter, JAPANESE_STOPWORDS
from .results import ExplanationResult, PredictionResult, TrainingResult


PUNCT_PATTERN = re.compile(
    r"^[\s\u3000。、！？「」『』（）［］【】.,!?()\"'`~:;<>/\[\]{}|+=\-—–…]+$"
)


def _default_tokenizer(
    text: str,
    stopwords: Optional[set] = None,
    filter_punct: bool = True,
) -> List[str]:
    """Tokenize Japanese text using Sudachi with optional stopword/punctuation filtering."""
    tokens = japanese_splitter(text)

    result = []
    for token in tokens:
        if filter_punct and PUNCT_PATTERN.match(token):
            continue

        # Skip single-character hiragana/katakana
        if len(token) == 1 and unicodedata.name(token, '').startswith(('HIRAGANA', 'KATAKANA')):
            continue

        if stopwords and token in stopwords:
            continue

        result.append(token)

    return result


class JapaneseTextClassifier:
    """End-to-end Japanese text classifier with LIME explanations and AI interpretation."""

    def __init__(
        self,
        class_names: List[str],
        classifier_type: Literal[
            'logistic_regression', 'random_forest', 'linear_svc', 'naive_bayes'
        ] = 'logistic_regression',
        max_features: int = 20000,
        stopwords: Optional[set] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        random_state: Optional[int] = 42,
        **classifier_kwargs,
    ):
        """Initialize with class names, classifier type, and optional configuration."""
        self.class_names = class_names
        self.classifier_type = classifier_type
        self.max_features = max_features
        if stopwords is not None:
            self.stopwords = set(stopwords) | set(JAPANESE_STOPWORDS)
        else:
            self.stopwords = set(JAPANESE_STOPWORDS)
        self.random_state = random_state
        self.classifier_kwargs = classifier_kwargs

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = lambda text: _default_tokenizer(
                text, stopwords=self.stopwords, filter_punct=True
            )

        self._pipeline: Optional[Pipeline] = None
        self._explainer: Optional[TextExplainer] = None
        self.is_trained = False

    def _create_pipeline(self) -> Pipeline:
        """Create the sklearn pipeline with vectorizer and classifier."""
        if self.classifier_type == 'naive_bayes':
            vectorizer = CountVectorizer(
                tokenizer=self._tokenizer,
                token_pattern=None,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
            )
        else:
            vectorizer = TfidfVectorizer(
                tokenizer=self._tokenizer,
                token_pattern=None,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
            )

        if self.classifier_type == 'logistic_regression':
            clf = LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=self.random_state,
                **self.classifier_kwargs,
            )
        elif self.classifier_type == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=self.classifier_kwargs.get('n_estimators', 300),
                max_depth=self.classifier_kwargs.get('max_depth', 50),
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                **{k: v for k, v in self.classifier_kwargs.items()
                   if k not in ['n_estimators', 'max_depth']},
            )
        elif self.classifier_type == 'linear_svc':
            clf = CalibratedClassifierCV(
                LinearSVC(
                    class_weight='balanced',
                    random_state=self.random_state,
                    max_iter=2000,
                    **self.classifier_kwargs,
                )
            )
        elif self.classifier_type == 'naive_bayes':
            clf = MultinomialNB(**self.classifier_kwargs)
        else:
            raise ValueError(
                f"Unknown classifier_type: {self.classifier_type}. "
                "Options: 'logistic_regression', 'random_forest', 'linear_svc', 'naive_bayes'"
            )

        return Pipeline([
            ('tfidf', vectorizer),
            ('clf', clf),
        ])

    def train(
        self,
        texts: Union[List[str], 'pd.Series'],
        labels: Union[List[int], 'np.ndarray', 'pd.Series'],
        test_size: float = 0.2,
    ) -> TrainingResult:
        """Train the classifier and return a TrainingResult with metrics."""
        texts_list = list(texts)
        labels_array = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            texts_list,
            labels_array,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels_array,
        )

        self._pipeline = self._create_pipeline()
        self._pipeline.fit(X_train, y_train)

        y_pred = self._pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
        )

        self._explainer = TextExplainer(
            class_names=self.class_names,
            split_expression=self._tokenizer,
            stopwords=self.stopwords,
            random_state=self.random_state,
        )

        self.is_trained = True

        return TrainingResult(
            accuracy=accuracy,
            classification_report=report,
            train_size=len(X_train),
            test_size=len(X_test),
            class_names=self.class_names,
        )

    def predict(
        self,
        text: str,
        explain: bool = True,
        num_features: int = 10,
        num_samples: int = 500,
        use_ai: Optional[bool] = None,
        fallback_to_template: bool = True,
        theme: Optional[str] = None,
    ) -> PredictionResult:
        """Classify a single text with optional LIME explanation and AI interpretation."""
        if not self.is_trained or self._pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call train() first or load a saved model."
            )

        proba = self._pipeline.predict_proba([text])[0]
        predicted_label = int(np.argmax(proba))
        predicted_class = self.class_names[predicted_label]

        probabilities = {
            name: float(proba[i])
            for i, name in enumerate(self.class_names)
        }

        explanation = None
        if explain and self._explainer is not None:
            explanation = self._generate_explanation(
                text, proba, num_features, num_samples
            )

        return PredictionResult(
            text=text,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            probabilities=probabilities,
            class_names=self.class_names,
            explanation=explanation,
            use_ai=use_ai,
            fallback_to_template=fallback_to_template,
            theme=theme,
        )

    def predict_batch(
        self,
        texts: List[str],
        explain: bool = False,
        num_features: int = 10,
        num_samples: int = 500,
        use_ai: Optional[bool] = None,
        fallback_to_template: bool = True,
    ) -> List[PredictionResult]:
        """Classify multiple texts, optionally generating explanations for each."""
        if not self.is_trained or self._pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call train() first or load a saved model."
            )

        probas = self._pipeline.predict_proba(texts)

        results = []
        for i, text in enumerate(texts):
            proba = probas[i]
            predicted_label = int(np.argmax(proba))
            predicted_class = self.class_names[predicted_label]

            probabilities = {
                name: float(proba[j])
                for j, name in enumerate(self.class_names)
            }

            explanation = None
            if explain and self._explainer is not None:
                explanation = self._generate_explanation(
                    text, proba, num_features, num_samples
                )

            results.append(PredictionResult(
                text=text,
                predicted_class=predicted_class,
                predicted_label=predicted_label,
                probabilities=probabilities,
                class_names=self.class_names,
                explanation=explanation,
                use_ai=use_ai,
                fallback_to_template=fallback_to_template,
            ))

        return results

    def _generate_explanation(
        self,
        text: str,
        proba: np.ndarray,
        num_features: int,
        num_samples: int,
    ) -> ExplanationResult:
        """Generate a LIME explanation for a single prediction."""
        predicted_label = int(np.argmax(proba))

        exp = self._explainer.explain_instance(
            text,
            self._pipeline.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=tuple(range(len(self.class_names))),
        )

        word_weights = exp.as_list(label=predicted_label)

        if self.stopwords:
            word_weights = [
                (word, weight) for word, weight in word_weights
                if word not in self.stopwords and not PUNCT_PATTERN.match(word)
            ]

        sentences_jp = summarize_lime_explanation_jp(
            exp, class_idx=predicted_label, stopwords=self.stopwords
        )
        sentences_en = summarize_lime_explanation(
            exp, class_idx=predicted_label, stopwords=self.stopwords
        )

        probabilities = {
            name: float(proba[i])
            for i, name in enumerate(self.class_names)
        }

        return ExplanationResult(
            word_weights=word_weights,
            class_name=self.class_names[predicted_label],
            class_names=self.class_names,
            probability=float(proba[predicted_label]),
            probabilities=probabilities,
            sentences_jp=sentences_jp,
            sentences_en=sentences_en,
            original_text=text,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to a file using joblib."""
        if not self.is_trained or self._pipeline is None:
            raise RuntimeError("Model has not been trained. Nothing to save.")

        tfidf = self._pipeline.named_steps['tfidf']
        classifier = self._pipeline.named_steps['clf']

        save_data = {
            'tfidf_vocabulary': tfidf.vocabulary_,
            'tfidf_idf': getattr(tfidf, 'idf_', None),
            'classifier': classifier,
            'class_names': self.class_names,
            'classifier_type': self.classifier_type,
            'max_features': self.max_features,
            'stopwords': self.stopwords,
            'random_state': self.random_state,
            'classifier_kwargs': self.classifier_kwargs,
            'version': '1.0',
        }

        joblib.dump(save_data, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'JapaneseTextClassifier':
        """Load a trained model from a file and reconstruct the pipeline."""
        save_data = joblib.load(path)

        instance = cls(
            class_names=save_data['class_names'],
            classifier_type=save_data['classifier_type'],
            max_features=save_data['max_features'],
            stopwords=save_data['stopwords'],
            random_state=save_data['random_state'],
            **save_data['classifier_kwargs'],
        )

        if instance.classifier_type == 'naive_bayes':
            vectorizer = CountVectorizer(
                tokenizer=instance._tokenizer,
                token_pattern=None,
                max_features=instance.max_features,
                vocabulary=save_data['tfidf_vocabulary'],
            )
        else:
            vectorizer = TfidfVectorizer(
                tokenizer=instance._tokenizer,
                token_pattern=None,
                max_features=instance.max_features,
                vocabulary=save_data['tfidf_vocabulary'],
            )
            vectorizer.idf_ = save_data['tfidf_idf']
            vectorizer._tfidf._idf_diag = None

        instance._pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', save_data['classifier']),
        ])

        instance.is_trained = True

        instance._explainer = TextExplainer(
            class_names=instance.class_names,
            split_expression=instance._tokenizer,
            stopwords=instance.stopwords,
            random_state=instance.random_state,
        )

        return instance

    def __repr__(self) -> str:
        """String representation of the classifier."""
        status = "trained" if self.is_trained else "untrained"
        return (
            f"JapaneseTextClassifier("
            f"class_names={self.class_names}, "
            f"classifier_type='{self.classifier_type}', "
            f"status='{status}')"
        )
