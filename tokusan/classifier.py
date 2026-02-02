"""
Japanese text classifier with LIME explanations.

This module provides the JapaneseTextClassifier class for end-to-end
text classification with built-in support for Japanese tokenization
and LIME-based explanations.

Example:
    >>> from tokusan import JapaneseTextClassifier
    >>>
    >>> # Create and train
    >>> clf = JapaneseTextClassifier(class_names=['Fake', 'Real'])
    >>> result = clf.train(texts, labels)
    >>> print(result.summary())
    >>>
    >>> # Predict with explanation
    >>> pred = clf.predict("ニュースのテキスト", explain=True)
    >>> print(pred.summary_jp)
    >>>
    >>> # Save and load
    >>> clf.save('model.pkl')
    >>> clf2 = JapaneseTextClassifier.load('model.pkl')
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .explainer import (
    TextExplainer,
    summarize_lime_explanation,
    summarize_lime_explanation_jp,
)
from .japanese import splitter as japanese_splitter
from .results import ExplanationResult, PredictionResult, TrainingResult


# Default punctuation pattern for filtering
PUNCT_PATTERN = re.compile(
    r"^[\s\u3000。、！？「」『』（）［］【】.,!?()\"'`~:;<>/\[\]{}|+=\-—–…]+$"
)


def _default_tokenizer(
    text: str,
    stopwords: Optional[set] = None,
    filter_punct: bool = True,
) -> List[str]:
    """
    Default Japanese tokenizer using Sudachi.

    Args:
        text: Text to tokenize.
        stopwords: Set of stopwords to filter out.
        filter_punct: Whether to filter punctuation tokens.

    Returns:
        List of tokens.
    """
    tokens = japanese_splitter(text)

    result = []
    for token in tokens:
        # Filter punctuation
        if filter_punct and PUNCT_PATTERN.match(token):
            continue

        # Filter stopwords
        if stopwords and token in stopwords:
            continue

        result.append(token)

    return result


class JapaneseTextClassifier:
    """
    End-to-end Japanese text classifier with LIME explanations.

    This class provides a complete pipeline for training text classification
    models on Japanese text, making predictions, and generating human-readable
    explanations using LIME.

    Attributes:
        class_names: List of class names (e.g., ['Fake', 'Real']).
        classifier_type: Type of classifier ('logistic_regression' or 'random_forest').
        is_trained: Whether the model has been trained.

    Example:
        >>> # Create classifier
        >>> clf = JapaneseTextClassifier(
        ...     class_names=['Fake', 'Real'],
        ...     classifier_type='logistic_regression'
        ... )
        >>>
        >>> # Train on data
        >>> import pandas as pd
        >>> df = pd.read_csv('fakenews.csv')
        >>> result = clf.train(df['context'], df['label'])
        >>> print(f"Accuracy: {result.accuracy:.2%}")
        >>>
        >>> # Predict with explanation
        >>> pred = clf.predict("ニュース記事のテキスト", explain=True)
        >>> print(pred.summary_jp)
        >>>
        >>> # Save model
        >>> clf.save('model.pkl')
    """

    def __init__(
        self,
        class_names: List[str],
        classifier_type: Literal['logistic_regression', 'random_forest'] = 'logistic_regression',
        max_features: int = 20000,
        stopwords: Optional[set] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        random_state: Optional[int] = 42,
        **classifier_kwargs,
    ):
        """
        Initialize the Japanese text classifier.

        Args:
            class_names: List of class names corresponding to label indices.
                        e.g., ['Fake', 'Real'] where 0='Fake', 1='Real'.
            classifier_type: Type of classifier to use.
                           Options: 'logistic_regression', 'random_forest'.
            max_features: Maximum number of features for TF-IDF vectorizer.
            stopwords: Optional set of stopwords to filter during tokenization.
            tokenizer: Optional custom tokenizer function. If None, uses
                      Sudachi with punctuation and stopword filtering.
            random_state: Random state for reproducibility.
            **classifier_kwargs: Additional arguments passed to the classifier.

        Raises:
            ValueError: If classifier_type is not recognized.
        """
        self.class_names = class_names
        self.classifier_type = classifier_type
        self.max_features = max_features
        self.stopwords = stopwords
        self.random_state = random_state
        self.classifier_kwargs = classifier_kwargs

        # Set up tokenizer
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = lambda text: _default_tokenizer(
                text, stopwords=self.stopwords, filter_punct=True
            )

        # Initialize model pipeline (will be set during training)
        self._pipeline: Optional[Pipeline] = None
        self._explainer: Optional[TextExplainer] = None
        self.is_trained = False

    def _create_pipeline(self) -> Pipeline:
        """
        Create the sklearn pipeline with TF-IDF and classifier.

        Returns:
            Pipeline: sklearn Pipeline with vectorizer and classifier.
        """
        # Create TF-IDF vectorizer with our tokenizer
        vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,
            token_pattern=None,
            max_features=self.max_features,
        )

        # Create classifier based on type
        if self.classifier_type == 'logistic_regression':
            clf = LogisticRegression(
                max_iter=2000,
                random_state=self.random_state,
                **self.classifier_kwargs,
            )
        elif self.classifier_type == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=self.classifier_kwargs.get('n_estimators', 300),
                max_depth=self.classifier_kwargs.get('max_depth', None),
                random_state=self.random_state,
                n_jobs=-1,
                **{k: v for k, v in self.classifier_kwargs.items()
                   if k not in ['n_estimators', 'max_depth']},
            )
        else:
            raise ValueError(
                f"Unknown classifier_type: {self.classifier_type}. "
                "Options: 'logistic_regression', 'random_forest'"
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
        """
        Train the classifier on the provided data.

        Args:
            texts: List or Series of text documents.
            labels: List, array, or Series of integer labels.
            test_size: Fraction of data to use for testing.

        Returns:
            TrainingResult: Object containing training metrics.

        Example:
            >>> result = clf.train(texts, labels, test_size=0.2)
            >>> print(result.summary())
            >>> print(f"Accuracy: {result.accuracy:.2%}")
        """
        # Convert to lists if needed
        texts_list = list(texts)
        labels_array = np.array(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts_list,
            labels_array,
            test_size=test_size,
            random_state=self.random_state,
        )

        # Create and train pipeline
        self._pipeline = self._create_pipeline()
        self._pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self._pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Get classification report as dict
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
        )

        # Create explainer
        self._explainer = TextExplainer(
            class_names=self.class_names,
            split_expression=self._tokenizer,
            lang='jp',
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
    ) -> PredictionResult:
        """
        Predict the class of a single text with optional explanation.

        Args:
            text: The text to classify.
            explain: Whether to generate LIME explanation.
            num_features: Number of top features to include in explanation.
            num_samples: Number of samples for LIME perturbation.

        Returns:
            PredictionResult: Object containing prediction and explanation.

        Raises:
            RuntimeError: If model has not been trained.

        Example:
            >>> result = clf.predict("ニュースのテキスト", explain=True)
            >>> print(result.predicted_class)
            >>> print(result.summary_jp)
        """
        if not self.is_trained or self._pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call train() first or load a saved model."
            )

        # Get prediction and probabilities
        proba = self._pipeline.predict_proba([text])[0]
        predicted_label = int(np.argmax(proba))
        predicted_class = self.class_names[predicted_label]

        # Build probabilities dict
        probabilities = {
            name: float(proba[i])
            for i, name in enumerate(self.class_names)
        }

        # Generate explanation if requested
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
        )

    def predict_batch(
        self,
        texts: List[str],
        explain: bool = False,
        num_features: int = 10,
        num_samples: int = 500,
    ) -> List[PredictionResult]:
        """
        Predict classes for multiple texts.

        Args:
            texts: List of texts to classify.
            explain: Whether to generate LIME explanations (slower).
            num_features: Number of top features for explanations.
            num_samples: Number of samples for LIME perturbation.

        Returns:
            List of PredictionResult objects.

        Example:
            >>> results = clf.predict_batch(["テキスト1", "テキスト2"])
            >>> for result in results:
            ...     print(f"{result.predicted_class}: {result.confidence:.1%}")
        """
        if not self.is_trained or self._pipeline is None:
            raise RuntimeError(
                "Model has not been trained. Call train() first or load a saved model."
            )

        # Get batch predictions
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
            ))

        return results

    def _generate_explanation(
        self,
        text: str,
        proba: np.ndarray,
        num_features: int,
        num_samples: int,
    ) -> ExplanationResult:
        """
        Generate LIME explanation for a prediction.

        Args:
            text: The input text.
            proba: Prediction probabilities.
            num_features: Number of features to explain.
            num_samples: Number of LIME samples.

        Returns:
            ExplanationResult with word weights and summaries.
        """
        # Determine which class to explain (the predicted one)
        predicted_label = int(np.argmax(proba))

        # Generate LIME explanation
        exp = self._explainer.explain_instance(
            text,
            self._pipeline.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=tuple(range(len(self.class_names))),
        )

        # Get word weights for the predicted class
        word_weights = exp.as_list(label=predicted_label)

        # Generate summaries
        sentences_jp = summarize_lime_explanation_jp(exp, class_idx=predicted_label)
        sentences_en = summarize_lime_explanation(exp, class_idx=predicted_label)

        # Build probabilities dict
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
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to a file.

        The model is saved using joblib, which efficiently handles
        large numpy arrays and sklearn objects.

        Note:
            The tokenizer function cannot be pickled directly, so the
            TF-IDF vocabulary and classifier are saved separately and
            reconstructed on load.

        Args:
            path: File path to save the model (e.g., 'model.pkl').

        Raises:
            RuntimeError: If model has not been trained.

        Example:
            >>> clf.save('my_model.pkl')
        """
        if not self.is_trained or self._pipeline is None:
            raise RuntimeError("Model has not been trained. Nothing to save.")

        # Extract fitted components that can be pickled
        tfidf = self._pipeline.named_steps['tfidf']
        classifier = self._pipeline.named_steps['clf']

        save_data = {
            # Fitted model components
            'tfidf_vocabulary': tfidf.vocabulary_,
            'tfidf_idf': tfidf.idf_,
            'classifier': classifier,
            # Configuration for reconstruction
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
        """
        Load a trained model from a file.

        Args:
            path: File path to load the model from.

        Returns:
            JapaneseTextClassifier: Loaded classifier ready for predictions.

        Example:
            >>> clf = JapaneseTextClassifier.load('my_model.pkl')
            >>> result = clf.predict("テスト文章")
        """
        save_data = joblib.load(path)

        # Create new instance with saved parameters
        instance = cls(
            class_names=save_data['class_names'],
            classifier_type=save_data['classifier_type'],
            max_features=save_data['max_features'],
            stopwords=save_data['stopwords'],
            random_state=save_data['random_state'],
            **save_data['classifier_kwargs'],
        )

        # Reconstruct the TF-IDF vectorizer with our tokenizer
        tfidf = TfidfVectorizer(
            tokenizer=instance._tokenizer,
            token_pattern=None,
            max_features=instance.max_features,
            vocabulary=save_data['tfidf_vocabulary'],
        )
        # Set the fitted IDF values
        tfidf.idf_ = save_data['tfidf_idf']
        tfidf._tfidf._idf_diag = None  # Will be rebuilt on first transform

        # Reconstruct the pipeline
        instance._pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', save_data['classifier']),
        ])

        instance.is_trained = True

        # Recreate explainer
        instance._explainer = TextExplainer(
            class_names=instance.class_names,
            split_expression=instance._tokenizer,
            lang='jp',
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
