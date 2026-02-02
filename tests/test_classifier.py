"""
Tests for the JapaneseTextClassifier module.

This module tests the end-to-end classifier functionality including:
- Initialization
- Training
- Prediction with explanation
- Model persistence (save/load)
- Result classes

Run tests with: pytest tests/test_classifier.py -v
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# Test Imports
# =============================================================================

class TestClassifierImports:
    """Test that classifier module imports correctly."""

    def test_import_classifier(self):
        """Test importing JapaneseTextClassifier."""
        from tokusan import JapaneseTextClassifier
        assert JapaneseTextClassifier is not None

    def test_import_result_classes(self):
        """Test importing result classes."""
        from tokusan import TrainingResult, PredictionResult, ExplanationResult
        assert TrainingResult is not None
        assert PredictionResult is not None
        assert ExplanationResult is not None


# =============================================================================
# Test Result Classes
# =============================================================================

class TestTrainingResult:
    """Test TrainingResult class."""

    def test_training_result_creation(self):
        """Test creating a TrainingResult instance."""
        from tokusan import TrainingResult

        result = TrainingResult(
            accuracy=0.85,
            classification_report={
                'Fake': {'precision': 0.87, 'recall': 0.59, 'f1-score': 0.70},
                'Real': {'precision': 0.85, 'recall': 0.96, 'f1-score': 0.90},
            },
            train_size=10000,
            test_size=2500,
            class_names=['Fake', 'Real'],
        )

        assert result.accuracy == 0.85
        assert result.train_size == 10000
        assert result.test_size == 2500

    def test_training_result_summary(self):
        """Test TrainingResult summary method."""
        from tokusan import TrainingResult

        result = TrainingResult(
            accuracy=0.85,
            classification_report={
                'Fake': {'precision': 0.87, 'recall': 0.59, 'f1-score': 0.70},
                'Real': {'precision': 0.85, 'recall': 0.96, 'f1-score': 0.90},
            },
            train_size=10000,
            test_size=2500,
            class_names=['Fake', 'Real'],
        )

        summary = result.summary()
        assert "Training completed successfully" in summary
        assert "85" in summary  # Accuracy percentage
        assert "Fake" in summary
        assert "Real" in summary

    def test_training_result_summary_jp(self):
        """Test TrainingResult Japanese summary."""
        from tokusan import TrainingResult

        result = TrainingResult(
            accuracy=0.85,
            classification_report={
                'Fake': {'precision': 0.87, 'recall': 0.59, 'f1-score': 0.70},
            },
            train_size=10000,
            test_size=2500,
            class_names=['Fake', 'Real'],
        )

        summary = result.summary_jp()
        assert "学習が完了しました" in summary
        assert "正確度" in summary

    def test_training_result_to_dict(self):
        """Test TrainingResult to_dict method."""
        from tokusan import TrainingResult

        result = TrainingResult(
            accuracy=0.85,
            classification_report={'Fake': {}},
            train_size=10000,
            test_size=2500,
            class_names=['Fake', 'Real'],
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['accuracy'] == 0.85
        assert d['train_size'] == 10000

    def test_training_result_to_html(self):
        """Test TrainingResult to_html method."""
        from tokusan import TrainingResult

        result = TrainingResult(
            accuracy=0.85,
            classification_report={
                'Fake': {'precision': 0.87, 'recall': 0.59, 'f1-score': 0.70},
            },
            train_size=10000,
            test_size=2500,
            class_names=['Fake'],
        )

        html = result.to_html()
        assert "<div" in html
        assert "training-result" in html
        assert "85" in html


class TestExplanationResult:
    """Test ExplanationResult class."""

    def test_explanation_result_creation(self):
        """Test creating an ExplanationResult instance."""
        from tokusan import ExplanationResult

        result = ExplanationResult(
            word_weights=[('年', 0.037), ('2020', 0.021), ('部', -0.035)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            sentences_jp=['テスト文1', 'テスト文2'],
            sentences_en=['Test sentence 1', 'Test sentence 2'],
        )

        assert result.class_name == 'Fake'
        assert result.probability == 0.88
        assert len(result.word_weights) == 3

    def test_explanation_result_top_positive_words(self):
        """Test top_positive_words property."""
        from tokusan import ExplanationResult

        result = ExplanationResult(
            word_weights=[('年', 0.037), ('2020', 0.021), ('部', -0.035)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
        )

        positive = result.top_positive_words
        assert len(positive) == 2
        assert all(wt > 0 for _, wt in positive)

    def test_explanation_result_top_negative_words(self):
        """Test top_negative_words property."""
        from tokusan import ExplanationResult

        result = ExplanationResult(
            word_weights=[('年', 0.037), ('2020', 0.021), ('部', -0.035)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
        )

        negative = result.top_negative_words
        assert len(negative) == 1
        assert all(wt < 0 for _, wt in negative)

    def test_explanation_result_summary_jp(self):
        """Test summary_jp property."""
        from tokusan import ExplanationResult

        result = ExplanationResult(
            word_weights=[('年', 0.037)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            sentences_jp=['テスト文1', 'テスト文2'],
            sentences_en=['Test 1', 'Test 2'],
        )

        assert result.summary_jp == "テスト文1\nテスト文2"

    def test_explanation_result_to_dict(self):
        """Test ExplanationResult to_dict method."""
        from tokusan import ExplanationResult

        result = ExplanationResult(
            word_weights=[('年', 0.037)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['class_name'] == 'Fake'
        assert d['probability'] == 0.88
        assert 'word_weights' in d

    def test_explanation_result_to_html(self):
        """Test ExplanationResult to_html method."""
        from tokusan import ExplanationResult

        result = ExplanationResult(
            word_weights=[('年', 0.037), ('部', -0.035)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            sentences_jp=['テスト文'],
            sentences_en=['Test'],
        )

        html = result.to_html(lang='jp')
        assert "<div" in html
        assert "explanation-result" in html


class TestPredictionResult:
    """Test PredictionResult class."""

    def test_prediction_result_creation(self):
        """Test creating a PredictionResult instance."""
        from tokusan import PredictionResult

        result = PredictionResult(
            text="テスト文章",
            predicted_class='Fake',
            predicted_label=1,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            class_names=['Real', 'Fake'],
        )

        assert result.predicted_class == 'Fake'
        assert result.predicted_label == 1
        assert result.confidence == 0.88

    def test_prediction_result_confidence(self):
        """Test confidence property."""
        from tokusan import PredictionResult

        result = PredictionResult(
            text="テスト",
            predicted_class='Real',
            predicted_label=0,
            probabilities={'Real': 0.75, 'Fake': 0.25},
            class_names=['Real', 'Fake'],
        )

        assert result.confidence == 0.75

    def test_prediction_result_summary_jp(self):
        """Test summary_jp property."""
        from tokusan import PredictionResult

        result = PredictionResult(
            text="テスト",
            predicted_class='Fake',
            predicted_label=1,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            class_names=['Real', 'Fake'],
        )

        summary = result.summary_jp
        assert "予測結果" in summary
        assert "Fake" in summary
        assert "88" in summary

    def test_prediction_result_summary_en(self):
        """Test summary_en property."""
        from tokusan import PredictionResult

        result = PredictionResult(
            text="test",
            predicted_class='Fake',
            predicted_label=1,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            class_names=['Real', 'Fake'],
        )

        summary = result.summary_en
        assert "Prediction" in summary
        assert "Fake" in summary

    def test_prediction_result_with_explanation(self):
        """Test PredictionResult with explanation."""
        from tokusan import PredictionResult, ExplanationResult

        explanation = ExplanationResult(
            word_weights=[('年', 0.037)],
            class_name='Fake',
            class_names=['Real', 'Fake'],
            probability=0.88,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            sentences_jp=['テスト説明'],
            sentences_en=['Test explanation'],
        )

        result = PredictionResult(
            text="テスト",
            predicted_class='Fake',
            predicted_label=1,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            class_names=['Real', 'Fake'],
            explanation=explanation,
        )

        summary = result.summary_jp
        assert "説明" in summary
        assert "テスト説明" in summary

    def test_prediction_result_to_dict(self):
        """Test PredictionResult to_dict method."""
        from tokusan import PredictionResult

        result = PredictionResult(
            text="テスト",
            predicted_class='Fake',
            predicted_label=1,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            class_names=['Real', 'Fake'],
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['predicted_class'] == 'Fake'
        assert d['confidence'] == 0.88
        assert 'summary_jp' in d
        assert 'summary_en' in d

    def test_prediction_result_to_html(self):
        """Test PredictionResult to_html method."""
        from tokusan import PredictionResult

        result = PredictionResult(
            text="テスト",
            predicted_class='Fake',
            predicted_label=1,
            probabilities={'Real': 0.12, 'Fake': 0.88},
            class_names=['Real', 'Fake'],
        )

        html = result.to_html(lang='jp')
        assert "<div" in html
        assert "prediction-result" in html


# =============================================================================
# Test JapaneseTextClassifier
# =============================================================================

class TestJapaneseTextClassifierInit:
    """Test JapaneseTextClassifier initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        from tokusan import JapaneseTextClassifier

        clf = JapaneseTextClassifier(class_names=['Fake', 'Real'])

        assert clf.class_names == ['Fake', 'Real']
        assert clf.classifier_type == 'logistic_regression'
        assert clf.is_trained is False

    def test_init_random_forest(self):
        """Test initialization with random forest."""
        from tokusan import JapaneseTextClassifier

        clf = JapaneseTextClassifier(
            class_names=['Fake', 'Real'],
            classifier_type='random_forest',
        )

        assert clf.classifier_type == 'random_forest'

    def test_init_with_stopwords(self):
        """Test initialization with custom stopwords."""
        from tokusan import JapaneseTextClassifier

        stopwords = {'の', 'は', 'が'}
        clf = JapaneseTextClassifier(
            class_names=['Fake', 'Real'],
            stopwords=stopwords,
        )

        assert clf.stopwords == stopwords

    def test_repr(self):
        """Test string representation."""
        from tokusan import JapaneseTextClassifier

        clf = JapaneseTextClassifier(class_names=['Fake', 'Real'])
        repr_str = repr(clf)

        assert "JapaneseTextClassifier" in repr_str
        assert "untrained" in repr_str


class TestJapaneseTextClassifierTraining:
    """Test JapaneseTextClassifier training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        texts = [
            "これは本物のニュースです",
            "信頼できる情報源からの記事",
            "正確な報道がされています",
            "これはフェイクニュースです",
            "偽の情報が含まれています",
            "でたらめな内容の記事",
        ] * 20  # Repeat to have enough data

        labels = [0, 0, 0, 1, 1, 1] * 20

        return texts, labels

    def test_train_basic(self, sample_data):
        """Test basic training."""
        from tokusan import JapaneseTextClassifier

        texts, labels = sample_data
        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])

        result = clf.train(texts, labels, test_size=0.2)

        assert clf.is_trained is True
        assert isinstance(result.accuracy, float)
        assert 0 <= result.accuracy <= 1

    def test_train_returns_training_result(self, sample_data):
        """Test that train returns TrainingResult."""
        from tokusan import JapaneseTextClassifier, TrainingResult

        texts, labels = sample_data
        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])

        result = clf.train(texts, labels)

        assert isinstance(result, TrainingResult)
        assert result.train_size > 0
        assert result.test_size > 0

    def test_train_with_random_forest(self, sample_data):
        """Test training with random forest classifier."""
        from tokusan import JapaneseTextClassifier

        texts, labels = sample_data
        clf = JapaneseTextClassifier(
            class_names=['Real', 'Fake'],
            classifier_type='random_forest',
            n_estimators=10,  # Small for speed
        )

        result = clf.train(texts, labels)

        assert clf.is_trained is True


class TestJapaneseTextClassifierPrediction:
    """Test JapaneseTextClassifier prediction."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier for testing."""
        from tokusan import JapaneseTextClassifier

        texts = [
            "これは本物のニュースです",
            "信頼できる情報源からの記事",
            "これはフェイクニュースです",
            "偽の情報が含まれています",
        ] * 30

        labels = [0, 0, 1, 1] * 30

        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])
        clf.train(texts, labels)

        return clf

    def test_predict_basic(self, trained_classifier):
        """Test basic prediction."""
        result = trained_classifier.predict("テスト文章", explain=False)

        assert result.predicted_class in ['Real', 'Fake']
        assert result.predicted_label in [0, 1]
        assert 0 <= result.confidence <= 1

    def test_predict_returns_prediction_result(self, trained_classifier):
        """Test that predict returns PredictionResult."""
        from tokusan import PredictionResult

        result = trained_classifier.predict("テスト", explain=False)

        assert isinstance(result, PredictionResult)

    def test_predict_with_explanation(self, trained_classifier):
        """Test prediction with explanation."""
        result = trained_classifier.predict(
            "これはテストです",
            explain=True,
            num_features=5,
            num_samples=100,  # Small for speed
        )

        assert result.explanation is not None
        assert len(result.explanation.word_weights) > 0
        assert len(result.explanation.sentences_jp) == 2

    def test_predict_probabilities(self, trained_classifier):
        """Test that probabilities are returned correctly."""
        result = trained_classifier.predict("テスト", explain=False)

        assert 'Real' in result.probabilities
        assert 'Fake' in result.probabilities
        assert abs(sum(result.probabilities.values()) - 1.0) < 0.01

    def test_predict_untrained_raises_error(self):
        """Test that predicting with untrained model raises error."""
        from tokusan import JapaneseTextClassifier

        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])

        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict("テスト")


class TestJapaneseTextClassifierBatchPrediction:
    """Test batch prediction functionality."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        from tokusan import JapaneseTextClassifier

        texts = ["本物のニュース", "フェイクニュース"] * 50
        labels = [0, 1] * 50

        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])
        clf.train(texts, labels)

        return clf

    def test_predict_batch_basic(self, trained_classifier):
        """Test basic batch prediction."""
        texts = ["テスト1", "テスト2", "テスト3"]
        results = trained_classifier.predict_batch(texts, explain=False)

        assert len(results) == 3
        for result in results:
            assert result.predicted_class in ['Real', 'Fake']

    def test_predict_batch_with_explanation(self, trained_classifier):
        """Test batch prediction with explanations."""
        texts = ["テスト1", "テスト2"]
        results = trained_classifier.predict_batch(
            texts,
            explain=True,
            num_samples=50,
        )

        assert len(results) == 2
        for result in results:
            assert result.explanation is not None


class TestJapaneseTextClassifierPersistence:
    """Test model save/load functionality."""

    @pytest.fixture
    def trained_classifier(self):
        """Create a trained classifier."""
        from tokusan import JapaneseTextClassifier

        texts = ["本物", "フェイク"] * 50
        labels = [0, 1] * 50

        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])
        clf.train(texts, labels)

        return clf

    def test_save_and_load(self, trained_classifier):
        """Test saving and loading a model."""
        from tokusan import JapaneseTextClassifier

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            # Save
            trained_classifier.save(path)
            assert path.exists()

            # Load
            loaded = JapaneseTextClassifier.load(path)
            assert loaded.is_trained
            assert loaded.class_names == trained_classifier.class_names

    def test_loaded_model_can_predict(self, trained_classifier):
        """Test that loaded model can make predictions."""
        from tokusan import JapaneseTextClassifier

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            trained_classifier.save(path)
            loaded = JapaneseTextClassifier.load(path)

            result = loaded.predict("テスト", explain=False)
            assert result.predicted_class in ['Real', 'Fake']

    def test_loaded_model_can_explain(self, trained_classifier):
        """Test that loaded model can generate explanations."""
        from tokusan import JapaneseTextClassifier

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            trained_classifier.save(path)
            loaded = JapaneseTextClassifier.load(path)

            result = loaded.predict("テスト", explain=True, num_samples=50)
            assert result.explanation is not None

    def test_save_untrained_raises_error(self):
        """Test that saving untrained model raises error."""
        from tokusan import JapaneseTextClassifier

        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            with pytest.raises(RuntimeError, match="not been trained"):
                clf.save(path)


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEndWorkflow:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self):
        """Test complete train -> predict -> explain workflow."""
        from tokusan import JapaneseTextClassifier

        # Sample data
        texts = [
            "これは信頼できるニュースです",
            "正確な情報が報道されています",
            "これはフェイクニュースです",
            "偽の情報が拡散されています",
        ] * 30
        labels = [0, 0, 1, 1] * 30

        # Create and train
        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])
        train_result = clf.train(texts, labels)

        assert train_result.accuracy > 0

        # Predict with explanation
        pred = clf.predict("新しいニュース記事です", explain=True, num_samples=100)

        assert pred.predicted_class in ['Real', 'Fake']
        assert pred.explanation is not None
        assert len(pred.summary_jp) > 0

        # Convert to dict for API
        d = pred.to_dict()
        assert 'predicted_class' in d
        assert 'explanation' in d

    def test_workflow_with_save_load(self):
        """Test workflow with model persistence."""
        from tokusan import JapaneseTextClassifier

        texts = ["本物", "フェイク"] * 50
        labels = [0, 1] * 50

        clf = JapaneseTextClassifier(class_names=['Real', 'Fake'])
        clf.train(texts, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"

            # Save
            clf.save(path)

            # Load and predict
            loaded = JapaneseTextClassifier.load(path)
            result = loaded.predict("テスト", explain=True, num_samples=50)

            assert result.predicted_class in ['Real', 'Fake']
            assert result.explanation is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
