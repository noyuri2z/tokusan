"""
Pytest configuration and fixtures for tokusan tests.
"""

import numpy as np
import pytest


@pytest.fixture
def sample_english_text():
    """Sample English text for testing."""
    return "This product is excellent and I highly recommend it to everyone."


@pytest.fixture
def sample_japanese_text():
    """Sample Japanese text for testing."""
    return "このニュースは信頼できる内容です。日本語のテキストを分析します。"


@pytest.fixture
def mock_binary_classifier():
    """Mock binary classifier returning [0.3, 0.7] for all texts."""
    def classifier(texts):
        n = len(texts)
        return np.array([[0.3, 0.7] for _ in range(n)])

    return classifier


@pytest.fixture
def mock_multiclass_classifier():
    """Mock 3-class classifier returning [0.2, 0.5, 0.3] for all texts."""
    def classifier(texts):
        n = len(texts)
        return np.array([[0.2, 0.5, 0.3] for _ in range(n)])

    return classifier


@pytest.fixture
def english_explainer():
    """TextExplainer configured for English text."""
    from tokusan import TextExplainer
    return TextExplainer(
        class_names=['negative', 'positive'],
        lang='en',
        random_state=42
    )


@pytest.fixture
def japanese_explainer():
    """TextExplainer configured for Japanese text."""
    from tokusan import TextExplainer
    return TextExplainer(
        class_names=['フェイク', '本物'],
        lang='jp',
        random_state=42
    )
