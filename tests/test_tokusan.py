"""
Comprehensive test suite for the tokusan package.

This module tests all key functionality including:
- Japanese tokenization
- Text explanation generation
- Plain language summaries (English and Japanese)
- Explanation visualization

Run tests with: pytest tests/test_tokusan.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

# =============================================================================
# Test Imports
# =============================================================================

class TestImports:
    """Test that all modules can be imported correctly."""

    def test_import_main_package(self):
        """Test importing the main tokusan package."""
        import tokusan
        assert hasattr(tokusan, '__version__')
        assert hasattr(tokusan, 'TextExplainer')

    def test_import_text_explainer(self):
        """Test importing TextExplainer class."""
        from tokusan import TextExplainer
        assert TextExplainer is not None

    def test_import_explanation(self):
        """Test importing Explanation class."""
        from tokusan import Explanation, DomainMapper
        assert Explanation is not None
        assert DomainMapper is not None

    def test_import_exceptions(self):
        """Test importing custom exceptions."""
        from tokusan import TokusanError, TokenizerError, ExplanationError
        assert issubclass(TokenizerError, TokusanError)
        assert issubclass(ExplanationError, TokusanError)

    def test_import_english_functions(self):
        """Test importing English explanation functions."""
        from tokusan import (
            generate_sentence_for_feature,
            summarize_lime_explanation,
            print_lime_narrative
        )
        assert callable(generate_sentence_for_feature)
        assert callable(summarize_lime_explanation)
        assert callable(print_lime_narrative)

    def test_import_japanese_functions(self):
        """Test importing Japanese explanation functions."""
        from tokusan import (
            generate_sentence_for_feature_jp,
            summarize_lime_explanation_jp,
            print_lime_narrative_jp
        )
        assert callable(generate_sentence_for_feature_jp)
        assert callable(summarize_lime_explanation_jp)
        assert callable(print_lime_narrative_jp)

    def test_import_japanese_tokenizer(self):
        """Test importing Japanese tokenizer utilities."""
        from tokusan import japanese_splitter, active_japanese_tokenizer
        assert callable(japanese_splitter)
        assert callable(active_japanese_tokenizer)


# =============================================================================
# Test Japanese Tokenization
# =============================================================================

class TestJapaneseTokenization:
    """Test Japanese tokenization functionality."""

    def test_active_tokenizer_returns_string(self):
        """Test that active_japanese_tokenizer returns a valid string."""
        from tokusan import active_japanese_tokenizer
        result = active_japanese_tokenizer()
        assert result in ['sudachi', 'fallback']

    def test_splitter_returns_list(self):
        """Test that japanese_splitter returns a list of tokens."""
        from tokusan import japanese_splitter
        text = "これはテストです"
        tokens = japanese_splitter(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_splitter_handles_empty_string(self):
        """Test splitter with empty string."""
        from tokusan import japanese_splitter
        tokens = japanese_splitter("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_splitter_handles_whitespace(self):
        """Test that whitespace is excluded from tokens."""
        from tokusan import japanese_splitter
        text = "テスト テスト"
        tokens = japanese_splitter(text)
        # No token should be whitespace
        for token in tokens:
            assert not token.isspace()


# =============================================================================
# Test IndexedString
# =============================================================================

class TestIndexedString:
    """Test IndexedString class for text indexing."""

    def test_indexed_string_basic(self):
        """Test basic IndexedString functionality with English text."""
        from tokusan.explainer import IndexedString
        text = "The quick brown fox"
        indexed = IndexedString(text)

        assert indexed.raw_string() == text
        assert indexed.num_words() == 4

    def test_indexed_string_word_retrieval(self):
        """Test retrieving words by ID."""
        from tokusan.explainer import IndexedString
        text = "hello world test"
        indexed = IndexedString(text)

        # Words should be retrievable
        words = [indexed.word(i) for i in range(indexed.num_words())]
        assert "hello" in words
        assert "world" in words
        assert "test" in words

    def test_indexed_string_inverse_removing(self):
        """Test removing words from the string."""
        from tokusan.explainer import IndexedString
        text = "hello world test"
        indexed = IndexedString(text)

        # Remove first word
        result = indexed.inverse_removing([0])
        assert indexed.word(0) not in result

    def test_indexed_string_bow_mode(self):
        """Test bag-of-words mode groups same words."""
        from tokusan.explainer import IndexedString
        text = "test test test"
        indexed = IndexedString(text, bow=True)

        # All "test" words should have same ID
        assert indexed.num_words() == 1

    def test_indexed_string_non_bow_mode(self):
        """Test non-bow mode treats each position separately."""
        from tokusan.explainer import IndexedString
        text = "test test test"
        indexed = IndexedString(text, bow=False)

        # Each position should be separate
        assert indexed.num_words() == 3

    def test_indexed_string_with_callable_tokenizer(self):
        """Test IndexedString with a custom tokenizer function."""
        from tokusan.explainer import IndexedString

        def custom_tokenizer(text):
            return text.split('-')

        text = "hello-world-test"
        indexed = IndexedString(text, split_expression=custom_tokenizer)

        assert indexed.num_words() == 3


# =============================================================================
# Test IndexedCharacters
# =============================================================================

class TestIndexedCharacters:
    """Test IndexedCharacters class for character-level indexing."""

    def test_indexed_characters_basic(self):
        """Test basic character indexing."""
        from tokusan.explainer import IndexedCharacters
        text = "hello"
        indexed = IndexedCharacters(text)

        assert indexed.raw_string() == text
        assert indexed.num_words() == 5  # 5 unique characters

    def test_indexed_characters_bow(self):
        """Test bow mode groups same characters."""
        from tokusan.explainer import IndexedCharacters
        text = "aaa"
        indexed = IndexedCharacters(text, bow=True)

        assert indexed.num_words() == 1

    def test_indexed_characters_inverse_removing(self):
        """Test removing characters."""
        from tokusan.explainer import IndexedCharacters
        text = "hello"
        indexed = IndexedCharacters(text, bow=False)

        result = indexed.inverse_removing([0])
        assert len(result) == len(text)  # Masked, not removed


# =============================================================================
# Test TextExplainer
# =============================================================================

class TestTextExplainer:
    """Test TextExplainer class."""

    def test_text_explainer_initialization_english(self):
        """Test TextExplainer initialization for English."""
        from tokusan import TextExplainer

        explainer = TextExplainer(
            class_names=['negative', 'positive'],
            lang='en'
        )

        assert explainer.class_names == ['negative', 'positive']
        assert explainer.lang == 'en'

    def test_text_explainer_initialization_japanese(self):
        """Test TextExplainer initialization for Japanese."""
        from tokusan import TextExplainer

        explainer = TextExplainer(
            class_names=['フェイク', '本物'],
            lang='jp'
        )

        assert explainer.class_names == ['フェイク', '本物']
        assert explainer.lang == 'jp'

    def test_text_explainer_explain_instance(self):
        """Test explain_instance method with mock classifier."""
        from tokusan import TextExplainer

        explainer = TextExplainer(
            class_names=['negative', 'positive'],
            lang='en'
        )

        # Create mock classifier that returns consistent probabilities
        def mock_classifier(texts):
            n = len(texts)
            # Return probabilities: [0.3, 0.7] for all texts
            return np.array([[0.3, 0.7] for _ in range(n)])

        text = "This is a great product"
        exp = explainer.explain_instance(
            text,
            mock_classifier,
            num_features=3,
            num_samples=100  # Small for speed
        )

        assert exp is not None
        assert hasattr(exp, 'predict_proba')
        assert hasattr(exp, 'local_exp')

    def test_text_explainer_explain_instance_japanese(self):
        """Test explain_instance with Japanese text."""
        from tokusan import TextExplainer

        explainer = TextExplainer(
            class_names=['フェイク', '本物'],
            lang='jp'
        )

        def mock_classifier(texts):
            n = len(texts)
            return np.array([[0.2, 0.8] for _ in range(n)])

        text = "これは信頼できるニュースです"
        exp = explainer.explain_instance(
            text,
            mock_classifier,
            num_features=3,
            num_samples=100
        )

        assert exp is not None
        assert exp.predict_proba is not None

    def test_text_explainer_char_level(self):
        """Test character-level explanation."""
        from tokusan import TextExplainer

        explainer = TextExplainer(
            class_names=['a', 'b'],
            char_level=True
        )

        def mock_classifier(texts):
            return np.array([[0.5, 0.5] for _ in range(len(texts))])

        text = "test"
        exp = explainer.explain_instance(
            text,
            mock_classifier,
            num_features=3,
            num_samples=50
        )

        assert exp is not None


# =============================================================================
# Test Explanation Class
# =============================================================================

class TestExplanation:
    """Test Explanation class and its methods."""

    def test_explanation_as_list(self):
        """Test getting explanation as list."""
        from tokusan import TextExplainer

        explainer = TextExplainer(class_names=['neg', 'pos'])

        def mock_classifier(texts):
            return np.array([[0.3, 0.7] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "good product",
            mock_classifier,
            labels=(1,),
            num_features=2,
            num_samples=100
        )

        result = exp.as_list(label=1)
        assert isinstance(result, list)
        # Each item should be (word, weight) tuple
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_explanation_as_map(self):
        """Test getting explanation as map."""
        from tokusan import TextExplainer

        explainer = TextExplainer(class_names=['neg', 'pos'])

        def mock_classifier(texts):
            return np.array([[0.3, 0.7] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "test text",
            mock_classifier,
            labels=(0, 1),
            num_features=2,
            num_samples=100
        )

        result = exp.as_map()
        assert isinstance(result, dict)

    def test_explanation_available_labels(self):
        """Test getting available labels."""
        from tokusan import TextExplainer

        explainer = TextExplainer(class_names=['neg', 'pos'])

        def mock_classifier(texts):
            return np.array([[0.3, 0.7] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "test",
            mock_classifier,
            labels=(0, 1),
            num_features=2,
            num_samples=100
        )

        labels = exp.available_labels()
        assert 0 in labels or 1 in labels


# =============================================================================
# Test Plain Language Functions (English)
# =============================================================================

class TestEnglishExplanationFunctions:
    """Test English plain language explanation functions."""

    def test_generate_sentence_positive_weight(self):
        """Test sentence generation with positive weight."""
        from tokusan import generate_sentence_for_feature

        sentence = generate_sentence_for_feature("excellent", 0.15, "positive")

        assert "excellent" in sentence
        assert "increased" in sentence
        assert "strongly" in sentence
        assert "positive" in sentence

    def test_generate_sentence_negative_weight(self):
        """Test sentence generation with negative weight."""
        from tokusan import generate_sentence_for_feature

        sentence = generate_sentence_for_feature("bad", -0.08, "positive")

        assert "bad" in sentence
        assert "decreased" in sentence
        assert "moderately" in sentence

    def test_generate_sentence_slight_weight(self):
        """Test sentence generation with slight weight."""
        from tokusan import generate_sentence_for_feature

        sentence = generate_sentence_for_feature("okay", 0.02, "neutral")

        assert "slightly" in sentence

    def test_summarize_lime_explanation(self):
        """Test summarizing LIME explanation."""
        from tokusan import TextExplainer, summarize_lime_explanation

        explainer = TextExplainer(class_names=['neg', 'pos'])

        def mock_classifier(texts):
            return np.array([[0.3, 0.7] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "great product",
            mock_classifier,
            labels=(1,),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation(exp, class_idx=1)

        assert isinstance(sentences, list)
        assert len(sentences) > 0
        for s in sentences:
            assert isinstance(s, str)


# =============================================================================
# Test Plain Language Functions (Japanese)
# =============================================================================

class TestJapaneseExplanationFunctions:
    """Test Japanese plain language explanation functions."""

    def test_generate_sentence_jp_positive(self):
        """Test Japanese sentence with positive weight."""
        from tokusan import generate_sentence_for_feature_jp

        sentence = generate_sentence_for_feature_jp("地震", 0.15, "フェイク")

        assert "地震" in sentence
        assert "上げました" in sentence
        assert "大きく" in sentence
        assert "フェイク" in sentence

    def test_generate_sentence_jp_negative(self):
        """Test Japanese sentence with negative weight."""
        from tokusan import generate_sentence_for_feature_jp

        sentence = generate_sentence_for_feature_jp("信頼", -0.08, "本物")

        assert "信頼" in sentence
        assert "下げました" in sentence
        assert "中程度に" in sentence

    def test_generate_sentence_jp_slight(self):
        """Test Japanese sentence with slight weight."""
        from tokusan import generate_sentence_for_feature_jp

        sentence = generate_sentence_for_feature_jp("テスト", 0.02, "クラス")

        assert "わずかに" in sentence

    def test_summarize_lime_explanation_jp(self):
        """Test Japanese explanation summary."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(class_names=['フェイク', '本物'], lang='jp')

        def mock_classifier(texts):
            return np.array([[0.2, 0.8] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "これは本物のニュースです",
            mock_classifier,
            labels=(1,),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=1)

        assert isinstance(sentences, list)
        assert len(sentences) == 2  # Two-sentence format


# =============================================================================
# Test Japanese Explanation Output Format
# =============================================================================

class TestJapaneseExplanationOutputFormat:
    """Test that Japanese explanation output matches the expected format."""

    def test_weight_format_strong_positive(self):
        """Test weight formatting for strong positive weight (> 0.10)."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(0.168)

        assert result == "重み=+0.168（強）"

    def test_weight_format_strong_negative(self):
        """Test weight formatting for strong negative weight."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(-0.105)

        assert result == "重み=-0.105（強）"

    def test_weight_format_medium_positive(self):
        """Test weight formatting for medium positive weight (0.05 < w <= 0.10)."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(0.051)

        assert result == "重み=+0.051（中）"

    def test_weight_format_medium_negative(self):
        """Test weight formatting for medium negative weight."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(-0.051)

        assert result == "重み=-0.051（中）"

    def test_weight_format_weak_positive(self):
        """Test weight formatting for weak positive weight (<= 0.05)."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(0.044)

        assert result == "重み=+0.044（弱）"

    def test_weight_format_weak_negative(self):
        """Test weight formatting for weak negative weight."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(-0.037)

        assert result == "重み=-0.037（弱）"

    def test_weight_format_zero(self):
        """Test weight formatting for zero weight."""
        from tokusan.explainer import _format_weight_jp

        result = _format_weight_jp(0.0)

        assert result == "重み=0.000（弱）"

    def test_summary_sentence1_structure(self):
        """Test that the first sentence has the correct structure."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.703, 0.297] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "年言っよるスポーツ病院わかっテスト",
            mock_classifier,
            labels=(0, 1),
            num_features=6,
            num_samples=200
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        # First sentence should contain classification result
        sent1 = sentences[0]
        assert "このインスタンスは" in sent1
        assert "対" in sent1
        assert "と分類されました" in sent1
        assert "への分類に最も強い影響を与えた言葉は" in sent1
        assert "それぞれの" in sent1
        assert "となっています" in sent1

    def test_summary_sentence2_structure(self):
        """Test that the second sentence has the correct structure."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.703, 0.297] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "年言っよるスポーツ病院わかっテスト",
            mock_classifier,
            labels=(0, 1),
            num_features=6,
            num_samples=200
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        # Second sentence should contain additional contributing words
        sent2 = sentences[1]
        assert "他に" in sent2
        assert "への分類の確率を上げた言葉として" in sent2
        assert "などが挙げられます" in sent2
        assert "への分類への確率を上げた言葉として" in sent2

    def test_summary_contains_probabilities(self):
        """Test that the summary includes probability values."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.703, 0.297] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テスト文章です",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)
        sent1 = sentences[0]

        # Should contain probabilities in format X.XXX対Y.YYY
        assert "0.703" in sent1
        assert "0.297" in sent1

    def test_summary_contains_class_names(self):
        """Test that the summary includes class names."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.703, 0.297] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テスト文章です",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)
        full_text = " ".join(sentences)

        # Both class names should appear
        assert "Fake" in full_text
        assert "Real" in full_text

    def test_summary_contains_weight_format(self):
        """Test that the summary contains weights in the correct format."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp
        import re

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.703, 0.297] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "年言っよるスポーツ病院わかっ",
            mock_classifier,
            labels=(0, 1),
            num_features=6,
            num_samples=200
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)
        full_text = " ".join(sentences)

        # Check weight format pattern appears
        weight_pattern = r"重み=[+-]?\d+\.\d{3}（[強中弱]）"
        matches = re.findall(weight_pattern, full_text)
        assert len(matches) >= 6  # Should have multiple weight indicators

    def test_summary_two_sentences(self):
        """Test that the summary returns exactly two sentences."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.7, 0.3] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テストの文章です",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        assert isinstance(sentences, list)
        assert len(sentences) == 2

    def test_summary_with_japanese_class_names(self):
        """Test summary generation with Japanese class names."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['フェイク', '本物'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.8, 0.2] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "これはテストです",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)
        full_text = " ".join(sentences)

        assert "フェイク" in full_text
        assert "本物" in full_text

    def test_summary_format_matches_expected(self):
        """Test that generated output matches the expected format structure."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.703, 0.297] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "年言っよるスポーツ病院わかっテスト",
            mock_classifier,
            labels=(0, 1),
            num_features=6,
            num_samples=300
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        # Verify sentence 1 follows the pattern:
        # このインスタンスはX対YでCLASSと分類されました。CLASSへの分類に最も強い影響を与えた言葉はW1, W2, W3で、それぞれのWEIGHT1, WEIGHT2, WEIGHT3となっています。
        sent1 = sentences[0]
        assert sent1.startswith("このインスタンスは")
        assert "Fakeと分類されました" in sent1 or "Realと分類されました" in sent1

        # Verify sentence 2 follows the pattern:
        # 他にCLASSへの分類の確率を上げた言葉としてW1 (WEIGHT1)、W2 (WEIGHT2)、W3 (WEIGHT3)などが挙げられます。CLASS2への分類への確率を上げた言葉として、W4 (WEIGHT4)、W5 (WEIGHT5)、W6 (WEIGHT6)などが挙げられます。
        sent2 = sentences[1]
        assert sent2.startswith("他に")
        assert "などが挙げられます" in sent2

    def test_print_lime_narrative_jp_output(self, capsys):
        """Test that print_lime_narrative_jp produces formatted output."""
        from tokusan import TextExplainer, print_lime_narrative_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.7, 0.3] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テスト文章",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=100
        )

        print_lime_narrative_jp(exp, class_idx=0)
        captured = capsys.readouterr()

        assert "LIME出力の自然言語による説明" in captured.out
        assert "このインスタンスは" in captured.out


class TestJapaneseExplanationEdgeCases:
    """Test edge cases for Japanese explanation generation."""

    def test_summary_with_single_word(self):
        """Test summary when input has very few words."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.6, 0.4] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テスト",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=50
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        assert len(sentences) == 2
        # Should handle padding with "-" for missing words
        assert "このインスタンスは" in sentences[0]

    def test_summary_with_equal_probabilities(self):
        """Test summary when class probabilities are equal."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['Fake', 'Real'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.5, 0.5] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テスト文章です",
            mock_classifier,
            labels=(0, 1),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        assert len(sentences) == 2
        assert "0.500" in sentences[0]

    def test_summary_multiclass(self):
        """Test summary with more than two classes."""
        from tokusan import TextExplainer, summarize_lime_explanation_jp

        explainer = TextExplainer(
            class_names=['クラスA', 'クラスB', 'クラスC'],
            lang='jp',
            random_state=42
        )

        def mock_classifier(texts):
            return np.array([[0.5, 0.3, 0.2] for _ in range(len(texts))])

        exp = explainer.explain_instance(
            "テスト文章です",
            mock_classifier,
            labels=(0, 1, 2),
            num_features=3,
            num_samples=100
        )

        sentences = summarize_lime_explanation_jp(exp, class_idx=0)

        assert len(sentences) == 2
        # Primary class should be A (highest probability)
        assert "クラスA" in sentences[0]

    def test_weight_boundaries(self):
        """Test weight formatting at exact boundary values."""
        from tokusan.explainer import _format_weight_jp

        # Exactly at strong boundary
        result_at_strong = _format_weight_jp(0.10)
        assert "弱" in result_at_strong or "中" in result_at_strong

        # Just above strong boundary
        result_above_strong = _format_weight_jp(0.101)
        assert "強" in result_above_strong

        # Exactly at medium boundary
        result_at_medium = _format_weight_jp(0.05)
        assert "弱" in result_at_medium

        # Just above medium boundary
        result_above_medium = _format_weight_jp(0.051)
        assert "中" in result_above_medium


# =============================================================================
# Test LimeBase
# =============================================================================

class TestLimeBase:
    """Test LimeBase class for core LIME algorithm."""

    def test_lime_base_initialization(self):
        """Test LimeBase initialization."""
        from tokusan.base import LimeBase

        def kernel_fn(distances):
            return np.exp(-distances)

        base = LimeBase(kernel_fn, verbose=False)

        assert base.kernel_fn is not None
        assert base.verbose is False

    def test_lime_base_feature_selection_none(self):
        """Test feature selection with 'none' method."""
        from tokusan.base import LimeBase

        def kernel_fn(distances):
            return np.ones_like(distances)

        base = LimeBase(kernel_fn)

        data = np.random.rand(100, 5)
        labels = np.random.rand(100)
        weights = np.ones(100)

        features = base.feature_selection(data, labels, weights, 3, 'none')

        assert len(features) == 5  # All features

    def test_lime_base_feature_selection_auto(self):
        """Test feature selection with 'auto' method."""
        from tokusan.base import LimeBase

        def kernel_fn(distances):
            return np.ones_like(distances)

        base = LimeBase(kernel_fn)

        data = np.random.rand(100, 10)
        labels = np.random.rand(100)
        weights = np.ones(100)

        features = base.feature_selection(data, labels, weights, 3, 'auto')

        assert len(features) <= 3


# =============================================================================
# Test Exceptions
# =============================================================================

class TestExceptions:
    """Test custom exception classes."""

    def test_tokusan_error(self):
        """Test TokusanError can be raised and caught."""
        from tokusan import TokusanError

        with pytest.raises(TokusanError):
            raise TokusanError("Test error")

    def test_tokenizer_error_inheritance(self):
        """Test TokenizerError inherits from TokusanError."""
        from tokusan import TokusanError, TokenizerError

        with pytest.raises(TokusanError):
            raise TokenizerError("Tokenizer failed")

    def test_explanation_error_inheritance(self):
        """Test ExplanationError inherits from TokusanError."""
        from tokusan import TokusanError, ExplanationError

        with pytest.raises(TokusanError):
            raise ExplanationError("Explanation failed")


# =============================================================================
# Test Domain Mapper
# =============================================================================

class TestDomainMapper:
    """Test DomainMapper class."""

    def test_domain_mapper_default_behavior(self):
        """Test default DomainMapper returns input unchanged."""
        from tokusan import DomainMapper

        mapper = DomainMapper()
        exp = [(0, 0.5), (1, -0.3)]

        result = mapper.map_exp_ids(exp)

        assert result == exp

    def test_text_domain_mapper(self):
        """Test TextDomainMapper maps IDs to words."""
        from tokusan.explainer import TextDomainMapper, IndexedString

        text = "hello world"
        indexed = IndexedString(text)
        mapper = TextDomainMapper(indexed)

        # Create fake explanation with feature IDs
        exp = [(0, 0.5), (1, -0.3)]
        result = mapper.map_exp_ids(exp)

        # Result should contain words, not IDs
        assert all(isinstance(item[0], str) for item in result)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_english_workflow(self):
        """Test complete workflow for English text."""
        from tokusan import (
            TextExplainer,
            summarize_lime_explanation,
            generate_sentence_for_feature
        )

        # Initialize
        explainer = TextExplainer(
            class_names=['negative', 'positive'],
            lang='en'
        )

        # Mock classifier
        def classifier(texts):
            return np.array([[0.2, 0.8] for _ in range(len(texts))])

        # Explain
        text = "This product is excellent and amazing"
        exp = explainer.explain_instance(
            text,
            classifier,
            num_features=5,
            num_samples=200
        )

        # Get list format
        word_weights = exp.as_list(label=1)
        assert len(word_weights) > 0

        # Generate sentences
        for word, weight in word_weights[:3]:
            sentence = generate_sentence_for_feature(word, weight, 'positive')
            assert isinstance(sentence, str)
            assert word in sentence

        # Get summary
        summary = summarize_lime_explanation(exp, class_idx=1)
        assert len(summary) > 0

    def test_full_japanese_workflow(self):
        """Test complete workflow for Japanese text."""
        from tokusan import (
            TextExplainer,
            summarize_lime_explanation_jp,
            generate_sentence_for_feature_jp,
            active_japanese_tokenizer
        )

        # Check tokenizer status
        tokenizer = active_japanese_tokenizer()
        print(f"Using tokenizer: {tokenizer}")

        # Initialize
        explainer = TextExplainer(
            class_names=['フェイク', '本物'],
            lang='jp'
        )

        # Mock classifier
        def classifier(texts):
            return np.array([[0.3, 0.7] for _ in range(len(texts))])

        # Explain
        text = "このニュースは信頼できる内容です"
        exp = explainer.explain_instance(
            text,
            classifier,
            num_features=5,
            num_samples=200
        )

        # Get list format
        word_weights = exp.as_list(label=1)
        assert len(word_weights) > 0

        # Generate Japanese sentences
        for word, weight in word_weights[:3]:
            sentence = generate_sentence_for_feature_jp(word, weight, '本物')
            assert isinstance(sentence, str)
            assert word in sentence

        # Get Japanese summary
        summary = summarize_lime_explanation_jp(exp, class_idx=1)
        assert len(summary) == 2  # Two sentences


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
