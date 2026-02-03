"""
Tests for the AI interpreter module.

These tests use mocked API calls to avoid requiring a real API key.
"""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestGeminiInterpreter:
    """Tests for GeminiInterpreter class."""

    def test_init_without_api_key(self):
        """Test that initialization fails without API key."""
        from tokusan.ai_interpreter import GeminiInterpreter
        from tokusan.exceptions import AIInterpretationError

        # Ensure API key is not set, but package is available
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GEMINI_API_KEY', None)
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                with pytest.raises(AIInterpretationError, match="GEMINI_API_KEY"):
                    GeminiInterpreter()

    def test_init_without_package(self):
        """Test that initialization fails without google-generativeai package."""
        from tokusan.exceptions import AIInterpretationError

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=False):
                from tokusan.ai_interpreter import GeminiInterpreter
                with pytest.raises(AIInterpretationError, match="google-generativeai"):
                    GeminiInterpreter()

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import GeminiInterpreter
                interpreter = GeminiInterpreter()
                assert interpreter.api_key == 'test_key'
                assert interpreter.model_name == 'gemini-1.5-flash'

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import GeminiInterpreter
                interpreter = GeminiInterpreter(model_name='gemini-pro')
                assert interpreter.model_name == 'gemini-pro'

    def test_build_prompt(self):
        """Test prompt building."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import GeminiInterpreter
                interpreter = GeminiInterpreter()

                prompt = interpreter._build_prompt(
                    text="テストテキスト",
                    predicted_class="Fake",
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    word_weights=[("年", 0.05), ("記事", -0.03)],
                    class_names=["Fake", "Real"],
                )

                # Check that prompt contains expected elements
                assert "テストテキスト" in prompt
                assert "Fake" in prompt
                assert "80.0%" in prompt or "0.8" in prompt
                assert "年" in prompt
                assert "記事" in prompt

    def test_build_prompt_truncates_long_text(self):
        """Test that long text is truncated in prompt."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import GeminiInterpreter
                interpreter = GeminiInterpreter()

                long_text = "あ" * 2000  # 2000 characters
                prompt = interpreter._build_prompt(
                    text=long_text,
                    predicted_class="Fake",
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    word_weights=[],
                    class_names=["Fake", "Real"],
                )

                assert "...(省略)" in prompt

    def test_interpret_success(self):
        """Test successful interpretation."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import GeminiInterpreter

                # Mock the Gemini model
                mock_response = MagicMock()
                mock_response.text = "この記事はフェイクニュースと判定されました。"

                mock_model = MagicMock()
                mock_model.generate_content.return_value = mock_response

                interpreter = GeminiInterpreter()
                interpreter._model = mock_model

                result = interpreter.interpret(
                    text="テストテキスト",
                    predicted_class="Fake",
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    word_weights=[("年", 0.05)],
                    class_names=["Fake", "Real"],
                )

                assert result == "この記事はフェイクニュースと判定されました。"
                mock_model.generate_content.assert_called_once()

    def test_interpret_api_error(self):
        """Test that API errors are wrapped in AIInterpretationError."""
        from tokusan.exceptions import AIInterpretationError

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import GeminiInterpreter

                mock_model = MagicMock()
                mock_model.generate_content.side_effect = Exception("API error")

                interpreter = GeminiInterpreter()
                interpreter._model = mock_model

                with pytest.raises(AIInterpretationError, match="API error"):
                    interpreter.interpret(
                        text="テスト",
                        predicted_class="Fake",
                        probabilities={"Fake": 0.8, "Real": 0.2},
                        word_weights=[],
                        class_names=["Fake", "Real"],
                    )


class TestIsAiAvailable:
    """Tests for is_ai_available function."""

    def test_available_when_package_and_key_present(self):
        """Test returns True when package installed and key set."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import is_ai_available
                assert is_ai_available() is True

    def test_not_available_without_key(self):
        """Test returns False when API key not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GEMINI_API_KEY', None)
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                from tokusan.ai_interpreter import is_ai_available
                assert is_ai_available() is False

    def test_not_available_without_package(self):
        """Test returns False when package not installed."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=False):
                from tokusan.ai_interpreter import is_ai_available
                assert is_ai_available() is False


class TestPredictionResultAiIntegration:
    """Tests for AI integration in PredictionResult."""

    def test_summary_jp_uses_template_without_api_key(self):
        """Test that summary_jp uses template when no API key."""
        from tokusan.results import PredictionResult, ExplanationResult

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GEMINI_API_KEY', None)

            explanation = ExplanationResult(
                word_weights=[("テスト", 0.1)],
                class_name="Fake",
                class_names=["Fake", "Real"],
                probability=0.8,
                probabilities={"Fake": 0.8, "Real": 0.2},
                sentences_jp=["テンプレート説明"],
                sentences_en=["Template explanation"],
            )

            result = PredictionResult(
                text="テストテキスト",
                predicted_class="Fake",
                predicted_label=0,
                probabilities={"Fake": 0.8, "Real": 0.2},
                class_names=["Fake", "Real"],
                explanation=explanation,
            )

            summary = result.summary_jp
            assert "予測結果: Fake" in summary
            assert "テンプレート説明" in summary

    def test_summary_jp_uses_ai_with_api_key(self):
        """Test that summary_jp uses AI when API key is set."""
        from tokusan.results import PredictionResult, ExplanationResult

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                # Mock the interpreter
                mock_interpreter = MagicMock()
                mock_interpreter.interpret.return_value = "AI生成の解釈です。"

                explanation = ExplanationResult(
                    word_weights=[("テスト", 0.1)],
                    class_name="Fake",
                    class_names=["Fake", "Real"],
                    probability=0.8,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    sentences_jp=["テンプレート説明"],
                    sentences_en=["Template explanation"],
                )

                result = PredictionResult(
                    text="テストテキスト",
                    predicted_class="Fake",
                    predicted_label=0,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    class_names=["Fake", "Real"],
                    explanation=explanation,
                )

                # Patch where the import happens (inside the method)
                with patch('tokusan.ai_interpreter.GeminiInterpreter', return_value=mock_interpreter):
                    summary = result.summary_jp
                    assert summary == "AI生成の解釈です。"
                    mock_interpreter.interpret.assert_called_once()

    def test_summary_jp_template_without_explanation(self):
        """Test template summary works without explanation."""
        from tokusan.results import PredictionResult

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GEMINI_API_KEY', None)

            result = PredictionResult(
                text="テストテキスト",
                predicted_class="Fake",
                predicted_label=0,
                probabilities={"Fake": 0.8, "Real": 0.2},
                class_names=["Fake", "Real"],
                explanation=None,
            )

            summary = result.summary_jp
            assert "予測結果: Fake" in summary
            assert "80.0%" in summary

    def test_summary_jp_ai_error_propagates(self):
        """Test that AI errors propagate correctly when fallback is disabled."""
        from tokusan.results import PredictionResult, ExplanationResult
        from tokusan.exceptions import AIInterpretationError

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                mock_interpreter = MagicMock()
                mock_interpreter.interpret.side_effect = AIInterpretationError("API failed")

                explanation = ExplanationResult(
                    word_weights=[("テスト", 0.1)],
                    class_name="Fake",
                    class_names=["Fake", "Real"],
                    probability=0.8,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    sentences_jp=["テンプレート説明"],
                    sentences_en=["Template explanation"],
                )

                result = PredictionResult(
                    text="テストテキスト",
                    predicted_class="Fake",
                    predicted_label=0,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    class_names=["Fake", "Real"],
                    explanation=explanation,
                    fallback_to_template=False,  # Must be False to raise error
                )

                # Patch where the import happens (inside the method)
                with patch('tokusan.ai_interpreter.GeminiInterpreter', return_value=mock_interpreter):
                    with pytest.raises(AIInterpretationError, match="API failed"):
                        _ = result.summary_jp

    def test_summary_jp_fallback_on_error(self):
        """Test that summary_jp falls back to template when fallback_to_template=True."""
        from tokusan.results import PredictionResult, ExplanationResult
        from tokusan.exceptions import AIInterpretationError

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                mock_interpreter = MagicMock()
                mock_interpreter.interpret.side_effect = Exception("API failed")

                explanation = ExplanationResult(
                    word_weights=[("テスト", 0.1)],
                    class_name="Fake",
                    class_names=["Fake", "Real"],
                    probability=0.8,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    sentences_jp=["テンプレート説明"],
                    sentences_en=["Template explanation"],
                )

                result = PredictionResult(
                    text="テストテキスト",
                    predicted_class="Fake",
                    predicted_label=0,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    class_names=["Fake", "Real"],
                    explanation=explanation,
                    use_ai=True,
                    fallback_to_template=True,  # Should fall back
                )

                with patch('tokusan.ai_interpreter.GeminiInterpreter', return_value=mock_interpreter):
                    summary = result.summary_jp
                    # Should get template, not raise error
                    assert "予測結果: Fake" in summary

    def test_use_ai_false_skips_ai(self):
        """Test that use_ai=False skips AI even with API key set."""
        from tokusan.results import PredictionResult, ExplanationResult

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            explanation = ExplanationResult(
                word_weights=[("テスト", 0.1)],
                class_name="Fake",
                class_names=["Fake", "Real"],
                probability=0.8,
                probabilities={"Fake": 0.8, "Real": 0.2},
                sentences_jp=["テンプレート説明"],
                sentences_en=["Template explanation"],
            )

            result = PredictionResult(
                text="テストテキスト",
                predicted_class="Fake",
                predicted_label=0,
                probabilities={"Fake": 0.8, "Real": 0.2},
                class_names=["Fake", "Real"],
                explanation=explanation,
                use_ai=False,  # Explicitly disable AI
            )

            # Should use template without calling AI
            summary = result.summary_jp
            assert "予測結果: Fake" in summary
            assert "テンプレート説明" in summary


class TestExplanationResultAiInterpretation:
    """Tests for AI interpretation in ExplanationResult."""

    def test_get_ai_interpretation_success(self):
        """Test successful AI interpretation from ExplanationResult."""
        from tokusan.results import ExplanationResult

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=True):
                mock_interpreter = MagicMock()
                mock_interpreter.interpret.return_value = "AI解釈です。"

                explanation = ExplanationResult(
                    word_weights=[("テスト", 0.1)],
                    class_name="Fake",
                    class_names=["Fake", "Real"],
                    probability=0.8,
                    probabilities={"Fake": 0.8, "Real": 0.2},
                    sentences_jp=["テンプレート説明"],
                    sentences_en=["Template explanation"],
                    original_text="テストテキスト",
                )

                with patch('tokusan.ai_interpreter.GeminiInterpreter', return_value=mock_interpreter):
                    result = explanation.get_ai_interpretation()
                    assert result == "AI解釈です。"

    def test_get_ai_interpretation_fallback(self):
        """Test fallback to template when AI unavailable."""
        from tokusan.results import ExplanationResult

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GEMINI_API_KEY', None)

            explanation = ExplanationResult(
                word_weights=[("テスト", 0.1)],
                class_name="Fake",
                class_names=["Fake", "Real"],
                probability=0.8,
                probabilities={"Fake": 0.8, "Real": 0.2},
                sentences_jp=["テンプレート説明"],
                sentences_en=["Template explanation"],
            )

            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=False):
                result = explanation.get_ai_interpretation(fallback_to_template=True)
                assert result == "テンプレート説明"

    def test_get_ai_interpretation_error_without_fallback(self):
        """Test that error is raised when AI unavailable and no fallback."""
        from tokusan.results import ExplanationResult
        from tokusan.exceptions import AIInterpretationError

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('GEMINI_API_KEY', None)

            explanation = ExplanationResult(
                word_weights=[("テスト", 0.1)],
                class_name="Fake",
                class_names=["Fake", "Real"],
                probability=0.8,
                probabilities={"Fake": 0.8, "Real": 0.2},
                sentences_jp=["テンプレート説明"],
                sentences_en=["Template explanation"],
            )

            with patch('tokusan.ai_interpreter._check_gemini_available', return_value=False):
                with pytest.raises(AIInterpretationError):
                    explanation.get_ai_interpretation(fallback_to_template=False)
