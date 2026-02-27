"""FastAPI web application for Tokusan text classifier."""

import re
from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tokusan import JapaneseTextClassifier
from tokusan.ai_interpreter import is_ai_available, _check_gemini_available

from .state import app_state

app = FastAPI(
    title="Tokusan Text Classifier",
    description="Japanese text classification with AI explanations",
    version="0.1.0",
)

# Configure templates and sample data path
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
SAMPLES_DIR = BASE_DIR / "samples"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with all forms."""
    session_id = request.cookies.get("session_id")
    session = app_state.get_or_create_session(session_id)

    is_trained = session.classifier is not None and session.classifier.is_trained

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session": session,
            "is_trained": is_trained,
        },
    )
    response.set_cookie("session_id", session.session_id, httponly=True)
    return response


@app.post("/api/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """Upload and validate training CSV."""
    session_id = request.cookies.get("session_id")
    session = app_state.get_or_create_session(session_id)

    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")))

        # Validate required columns
        if "text" not in df.columns or "label" not in df.columns:
            return templates.TemplateResponse(
                "partials/error.html",
                {
                    "request": request,
                    "error": "CSVには 'text' と 'label' の列が必要です。"
                    f"検出された列：{', '.join(df.columns)}",
                    "hint": "CSVファイルを修正して、もう一度アップロードしてください。",
                },
                status_code=400,
            )

        # Store in session
        session.training_data = df

        # Get unique labels
        unique_labels = sorted(df["label"].unique())

        response = templates.TemplateResponse(
            "partials/config_form.html",
            {
                "request": request,
                "num_samples": len(df),
                "unique_labels": unique_labels,
                "suggested_classes": [f"Class_{l}" for l in unique_labels],
            },
        )
        response.set_cookie("session_id", session.session_id, httponly=True)
        return response

    except Exception as e:
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "error": f"CSVの読み込みに失敗しました：{str(e)}",
                "hint": "ファイル形式を確認して、もう一度お試しください。",
            },
            status_code=400,
        )


@app.post("/api/load-sample", response_class=HTMLResponse)
async def load_sample(request: Request, sample: str = Form("fakenews")):
    """Load a bundled sample dataset into the session."""
    session_id = request.cookies.get("session_id")
    session = app_state.get_or_create_session(session_id)

    sample_files = {
        "fakenews": "fakenews_sample.csv",
        "ramen": "ramen_review_sample.csv",
    }
    filename = sample_files.get(sample)
    if filename is None:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "error": f"不明なサンプル名: {sample}", "hint": ""},
            status_code=400,
        )

    sample_path = SAMPLES_DIR / filename
    if not sample_path.exists():
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "error": "サンプルデータファイルが見つかりませんでした。",
                "hint": "管理者に連絡してください。",
            },
            status_code=500,
        )

    df = pd.read_csv(sample_path)
    session.training_data = df

    unique_labels = sorted(df["label"].unique())
    response = templates.TemplateResponse(
        "partials/config_form.html",
        {
            "request": request,
            "num_samples": len(df),
            "unique_labels": unique_labels,
            "suggested_classes": [f"Class_{l}" for l in unique_labels],
        },
    )
    response.set_cookie("session_id", session.session_id, httponly=True)
    return response


@app.post("/api/train", response_class=HTMLResponse)
async def train_model(
    request: Request,
    class_names: str = Form(...),
    classifier_type: str = Form("logistic_regression"),
):
    """Train the classifier on uploaded data."""
    session_id = request.cookies.get("session_id")
    session = app_state.get_or_create_session(session_id)

    if session.training_data is None:
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "error": "学習データがアップロードされていません。まずCSVをアップロードしてください。",
                "hint": "ステップ1からCSVをアップロードしてください。",
            },
            status_code=400,
        )

    try:
        # Parse class names
        names = [n.strip() for n in re.split(r"[,、]", class_names) if n.strip()]
        if len(names) < 2:
            return templates.TemplateResponse(
                "partials/error.html",
                {
                    "request": request,
                    "error": "クラス名を2つ以上入力してください。",
                    "hint": "カンマ区切りで2つ以上のクラス名を入力してください。",
                },
                status_code=400,
            )

        # Create and train classifier
        clf = JapaneseTextClassifier(
            class_names=names, classifier_type=classifier_type
        )

        df = session.training_data
        result = clf.train(df["text"].tolist(), df["label"].tolist())

        # Store in session
        session.classifier = clf
        session.training_result = result
        session.class_names = names
        session.classifier_type = classifier_type

        response = templates.TemplateResponse(
            "partials/training_result.html",
            {
                "request": request,
                "result": result,
            },
        )
        response.set_cookie("session_id", session.session_id, httponly=True)
        return response

    except Exception as e:
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "error": f"学習に失敗しました：{str(e)}",
                "hint": "設定を確認して、もう一度「モデルを学習」を押してください。",
            },
            status_code=500,
        )


@app.post("/api/predict", response_class=HTMLResponse)
async def predict_text(
    request: Request,
    text: str = Form(...),
):
    """Classify text and return prediction with LIME explanation and AI interpretation."""
    session_id = request.cookies.get("session_id")
    session = app_state.get_or_create_session(session_id)

    if session.classifier is None or not session.classifier.is_trained:
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "error": "学習済みモデルがありません。データをアップロードして学習を行ってください。",
                "hint": "ステップ1からやり直してください。",
            },
            status_code=400,
        )

    # Determine AI availability and reason for unavailability
    ai_available = is_ai_available()
    if not ai_available:
        if not _check_gemini_available():
            ai_fallback_reason = "google-genai パッケージがインストールされていません"
        else:
            ai_fallback_reason = "GEMINI_API_KEY が設定されていません"
    else:
        ai_fallback_reason = None

    try:
        result = session.classifier.predict(
            text=text,
            explain=True,
            use_ai=True,
            fallback_to_template=True,
        )

        # Calculate max weight for visualization
        max_weight = 1.0
        if result.explanation and result.explanation.word_weights:
            max_weight = max(abs(w) for _, w in result.explanation.word_weights)

        return templates.TemplateResponse(
            "partials/prediction_result.html",
            {
                "request": request,
                "result": result,
                "text": text,
                "max_weight": max_weight,
                "ai_available": ai_available,
                "ai_fallback_reason": ai_fallback_reason,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "partials/error.html",
            {
                "request": request,
                "error": f"分類に失敗しました：{str(e)}",
                "hint": "テキストを確認して、もう一度お試しください。",
            },
            status_code=500,
        )
