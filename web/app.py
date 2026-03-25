"""FastAPI web application for Tokusan text classifier."""

import re
from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from tokusan import JapaneseTextClassifier
from tokusan.ai_interpreter import is_ai_available, _check_gemini_available

from .auth import (
    COOKIE_NAME,
    _AuthRedirectException,
    create_session_token,
    get_current_user,
    hash_password,
    require_auth,
    verify_password,
)
from .db import (
    create_user,
    get_latest_user_model,
    get_user_by_username,
    init_db,
    save_user_model,
)
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
USER_DATA_DIR = BASE_DIR / "user_data"


# ---------------------------------------------------------------------------
# Exception handler for auth redirects
# ---------------------------------------------------------------------------
@app.exception_handler(_AuthRedirectException)
async def auth_redirect_handler(request: Request, exc: _AuthRedirectException):
    return exc.response


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    await init_db()
    USER_DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _error_response(request: Request, error: str, hint: str = "", status_code: int = 400):
    """Return an error partial HTML response."""
    return templates.TemplateResponse(
        request,
        "partials/error.html",
        {"error": error, "hint": hint},
        status_code=status_code,
    )


def _get_session_for_user(user: dict):
    """Get or create an in-memory session keyed by user ID."""
    session_key = f"user_{user['user_id']}"
    return app_state.get_or_create_session(session_key)


async def _try_load_saved_model(user: dict, session):
    """If the user has a saved model and the session has none, load it."""
    if session.classifier is not None:
        return

    model_record = await get_latest_user_model(user["user_id"])
    if model_record is None:
        return

    model_path = Path(model_record["model_path"])
    if not model_path.exists():
        return

    try:
        session.classifier = JapaneseTextClassifier.load(model_path)
        session.class_names = model_record["class_names"]
        session.classifier_type = model_record["classifier_type"]

        # Load training data if available
        data_path = model_record.get("training_data_path")
        if data_path and Path(data_path).exists():
            session.training_data = pd.read_csv(data_path)
    except Exception:
        pass  # If loading fails, user starts fresh


async def _save_model_for_user(user: dict, session):
    """Persist the trained model to disk and record in DB."""
    if session.classifier is None or not session.classifier.is_trained:
        return

    user_dir = USER_DATA_DIR / str(user["user_id"])
    user_dir.mkdir(parents=True, exist_ok=True)

    model_name = "latest"
    model_path = user_dir / "model_latest.joblib"
    session.classifier.save(model_path)

    data_path = None
    if session.training_data is not None:
        data_path = str(user_dir / "data_latest.csv")
        session.training_data.to_csv(data_path, index=False)

    await save_user_model(
        user_id=user["user_id"],
        name=model_name,
        model_path=str(model_path),
        training_data_path=data_path,
        class_names=session.class_names,
        classifier_type=session.classifier_type,
    )


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = get_current_user(request)
    if user is not None:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    db_user = await get_user_by_username(username)
    if db_user is None or not verify_password(password, db_user["password_hash"]):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "ユーザー名またはパスワードが正しくありません。", "username": username},
        )

    token = create_session_token(db_user["id"], db_user["username"])
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(COOKIE_NAME, token, httponly=True, max_age=86400)
    response.delete_cookie("session_id")  # Clean up old anonymous cookie
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    user = get_current_user(request)
    if user is not None:
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"error": None})


@app.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    # Validation
    if len(username.strip()) < 3:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "ユーザー名は3文字以上で入力してください。", "username": username},
        )

    if len(password) < 6:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "パスワードは6文字以上で入力してください。", "username": username},
        )

    if password != password_confirm:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "パスワードが一致しません。", "username": username},
        )

    # Check if username already taken
    existing = await get_user_by_username(username.strip())
    if existing is not None:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "このユーザー名は既に使用されています。", "username": username},
        )

    # Create user
    hashed = hash_password(password)
    user_id = await create_user(username.strip(), hashed)

    # Auto-login
    token = create_session_token(user_id, username.strip())
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(COOKIE_NAME, token, httponly=True, max_age=86400)
    return response


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    response.delete_cookie("session_id")
    return response


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page with all forms."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    # Auto-load saved model if session is empty
    await _try_load_saved_model(user, session)

    is_trained = session.classifier is not None and session.classifier.is_trained

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "session": session,
            "is_trained": is_trained,
            "user": user,
        },
    )


# ---------------------------------------------------------------------------
# API routes (all protected)
# ---------------------------------------------------------------------------
@app.post("/api/upload", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    """Upload and validate training CSV."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    if "text" not in df.columns or "label" not in df.columns:
        return _error_response(
            request,
            error="CSVには 'text' と 'label' の列が必要です。"
            f"検出された列：{', '.join(df.columns)}",
            hint="CSVファイルを修正して、もう一度アップロードしてください。",
        )

    session.training_data = df
    unique_labels = sorted(df["label"].unique())

    return templates.TemplateResponse(
        request,
        "partials/config_form.html",
        {
            "num_samples": len(df),
            "unique_labels": unique_labels,
            "suggested_classes": [f"Class_{l}" for l in unique_labels],
        },
    )


@app.post("/api/load-sample", response_class=HTMLResponse)
async def load_sample(request: Request, sample: str = Form("fakenews")):
    """Load a bundled sample dataset into the session."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    sample_config = {
        "fakenews": {"file": "fakenews_sample.csv", "suggested_classes": None},
        "ramen": {"file": "ramen_review_sample.csv", "suggested_classes": None},
        "wrime": {
            "file": "wrime_sample.csv",
            "suggested_classes": ["喜び", "悲しみ", "期待", "驚き", "怒り", "恐れ", "嫌悪", "信頼"],
        },
        "spam": {
            "file": "spam_sample.csv",
            "suggested_classes": ["通常メール", "迷惑メール"],
        },
    }
    config = sample_config.get(sample)
    if config is None:
        return templates.TemplateResponse(
            request,
            "partials/error.html",
            {"error": f"不明なサンプル名: {sample}", "hint": ""},
            status_code=400,
        )

    sample_path = SAMPLES_DIR / config["file"]
    if not sample_path.exists():
        return templates.TemplateResponse(
            request,
            "partials/error.html",
            {
                "error": "サンプルデータファイルが見つかりませんでした。",
                "hint": "管理者に連絡してください。",
            },
            status_code=500,
        )

    df = pd.read_csv(sample_path)
    session.training_data = df

    unique_labels = sorted(df["label"].unique())
    preset = config["suggested_classes"]
    suggested = preset if preset is not None else [f"Class_{l}" for l in unique_labels]
    return templates.TemplateResponse(
        request,
        "partials/config_form.html",
        {
            "num_samples": len(df),
            "unique_labels": unique_labels,
            "suggested_classes": suggested,
        },
    )


@app.post("/api/train", response_class=HTMLResponse)
async def train_model(
    request: Request,
    class_names: str = Form(...),
    classifier_type: str = Form("logistic_regression"),
):
    """Train the classifier on uploaded data."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.training_data is None:
        return _error_response(
            request,
            error="学習データがアップロードされていません。まずCSVをアップロードしてください。",
            hint="ステップ1からCSVをアップロードしてください。",
        )

    names = [n.strip() for n in re.split(r"[,、]", class_names) if n.strip()]
    if len(names) < 2:
        return _error_response(
            request,
            error="クラス名を2つ以上入力してください。",
            hint="カンマ区切りで2つ以上のクラス名を入力してください。",
        )

    clf = JapaneseTextClassifier(
        class_names=names, classifier_type=classifier_type
    )

    df = session.training_data
    result = clf.train(df["text"].tolist(), df["label"].tolist())

    session.classifier = clf
    session.training_result = result
    session.class_names = names
    session.classifier_type = classifier_type

    # Auto-save model for the user
    await _save_model_for_user(user, session)

    return templates.TemplateResponse(
        request,
        "partials/training_result.html",
        {"result": result},
    )


@app.post("/api/predict", response_class=HTMLResponse)
async def predict_text(
    request: Request,
    text: str = Form(...),
):
    """Classify text and return prediction with LIME explanation and AI interpretation."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.classifier is None or not session.classifier.is_trained:
        return _error_response(
            request,
            error="学習済みモデルがありません。データをアップロードして学習を行ってください。",
            hint="ステップ1からやり直してください。",
        )

    ai_available = is_ai_available()
    if not ai_available:
        if not _check_gemini_available():
            ai_fallback_reason = "google-genai パッケージがインストールされていません"
        else:
            ai_fallback_reason = "GEMINI_API_KEY が設定されていません"
    else:
        ai_fallback_reason = None

    result = session.classifier.predict(
        text=text,
        explain=True,
        use_ai=True,
        fallback_to_template=True,
    )

    max_weight = 1.0
    if result.explanation and result.explanation.word_weights:
        max_weight = max(abs(w) for _, w in result.explanation.word_weights)

    return templates.TemplateResponse(
        request,
        "partials/prediction_result.html",
        {
            "result": result,
            "text": text,
            "max_weight": max_weight,
            "ai_available": ai_available,
            "ai_fallback_reason": ai_fallback_reason,
        },
    )
