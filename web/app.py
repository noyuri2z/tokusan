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
    delete_user_model,
    get_latest_user_model,
    get_user_by_username,
    get_user_model,
    init_db,
    list_user_models,
    save_user_model,
)
from .sample_config import MODEL_DESCRIPTIONS, SAMPLE_DATASETS
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

    model_name = session.project_name or "latest"
    safe_name = re.sub(r'[^\w\-]', '_', model_name)
    model_path = user_dir / f"model_{safe_name}.joblib"
    session.classifier.save(model_path)

    data_path = None
    if session.training_data is not None:
        data_path = str(user_dir / f"data_{safe_name}.csv")
        session.training_data.to_csv(data_path, index=False)

    await save_user_model(
        user_id=user["user_id"],
        name=model_name,
        model_path=str(model_path),
        training_data_path=data_path,
        class_names=session.class_names,
        classifier_type=session.classifier_type,
    )


# Classifier type Japanese names for dashboard display
_CLASSIFIER_NAMES = {k: v["name"] for k, v in MODEL_DESCRIPTIONS.items()}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def welcome_or_redirect(request: Request):
    """Welcome page for unauthenticated users, redirect for authenticated."""
    user = get_current_user(request)
    if user is not None:
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(
        request,
        "welcome.html",
        {
            "active_tab": "login",
            "login_error": None,
            "register_error": None,
            "username": "",
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_redirect(request: Request):
    """Backward compat: redirect to welcome page."""
    return RedirectResponse("/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_redirect(request: Request):
    """Backward compat: redirect to welcome page."""
    return RedirectResponse("/", status_code=302)


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    db_user = await get_user_by_username(username)
    if db_user is None or not verify_password(password, db_user["password_hash"]):
        return templates.TemplateResponse(
            request,
            "welcome.html",
            {
                "active_tab": "login",
                "login_error": "ユーザー名またはパスワードが正しくありません。",
                "register_error": None,
                "username": username,
            },
        )

    token = create_session_token(db_user["id"], db_user["username"])
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(COOKIE_NAME, token, httponly=True, max_age=86400 * 30)
    response.delete_cookie("session_id")
    return response


@app.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    # Validation
    error = None
    if len(username.strip()) < 3:
        error = "ユーザー名は3文字以上で入力してください。"
    elif len(password) < 6:
        error = "パスワードは6文字以上で入力してください。"
    elif password != password_confirm:
        error = "パスワードが一致しません。"
    else:
        existing = await get_user_by_username(username.strip())
        if existing is not None:
            error = "このユーザー名は既に使用されています。"

    if error:
        return templates.TemplateResponse(
            request,
            "welcome.html",
            {
                "active_tab": "register",
                "login_error": None,
                "register_error": error,
                "username": username,
            },
        )

    # Create user
    hashed = hash_password(password)
    user_id = await create_user(username.strip(), hashed)

    # Auto-login
    token = create_session_token(user_id, username.strip())
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie(COOKIE_NAME, token, httponly=True, max_age=86400 * 30)
    return response


@app.get("/logout")
async def logout(request: Request):
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    response.delete_cookie("session_id")
    return response


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = require_auth(request)
    models = await list_user_models(user["user_id"])
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "models": models,
            "model_names": _CLASSIFIER_NAMES,
        },
    )


@app.post("/dashboard/delete/{model_id}", response_class=HTMLResponse)
async def dashboard_delete_model(request: Request, model_id: int):
    user = require_auth(request)
    await delete_user_model(user["user_id"], model_id)
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/dashboard/load/{model_id}", response_class=HTMLResponse)
async def dashboard_load_model(request: Request, model_id: int):
    user = require_auth(request)
    session = _get_session_for_user(user)

    model_record = await get_user_model(user["user_id"], model_id)
    if model_record is None:
        return RedirectResponse("/dashboard", status_code=302)

    model_path = Path(model_record["model_path"])
    if not model_path.exists():
        return RedirectResponse("/dashboard", status_code=302)

    try:
        session.classifier = JapaneseTextClassifier.load(model_path)
        session.class_names = model_record["class_names"]
        session.classifier_type = model_record["classifier_type"]

        data_path = model_record.get("training_data_path")
        if data_path and Path(data_path).exists():
            session.training_data = pd.read_csv(data_path)

        # Re-run training result from loaded model for display
        if session.training_data is not None:
            df = session.training_data
            clf = JapaneseTextClassifier(
                class_names=session.class_names,
                classifier_type=session.classifier_type,
            )
            result = clf.train(df["text"].tolist(), df["label"].tolist())
            session.classifier = clf
            session.training_result = result
    except Exception:
        return RedirectResponse("/dashboard", status_code=302)

    return RedirectResponse("/project/results", status_code=302)


# ---------------------------------------------------------------------------
# Project: Data page
# ---------------------------------------------------------------------------
@app.get("/project/data", response_class=HTMLResponse)
async def project_data(request: Request):
    user = require_auth(request)
    session = _get_session_for_user(user)

    # Reset session for new project
    session.training_data = None
    session.classifier = None
    session.training_result = None
    session.class_names = []
    session.classifier_type = "logistic_regression"
    session.current_sample = None
    session.project_name = None

    return templates.TemplateResponse(
        request,
        "project/data.html",
        {
            "user": user,
            "samples": SAMPLE_DATASETS,
            "current_step": 1,
        },
    )


@app.get("/api/sample-info/{sample}", response_class=HTMLResponse)
async def sample_info(request: Request, sample: str):
    """Return sample dataset info partial."""
    user = require_auth(request)
    config = SAMPLE_DATASETS.get(sample)
    if config is None:
        return _error_response(request, f"不明なサンプル名: {sample}")
    return templates.TemplateResponse(
        request,
        "partials/sample_info.html",
        {"sample": config},
    )


@app.post("/project/data/confirm", response_class=HTMLResponse)
async def project_data_confirm(
    request: Request,
    project_name: str = Form(""),
    class_names: str = Form(""),
):
    """Validate data exists and proceed to model selection."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.training_data is None:
        return RedirectResponse("/project/data", status_code=302)

    session.project_name = project_name.strip() or None

    # Save class names from form
    names = [n.strip() for n in re.split(r"[,、]", class_names) if n.strip()]
    if names:
        session.class_names = names

    return RedirectResponse("/project/model", status_code=302)


# ---------------------------------------------------------------------------
# Project: Model page
# ---------------------------------------------------------------------------
@app.get("/project/model", response_class=HTMLResponse)
async def project_model(request: Request):
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.training_data is None:
        return RedirectResponse("/project/data", status_code=302)

    df = session.training_data
    unique_labels = sorted(df["label"].unique())

    return templates.TemplateResponse(
        request,
        "project/model.html",
        {
            "user": user,
            "model_descriptions": MODEL_DESCRIPTIONS,
            "num_samples": len(df),
            "num_classes": len(unique_labels),
            "current_step": 2,
        },
    )


@app.get("/project/model/{model_key}", response_class=HTMLResponse)
async def project_model_detail(request: Request, model_key: str):
    """Show detailed model explanation page."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.training_data is None:
        return RedirectResponse("/project/data", status_code=302)

    model = MODEL_DESCRIPTIONS.get(model_key)
    if model is None:
        return RedirectResponse("/project/model", status_code=302)

    return templates.TemplateResponse(
        request,
        "project/model_detail.html",
        {
            "user": user,
            "model_key": model_key,
            "model": model,
            "current_step": 2,
        },
    )


# ---------------------------------------------------------------------------
# Project: Results page
# ---------------------------------------------------------------------------
@app.get("/project/results", response_class=HTMLResponse)
async def project_results(request: Request):
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.training_result is None:
        return RedirectResponse("/dashboard", status_code=302)

    return templates.TemplateResponse(
        request,
        "project/results.html",
        {
            "user": user,
            "result": session.training_result,
            "current_step": 3,
        },
    )


# ---------------------------------------------------------------------------
# Project: Play page
# ---------------------------------------------------------------------------
@app.get("/project/play", response_class=HTMLResponse)
async def project_play(request: Request):
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.classifier is None or not session.classifier.is_trained:
        return RedirectResponse("/dashboard", status_code=302)

    return templates.TemplateResponse(
        request,
        "project/play.html",
        {
            "user": user,
            "current_step": 4,
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
    session.current_sample = None
    unique_labels = sorted(df["label"].unique())

    return templates.TemplateResponse(
        request,
        "partials/data_summary.html",
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

    config = SAMPLE_DATASETS.get(sample)
    if config is None:
        return _error_response(request, f"不明なサンプル名: {sample}")

    sample_path = SAMPLES_DIR / config["file"]
    if not sample_path.exists():
        return _error_response(
            request,
            error="サンプルデータファイルが見つかりませんでした。",
            hint="管理者に連絡してください。",
            status_code=500,
        )

    df = pd.read_csv(sample_path)
    session.training_data = df
    session.current_sample = sample

    unique_labels = sorted(df["label"].unique())
    preset = config["suggested_classes"]
    suggested = preset if preset is not None else [f"Class_{l}" for l in unique_labels]

    return templates.TemplateResponse(
        request,
        "partials/data_summary.html",
        {
            "num_samples": len(df),
            "unique_labels": unique_labels,
            "suggested_classes": suggested,
        },
    )


@app.post("/api/train", response_class=HTMLResponse)
async def train_model(
    request: Request,
    class_names: str = Form(""),
    classifier_type: str = Form("logistic_regression"),
):
    """Train the classifier on uploaded data."""
    user = require_auth(request)
    session = _get_session_for_user(user)

    if session.training_data is None:
        return _error_response(
            request,
            error="学習データがアップロードされていません。まずCSVをアップロードしてください。",
            hint="データ選択ページからCSVをアップロードしてください。",
        )

    names = [n.strip() for n in re.split(r"[,、]", class_names) if n.strip()]
    # Fallback to session class_names if not provided in form
    if not names and session.class_names:
        names = session.class_names
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

    # Return HX-Redirect to results page
    response = HTMLResponse(content="", status_code=200)
    response.headers["HX-Redirect"] = "/project/results"
    return response


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
            hint="データ選択ページからやり直してください。",
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
