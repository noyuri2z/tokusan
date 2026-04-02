


# Tokusan マルチページ化リデザイン計画

## Context

現在のTokusanは単一ページ（`index.html`）にHTMXで4ステップ（アップロード→設定→学習→予測）を詰め込んでいる。これを6つの独立ページに分割し、各ステップに教育的>な説明を加えて、非技術者向けのガイド付き体験にする。すべてのUI文言は日本語。

## 新しいユーザージャーニー

```
/ (Welcome) → /dashboard → /project/data → /project/model → /project/results → /project/play
```

---

## Phase 1: 設定ファイル作成

### 1. `/web/sample_config.py` — 新規作成

サンプルデータセットの説明とモデル説明を格納する辞書を定義：

- **`SAMPLE_DATASETS`** — 4つのサンプル（fakenews, ramen, wrime, spam）それぞれに：
  - `title`, `file`, `sample_count`
  - `suggested_classes` — クラス名リスト
  - `description` — データセットの背景説明（日本語）
  - `analysis_hint` — このデータでできる分析の説明（日本語）

- **`MODEL_DESCRIPTIONS`** — 4モデル（logistic_regression, random_forest, linear_svc, naive_bayes）それぞれに：
  - `name` — 日本語名
  - `intuition` — 非技術者向け直感的説明
  - `formula_hint` — 数学的な仕組み
  - `suitable_for` — 適したユースケース

### 2. `/web/state.py` — 修正

`SessionState` に `current_sample: Optional[str] = None` フィールド追加。

---

## Phase 2: ウェルカムページ（ページ1）

### 3. `/web/templates/welcome.html` — 新規作成

- `base.html` を継承
- ヒーローセクション：「Tokusanで自然言語処理を学ぼう！」
- サブタイトル：アプリの説明
- 2つのカード横並び：ログイン（左）・新規登録（右）
- タブ切替でログイン/登録フォームを切替
- エラー表示はフォーム内に表示（`active_tab` + `login_error`/`register_error` テンプレート変数）

### 4. `/web/app.py` — 認証ルート修正

- `GET /` — 未認証→`welcome.html`、認証済→`/dashboard`にリダイレクト
- `POST /login` — 失敗→`welcome.html`（`active_tab="login"`）、成功→`/dashboard`にリダイレクト
- `POST /register` — 失敗→`welcome.html`（`active_tab="register"`）、成功→`/dashboard`にリダイレクト
- `GET /logout` — `/`にリダイレクト
- `GET /login`, `GET /register` — `/`にリダイレクト（後方互換）

---

## Phase 3: ダッシュボード（ページ2）

### 5. `/web/templates/dashboard.html` — 新規作成

- ユーザー名表示 + ログアウトリンク
- 「新しいプロジェクトを作成」ボタン → `/project/data`
- 保存済みモデル一覧（`list_user_models`）：
  - モデル名、分類器タイプ（日本語名）、クラス名、作成日
  - 「読み込む」ボタン、「削除」ボタン
- モデルがない場合の空状態メッセージ

### 6. `/web/app.py` — ダッシュボードルート追加

- `GET /dashboard` — `list_user_models(user_id)` → `dashboard.html`
- `POST /dashboard/delete/{model_id}` — モデル削除、ダッシュボードにリダイレクト
- `GET /dashboard/load/{model_id}` — モデルをセッションに読み込み → `/project/results`にリダイレクト
- `db.py` から `list_user_models`, `get_user_model`, `delete_user_model` をインポート追加

---
## Phase 4: データアップロードページ（ページ3）

### 7. `/web/templates/project/data.html` — 新規作成

- 進捗インジケーター（ステップ1/4）
- 2タブ：「サンプルデータ」/「CSVアップロード」
- サンプルデータ：4つのカード。選択時にHTMXで説明パネルを読み込み
  - 背景説明（`description`）
  - クラス名（`suggested_classes`）
  - 分析の説明（`analysis_hint`）
- CSVアップロード：ドラッグ＆ドロップ（既存JS流用）
- 情報パネル（`#data-info`）：データ読み込み後に表示
- 「次へ」ボタン → `/project/data/confirm` にPOST

### 8. `/web/templates/partials/sample_info.html` — 新規作成

サンプルデータセットの説明表示パーシャル。

### 9. `/web/templates/partials/data_summary.html` — 新規作成

アップロードされたデータの概要表示パーシャル。

### 10. `/web/app.py` — データルート追加

- `GET /project/data` — セッションリセット（新プロジェクト用） → `project/data.html`
- `GET /api/sample-info/{sample}` — `partials/sample_info.html` を返す
- `POST /project/data/confirm` — データ存在確認 + クラス名保存 → `/project/model`にリダイレクト
- 既存の `POST /api/upload`、`POST /api/load-sample` — レスポンスを `partials/data_summary.html` に変更

---

## Phase 5: モデル選択ページ（ページ4）

### 11. `/web/templates/project/model.html` — 新規作成

- 進捗インジケーター（ステップ2/4）
- データ概要表示（サンプル名、行数、クラス数）
- 4モデルカード（2×2グリッド）：
  - 日本語名、直感的説明、数式ヒント（折りたたみ）、適した分析
  - ラジオボタンで選択
- クラス名入力フィールド（セッションデータからプリフィル）
- 「学習を開始」ボタン → HTMX POST `/api/train`
  - ローディングスピナー表示
  - 成功時：`HX-Redirect: /project/results`

### 12. `/web/app.py` — モデルルート追加

- `GET /project/model` — データ存在チェック → `project/model.html`（`MODEL_DESCRIPTIONS`渡す）
- `POST /api/train` 修正 — 成功時に `HX-Redirect: /project/results` ヘッダーを返す

---

## Phase 6: 結果ページ（ページ5）+ プレイページ（ページ6）

### 13. `/web/templates/project/results.html` — 新規作成

- 進捗インジケーター（ステップ3/4）
- 学習メトリクス：正解率、学習/テストサイズ、クラス別精度表
- **正解率 < 50% の場合**：赤いエラーバナー「警告：正解率が50%未満です。このデータはこの分析には適していない可能性があります。」
- 2つのボタン：
  - 「新しいデータで試す」→ `/project/play`
  - 「ダッシュボードに戻る」→ `/dashboard`


### 14. `/web/templates/project/play.html` — 新規作成

- 進捗インジケーター（ステップ4/4）
- テキスト入力エリア
- 「分類する」ボタン → HTMX POST `/api/predict`
- 結果セクション：既存の `partials/prediction_result.html` を再利用（確率、LIME、AI解釈）
- ナビゲーションリンク：「結果ページに戻る」「ダッシュボードに戻る」

### 15. `/web/app.py` — 結果/プレイルート追加

- `GET /project/results` — `training_result` 存在チェック → `project/results.html`
- `GET /project/play` — 学習済みモデル存在チェック → `project/play.html`

---

## Phase 7: base.html更新 + クリーンアップ

### 16. `/web/templates/base.html` — 修正

- 認証済みの場合にステップ進捗バーを表示（データ → モデル → 結果 → 予測）
- `{% block page_header %}` 追加
- ヘッダーにダッシュボードリンク追加

### 17. 不要テンプレート削除

- `login.html` → `welcome.html` に統合済
- `register.html` → `welcome.html` に統合済
- `index.html` → マルチページに置換済
- `partials/config_form.html` → 機能が `project/model.html` に移動
- `partials/predict_form.html` → 機能が `project/play.html` に移動

---
---

## 変更対象ファイルまとめ

| ファイル | 操作 |
|---------|------|
| `/web/sample_config.py` | 新規作成 |
| `/web/state.py` | `current_sample` フィールド追加 |
| `/web/app.py` | ルート大幅変更（既存修正 + 新規追加） |
| `/web/templates/welcome.html` | 新規作成 |
| `/web/templates/dashboard.html` | 新規作成 |
| `/web/templates/project/data.html` | 新規作成 |
| `/web/templates/project/model.html` | 新規作成 |
| `/web/templates/project/results.html` | 新規作成 |
| `/web/templates/project/play.html` | 新規作成 |
| `/web/templates/partials/sample_info.html` | 新規作成 |
| `/web/templates/partials/data_summary.html` | 新規作成 |
| `/web/templates/base.html` | ステップ進捗バー追加 |
| `/web/templates/login.html` | 削除 |
| `/web/templates/register.html` | 削除 |
| `/web/templates/index.html` | 削除 |
| `/web/templates/partials/config_form.html` | 削除 |
| `/web/templates/partials/predict_form.html` | 削除 |

## 既存コードの再利用
- `web/db.py:98` `list_user_models()` — ダッシュボードで使用（既存だが未使用）
- `web/db.py:119` `get_user_model()` — モデル読み込みで使用（既存だが未使用）
- `web/db.py:161` `delete_user_model()` — モデル削除で使用（既存だが未使用）
- `web/templates/partials/prediction_result.html` — プレイページでそのまま再利用
- `web/templates/partials/error.html` — エラー表示でそのまま再利用
- 既存のドラッグ＆ドロップJS — データアップロードページで流用

## 検証方法

1. `python run_web.py` でサーバー起動
2. `/` にアクセス → ウェルカムページ表示確認
3. 新規登録 → `/dashboard` にリダイレクト確認
4. 「新しいプロジェクト作成」→ データアップロードページ確認
5. サンプルデータ選択 → 説明表示確認、「次へ」→ モデル選択ページ確認
6. モデル選択 + 「学習を開始」→ 学習実行 → 結果ページにリダイレクト確認
7. 正解率 < 50% の場合の赤いエラーバナー確認
8. 「新しいデータで試す」→ プレイページでテキスト入力 → 分類結果 + AI解釈表示確認
9. ダッシュボードに戻り、保存済みモデル読み込み確認