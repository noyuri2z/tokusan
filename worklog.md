# Worklog

## 2026-02-26: User Feedback Feature Improvements

**Files changed:**
- `web/templates/index.html`
- `web/templates/partials/predict_form.html`
- `web/templates/partials/prediction_result.html`
- `web/app.py`
- `web/samples/fakenews_sample.csv` (new)
- `tokusan/japanese/splitters.py`
- `tests/test_tokusan.py`

**Changes:**

1. **Drag-and-drop upload:** Replaced the plain file input in Step 1 with a styled drop zone. Users can drop a CSV directly onto the zone or click to browse. Drag-and-drop uses a `DataTransfer` object to assign the file to the input and calls `htmx.trigger` to submit the existing form.

2. **Sample data tab:** Step 1 now has two tabs: "サンプルデータを使用" (default) and "CSVをアップロード". The sample tab loads a pre-bundled 501-row fakenews dataset (stratified: 167 rows per class, 3 classes) via a new `POST /api/load-sample` endpoint. The sample CSV is stored at `web/samples/fakenews_sample.csv` (columns: `text`, `label`).

3. **Removed checkboxes:** The "説明を表示" and "Gemini AIの説明をオンにする" checkboxes were removed from the prediction form. LIME explanation and AI interpretation are now always enabled. The `/api/predict` endpoint checks Gemini availability before calling `predict()` and passes `ai_available` and `ai_fallback_reason` to the template. When AI is unavailable, a yellow banner is shown above the explanation with the specific reason (missing package or missing API key).

4. **Tokenization fixes (verified and corrected):**
   - `tokusan/japanese/splitters.py`: Fixed SudachiPy path to filter whitespace morphemes from output. The fallback path already excluded whitespace, but the SudachiPy path did not, allowing spaces to appear as vocabulary tokens in `IndexedString`.
   - The `doc_size == 0` guard in `_data_labels_distances` (added in the prior session) now works correctly once whitespace is filtered.
   - Corrected the worklog diagnosis from the prior session: "金" is filtered because it is in `JAPANESE_STOPWORDS` (as a day-of-week abbreviation), not due to a `len(word) < 2` filter. The length filter removal is still correct for other single-char kanji like "愛" that are not stopwords.
   - Added `TestTokenizationFixes` class to `tests/test_tokusan.py` with 3 regression tests.

**Verification:**
- `pytest tests/test_tokusan.py::TestTokenizationFixes -v` → 3 passed

## 2026-02-26: Improve "low >= high" error message + fix single-char token filtering

**Files changed:** `tokusan/explainer.py`, `tokusan/exceptions.py`

**Problem:** When users submitted text containing only stopwords or punctuation (e.g., `"は が で"`), the UI showed the cryptic error `分類に失敗しました：low >= high`. This originated from `numpy.random.randint(1, 1)` in `_data_labels_distances` when LIME's `doc_size` was 0 (no valid tokens remained after filtering).

Additionally, the `IndexedString.__init__` was incorrectly discarding all single-character tokens regardless of whether they were stopwords or punctuation. Meaningful single-character kanji (e.g., `"金"`, `"愛"`) were being silently dropped.

**Changes:**

- `tokusan/explainer.py` — `IndexedString.__init__`: Removed the `len(word) < 2` filter; single non-stopword, non-punctuation characters are now kept as valid tokens.
- `tokusan/explainer.py` — `_data_labels_distances`: Added a `doc_size == 0` guard that raises `ExplanationError` with a Japanese-language message before the numpy call fails.
- `tokusan/explainer.py` — Added `from .exceptions import ExplanationError` import.
- `tokusan/exceptions.py` — Updated `ExplanationError` docstring to document the empty-token failure condition.

**Verification:**
- Submit only stopwords/particles (e.g., `"は が で を"`) → descriptive error shown in UI
- Submit a single meaningful kanji (e.g., `"金"`) → classifies successfully (no longer filtered)
- Submit a normal sentence → classifies successfully as before

## 2026-02-23: Revert SudachiPy SplitMode from A back to C

**File changed:** `tokusan/japanese/tokenizers.py`

**Problem:** Commit `32e90ab` changed the SudachiPy split mode from `SplitMode.C` (longest units) to `SplitMode.A` (shortest units). This caused overly aggressive tokenization — compound words like "委員会" were split into "委員/会", losing semantic meaning needed for classification and LIME explanations.

**Changes:**
- Line 39: Updated comment to say "SplitMode.C (long unit)" instead of "SplitMode.A (shortest morphological units)"
- Line 44: Changed `SplitMode.A` to `SplitMode.C`

**Verification:**
- `python -c "from tokusan.japanese.splitters import split; print(split('選挙管理委員会'))"` outputs `['選挙管理委員会']` (kept as single token)
- `pytest`: 121 passed, 6 failed (all 6 failures are pre-existing and unrelated to this change)
