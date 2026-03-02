# 出力ファイル名規則統一 + revision-ops リネーム 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 回答支援AIと運用保守効率化AI（改定影響調査）の出力ファイル名規則を統一し、コード名・ディレクトリ名・ドキュメントを正式名称に合わせてリネームする。

**Architecture:** `OutputHandler` に `app_prefix` パラメータを追加してアプリ別サブディレクトリに出力を分離。パラメータ情報はファイル名から削除し Excel 内 Metadata シートに移動。改定影響調査アプリのディレクトリを `revision-eval` → `revision-ops` にリネームし、ソースファイル名も一新。

**Tech Stack:** Python, Streamlit, pandas, xlsxwriter, pytest

---

## 命名規則リファレンス

### 英語名対応表

| 日本語 | 英語 (コード/ディレクトリ) | 短縮 (出力ファイル/フォルダ) |
|--------|--------------------------|----------------------------|
| 回答支援AI（類似回答検索） | answer-support | `answer` |
| 運用保守効率化AI（改定影響調査） | **revision-ops** | `rev` |
| 評価モード | evaluation | `eval` |
| 影響調査モード | impact_analysis | `impact` |

### 出力ファイル命名規則

```
{app}_{mode}_{YYYYMMDD_HHMMSS}.xlsx
```

### 出力ディレクトリ構造（変更後）

```
data/output/latest/
├── answer/                                    # 回答支援AI
│   ├── answer_chat_20260302_143025.xlsx
│   ├── answer_batch_20260302_143025.xlsx
│   ├── answer_multi_stage_20260302_143025.xlsx
│   └── answer_dual_provider_20260302_143025.xlsx
└── rev/                                       # 運用保守効率化AI
    ├── rev_eval_chat_20260302_143025.xlsx
    ├── rev_eval_batch_20260302_143025.xlsx
    └── rev_impact_chat_20260302_143025.xlsx
```

### ディレクトリ・ファイル リネーム一覧

| 変更前 | 変更後 |
|--------|--------|
| `apps/revision-eval/` | `apps/revision-ops/` |
| `apps/revision-ops/evaluate_revisions.py` | `apps/revision-ops/run_eval.py` |
| `apps/revision-ops/ui/eval_ui.py` | `apps/revision-ops/ui/ops_ui.py` |
| `tests/unit/test_evaluate_revisions_cache.py` | `tests/unit/test_run_eval_cache.py` |
| `docs/REVISION_EVALUATION.md` | `docs/REVISION_OPS.md` |

---

## Task 1: git mv でディレクトリ・ファイルリネーム

**Files:**
- Rename: `apps/revision-eval/` → `apps/revision-ops/`
- Rename: `apps/revision-ops/evaluate_revisions.py` → `apps/revision-ops/run_eval.py`
- Rename: `apps/revision-ops/ui/eval_ui.py` → `apps/revision-ops/ui/ops_ui.py`
- Rename: `tests/unit/test_evaluate_revisions_cache.py` → `tests/unit/test_run_eval_cache.py`
- Rename: `docs/REVISION_EVALUATION.md` → `docs/REVISION_OPS.md`

**Step 1: git mv を順次実行**

```bash
cd /c/VSCode/rag/rag-local
git mv apps/revision-eval apps/revision-ops
git mv apps/revision-ops/evaluate_revisions.py apps/revision-ops/run_eval.py
git mv apps/revision-ops/ui/eval_ui.py apps/revision-ops/ui/ops_ui.py
git mv tests/unit/test_evaluate_revisions_cache.py tests/unit/test_run_eval_cache.py
git mv docs/REVISION_EVALUATION.md docs/REVISION_OPS.md
```

**Step 2: Commit**

```bash
git add -A
git commit -m "rename: revision-eval → revision-ops, source files renamed

- apps/revision-eval/ → apps/revision-ops/
- evaluate_revisions.py → run_eval.py
- eval_ui.py → ops_ui.py
- test file and doc file renamed accordingly"
```

---

## Task 2: `output_handler.py` — app_prefix + Metadata シート + 新ファイル名

**Files:**
- Modify: `src/handlers/output_handler.py`

**Step 1: `OutputHandler.__init__` に `app_prefix` 追加 (L10-14)**

変更前:
```python
class OutputHandler:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.output_dir = os.path.join(config.base_dir, "data", "output", "latest")
        os.makedirs(self.output_dir, exist_ok=True)
```

変更後:
```python
class OutputHandler:
    def __init__(self, config: SearchConfig, app_prefix: str = ""):
        self.config = config
        base_output_dir = os.path.join(config.base_dir, "data", "output", "latest")
        self.output_dir = os.path.join(base_output_dir, app_prefix) if app_prefix else base_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
```

**Step 2: `ExcelOutputHandler.save_data` のファイル名変更 (L21-51)**

変更前 (L29-34):
```python
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_summary = self.config.get_param_summary()
        # ファイル名にモードを追加
        output_file = os.path.join(
            self.output_dir,
            f"output_{mode}_{param_summary}_{timestamp}.xlsx"
        )
```

変更後:
```python
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"answer_{mode}_{timestamp}.xlsx"
        )
```

ExcelWriter ブロック内（L44-46）に Metadata シート書き出し追加:
```python
            with pd.ExcelWriter(output_file, **writer_options) as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                self._format_excel(writer, df)
                self._write_metadata_sheet(writer)  # 追加
```

**Step 3: `save_data_multi_stage` のファイル名変更 (L120-124)**

変更前:
```python
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"output_{mode}_{self.config.get_param_summary()}_{timestamp}.xlsx"
        )
```

変更後:
```python
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"answer_{mode}_{timestamp}.xlsx"
        )
```

ExcelWriter ブロック内（L127-140）に `_write_metadata_sheet` 追加（最終シート書き出し後）。

**Step 4: `save_data_dual_provider` のファイル名変更 (L246-250)**

変更前:
```python
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"output_{mode}_{self.config.get_param_summary()}_{timestamp}.xlsx"
        )
```

変更後:
```python
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"answer_{mode}_{timestamp}.xlsx"
        )
```

ExcelWriter ブロック内（L252-256）に `_write_metadata_sheet` 追加。

**Step 5: `_write_metadata_sheet` メソッド新設**

`_format_excel_dual_provider` の後（L421付近）に追加:

```python
    def _write_metadata_sheet(self, writer: pd.ExcelWriter):
        """検索パラメータを Metadata シートに記録"""
        metadata = {
            "Parameter": [
                "vector_weight", "keyword_weight", "search_mode",
                "search_type", "top_k", "embedding_provider",
                "embedding_model", "timestamp",
            ],
            "Value": [
                self.config.vector_weight,
                self.config.keyword_weight,
                self.config.search_mode,
                self.config.search_type,
                self.config.top_k,
                self.config.embedding_provider,
                self.config.embedding_model,
                datetime.now().isoformat(),
            ],
        }
        pd.DataFrame(metadata).to_excel(writer, index=False, sheet_name="Metadata")
```

**Step 6: `OutputHandlerFactory.create` に `app_prefix` 追加 (L425-432)**

変更前:
```python
class OutputHandlerFactory:
    @staticmethod
    def create(output_type: str, config: SearchConfig) -> OutputHandler:
        if output_type == "excel":
            return ExcelOutputHandler(config)
```

変更後:
```python
class OutputHandlerFactory:
    @staticmethod
    def create(output_type: str, config: SearchConfig, app_prefix: str = "") -> OutputHandler:
        if output_type == "excel":
            return ExcelOutputHandler(config, app_prefix=app_prefix)
```

**Step 7: Commit**

```bash
git add src/handlers/output_handler.py
git commit -m "feat: add app_prefix to OutputHandler, replace param_summary with Metadata sheet

- OutputHandler accepts app_prefix for subdirectory routing
- Filename: output_{mode}_{param}_{ts} → answer_{mode}_{ts}
- New _write_metadata_sheet() stores search params in Excel"
```

---

## Task 3: `processor.py` — `app_prefix="answer"` を渡す

**Files:**
- Modify: `src/core/processor.py:23`

**Step 1: OutputHandlerFactory.create に app_prefix 追加**

変更前 (L23):
```python
        self.output_handler = OutputHandlerFactory.create(config.output_type, config)
```

変更後:
```python
        self.output_handler = OutputHandlerFactory.create(config.output_type, config, app_prefix="answer")
```

**Step 2: Commit**

```bash
git add src/core/processor.py
git commit -m "feat: route answer-support output to answer/ subdirectory"
```

---

## Task 4: `ops_ui.py` — モード別ファイル名 + 出力先 + UIテキスト

**Files:**
- Modify: `apps/revision-ops/ui/ops_ui.py`

**Step 1: モジュール docstring 更新 (L1-4)**

変更前:
```python
"""運用保守効率化AI（改定影響調査）— 評価 Streamlit UI

改定番号を選択し、Azure OpenAI / VertexAI 両方で検索して正解IDとのマッチを評価する。
バッチ版は evaluate_revisions.py（Excel出力）。
"""
```

変更後:
```python
"""運用保守効率化AI（改定影響調査）— Streamlit UI

評価モード: 改定番号を選択し、Azure/VertexAI 両方で検索して正解IDとのマッチを評価。
影響調査モード: キーワード検索で改定の影響範囲を調査。
バッチ版は run_eval.py（Excel出力）。
"""
```

**Step 2: `save_chat_history()` をモード別ファイル名に変更 (L573-608)**

変更前 (L596-600):
```python
            output_dir = PROJECT_ROOT / "data" / "output" / "latest"
            output_dir.mkdir(parents=True, exist_ok=True)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"eval_chat_history_{timestamp}.xlsx"
```

変更後:
```python
            output_dir = PROJECT_ROOT / "data" / "output" / "latest" / "rev"
            output_dir.mkdir(parents=True, exist_ok=True)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            app_mode = st.session_state.get("app_mode", "evaluation")
            mode_prefix = "rev_impact" if app_mode == "impact_analysis" else "rev_eval"
            output_path = output_dir / f"{mode_prefix}_chat_{timestamp}.xlsx"
```

**Step 3: UIテキスト更新**

L708:
```python
# 変更前
st.set_page_config(page_title="改定影響調査", layout="wide", initial_sidebar_state="expanded")
# 変更後
st.set_page_config(page_title="運用保守効率化AI", layout="wide", initial_sidebar_state="expanded")
```

L713:
```python
# 変更前
st.title("事務改定 AI")
# 変更後
st.title("改定影響調査")
```

**Step 4: Commit**

```bash
git add apps/revision-ops/ui/ops_ui.py
git commit -m "feat: mode-aware output filenames in ops_ui, output to rev/ subdirectory"
```

---

## Task 5: `run_eval.py` — 出力先・ファイル名変更

**Files:**
- Modify: `apps/revision-ops/run_eval.py`

**Step 1: docstring 更新 (L1-8)**

変更前:
```python
"""
改定影響調査スクリプト（多段階検索・横並び比較版）

多段階ハイブリッド検索を使用して、Azure/VertexAI両方で検索を実行。
結果を横並びで比較できるExcelファイルに出力。

入力: input/multi_stage_input.xlsx
出力: output/revision_evaluation_{timestamp}.xlsx
```

変更後:
```python
"""
改定影響調査スクリプト（多段階検索・横並び比較版）

多段階ハイブリッド検索を使用して、Azure/VertexAI両方で検索を実行。
結果を横並びで比較できるExcelファイルに出力。

入力: data/input/multi_stage_input.xlsx
出力: data/output/latest/rev/rev_eval_batch_{timestamp}.xlsx
```

**Step 2: OUTPUT_DIR 変更 (L102)**

変更前:
```python
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "latest"
```

変更後:
```python
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "latest" / "rev"
```

**Step 3: save_results のファイル名変更 (L807)**

変更前:
```python
        output_file = OUTPUT_DIR / f"revision_evaluation_{timestamp}.xlsx"
```

変更後:
```python
        output_file = OUTPUT_DIR / f"rev_eval_batch_{timestamp}.xlsx"
```

**Step 4: Commit**

```bash
git add apps/revision-ops/run_eval.py
git commit -m "feat: rename batch output to rev_eval_batch, output to rev/ subdirectory"
```

---

## Task 6: テスト・共通モジュール・設定の修正

**Files:**
- Modify: `tests/unit/test_run_eval_cache.py`
- Modify: `ui/shared.py`
- Modify: `config.py`
- Modify: `config/settings.yaml`

**Step 1: テストファイルのパス修正 (`test_run_eval_cache.py`)**

L1:
```python
# 変更前
"""evaluate_revisions.py のキャッシュ機構テスト"""
# 変更後
"""run_eval.py のキャッシュ機構テスト"""
```

L9-12:
```python
# 変更前
# apps/revision-eval はハイフン付きディレクトリのため importlib で読み込む
PROJECT_ROOT = Path(__file__).parent.parent.parent
_mod_path = PROJECT_ROOT / "apps" / "revision-eval" / "evaluate_revisions.py"
_spec = importlib.util.spec_from_file_location("evaluate_revisions", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["evaluate_revisions"] = _mod

# 変更後
# apps/revision-ops はハイフン付きディレクトリのため importlib で読み込む
PROJECT_ROOT = Path(__file__).parent.parent.parent
_mod_path = PROJECT_ROOT / "apps" / "revision-ops" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = _mod
```

**Step 2: テスト実行**

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_cache.py -v
```

Expected: 全テスト PASS

**Step 3: `ui/shared.py` docstring 更新 (L1-5)**

変更前:
```python
"""共通UI部品 — 回答支援AI（類似回答検索）・運用保守効率化AI（改定影響調査）共用

このモジュールは以下から import される:
- apps/answer-support/ui/chat.py（回答支援UI）
- apps/revision-eval/ui/eval_ui.py（改定影響調査UI）
```

変更後:
```python
"""共通UI部品 — 回答支援AI（類似回答検索）・運用保守効率化AI（改定影響調査）共用

このモジュールは以下から import される:
- apps/answer-support/ui/chat.py（回答支援UI）
- apps/revision-ops/ui/ops_ui.py（改定影響調査UI）
```

**Step 4: `config.py` の `get_param_summary` に非推奨注記 (L284-288)**

変更前:
```python
    def get_param_summary(self) -> str:
        """パラメータのサマリー文字列を生成（LLM拡張検索対応）"""
```

変更後:
```python
    def get_param_summary(self) -> str:
        """パラメータのサマリー文字列を生成（LLM拡張検索対応）

        Note: ファイル名への使用は廃止。Excel Metadata シートに移行済み。
        ログ出力・デバッグ用に保持。
        """
```

**Step 5: `config/settings.yaml` コメント更新**

L11:
```yaml
# 変更前
#   evaluation: 改定影響調査専用設定（evaluate_revisions.py）
# 変更後
#   evaluation: 改定影響調査専用設定（run_eval.py）
```

L111:
```yaml
# 変更前
# 改定影響調査専用設定（evaluate_revisions.py）
# 変更後
# 改定影響調査専用設定（run_eval.py）
```

**Step 6: Commit**

```bash
git add tests/unit/test_run_eval_cache.py ui/shared.py config.py config/settings.yaml
git commit -m "fix: update path references after revision-ops rename"
```

---

## Task 7: ドキュメント更新

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/REVISION_OPS.md` (旧 REVISION_EVALUATION.md)
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `docs/CONFIGURATION.md`

以下の置換を全ドキュメントに適用:

| 検索 | 置換 |
|------|------|
| `apps/revision-eval/evaluate_revisions.py` | `apps/revision-ops/run_eval.py` |
| `apps/revision-eval/ui/eval_ui.py` | `apps/revision-ops/ui/ops_ui.py` |
| `apps/revision-eval/` | `apps/revision-ops/` |
| `revision-eval/` (ディレクトリツリー内) | `revision-ops/` |
| `evaluate_revisions.py` (単独参照) | `run_eval.py` |
| `eval_ui.py` (単独参照) | `ops_ui.py` |
| `revision_evaluation_YYYYMMDD` | `rev_eval_batch_YYYYMMDD` |
| `scripts/evaluate_revisions.py` (docs内の古いパス) | `apps/revision-ops/run_eval.py` |
| `REVISION_EVALUATION.md` | `REVISION_OPS.md` |
| `事務改定評価` (ドキュメントタイトル等) | `改定影響調査` |

### README.md 変更箇所

- L10: テーブル内のパス → `apps/revision-ops/run_eval.py`, `apps/revision-ops/ui/ops_ui.py`
- L63: `python apps/revision-eval/evaluate_revisions.py` → `python apps/revision-ops/run_eval.py`
- L66: `streamlit run apps/revision-eval/ui/eval_ui.py` → `streamlit run apps/revision-ops/ui/ops_ui.py`
- L80-83: ディレクトリツリー `revision-eval/` → `revision-ops/`、ファイル名更新
- L121: 出力パス `revision_evaluation_YYYYMMDD.xlsx` → `rev_eval_batch_YYYYMMDD.xlsx`、`data/output/latest/rev/` パスに

### CLAUDE.md 変更箇所

- L187: ディレクトリツリー内 `revision-eval/` → `revision-ops/`
- L188: `evaluate_revisions.py` → `run_eval.py`
- L201: `REVISION_EVALUATION.md` → `REVISION_OPS.md`
- 出力ディレクトリ構造セクションがあれば更新

### docs/REVISION_OPS.md 変更箇所

- L1: タイトル `事務改定評価システム` → `改定影響調査システム`
- L18: `evaluate_revisions.py（多段階検索版）` → `run_eval.py（多段階検索版）`
- L172: `python scripts/evaluate_revisions.py` → `python apps/revision-ops/run_eval.py`
- L185: 同上パターン
- L194: `data/output/revision_evaluation_*.xlsx` → `data/output/latest/rev/rev_eval_batch_*.xlsx`
- L304: テーブル内 `scripts/evaluate_revisions.py` → `apps/revision-ops/run_eval.py`

### docs/ARCHITECTURE.md 変更箇所

- L303: `scripts/evaluate_revisions.py` → `apps/revision-ops/run_eval.py`
- L308: `data/output/revision_evaluation_*.xlsx` → `data/output/latest/rev/rev_eval_batch_*.xlsx`

### docs/TROUBLESHOOTING.md 変更箇所

- L291: `python apps/revision-eval/evaluate_revisions.py` → `python apps/revision-ops/run_eval.py`

### docs/CONFIGURATION.md 変更箇所

- streamlit コマンドパス更新（3箇所程度）

### docs/plans/2026-03-02-terminal-log-redesign.md

- **変更しない**: 過去の計画書は実行済みのため、そのまま保持

**Commit:**

```bash
git add README.md CLAUDE.md docs/REVISION_OPS.md docs/ARCHITECTURE.md docs/TROUBLESHOOTING.md docs/CONFIGURATION.md
git commit -m "docs: update all references after revision-ops rename and output naming unification"
```

---

## Task 8: 最終検証

**Step 1: 全テスト実行**

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v
```

Expected: 全テスト PASS

**Step 2: 旧パス参照の残存チェック**

```bash
cd /c/VSCode/rag/rag-local
grep -r "revision-eval" --include="*.py" --include="*.md" --include="*.yaml" . | grep -v ".plans/" | grep -v ".git/"
grep -r "evaluate_revisions" --include="*.py" --include="*.md" --include="*.yaml" . | grep -v ".plans/" | grep -v ".git/"
grep -r "eval_ui" --include="*.py" --include="*.md" --include="*.yaml" . | grep -v ".plans/" | grep -v ".git/"
grep -r "eval_chat_history" --include="*.py" . | grep -v ".git/"
grep -r "revision_evaluation" --include="*.py" . | grep -v ".git/"
grep -r "output_chat\|output_batch\|output_multi_stage\|output_dual_provider" --include="*.py" . | grep -v ".git/"
```

Expected: 各コマンドの出力が空（残存参照なし）。`.plans/` 内の過去計画書は無視。

**Step 3: import 動作確認**

```bash
cd /c/VSCode/rag/rag-local
python -c "from src.handlers.output_handler import OutputHandlerFactory; print('OK')"
python -c "from src.core.processor import Processor; print('OK')"
```

Expected: 両方 `OK`

**Step 4: UI 起動テスト（手動）**

```bash
cd /c/VSCode/rag/rag-local
streamlit run apps/answer-support/ui/chat.py
# → チャット → 保存 → data/output/latest/answer/answer_chat_*.xlsx 確認 + Metadata シート確認

streamlit run apps/revision-ops/ui/ops_ui.py
# → 評価モード → チャット → 保存 → data/output/latest/rev/rev_eval_chat_*.xlsx 確認
# → 影響調査モード → チャット → 保存 → data/output/latest/rev/rev_impact_chat_*.xlsx 確認
```

**Step 5: 残存チェックで見つかった漏れがあれば修正 → 追加コミット**
