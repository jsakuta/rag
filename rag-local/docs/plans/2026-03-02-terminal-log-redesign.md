# ターミナルログ出力リデザイン 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 両アプリ（回答支援AI（類似回答検索）/ 運用保守効率化AI（改定影響調査））のターミナル出力を、引き継ぎ先の顧客が検索フローを追えるダッシュボード型に刷新する

**Architecture:** ログを3レイヤー(App/Infra/Noise)に分離。App層のみデフォルト表示、Infra層はDEBUGレベル、Noise層は常時抑制。新規ヘルパー関数(suppress_noise, print_startup_summary, print_query_panel)をlogger.pyに追加し、各アプリから呼び出す。

**Tech Stack:** Python logging, Rich library (既存依存), Streamlit

---

## Task 1: logger.py にノイズ抑制関数を追加

**Files:**
- Modify: `src/utils/logger.py`
- Test: `tests/unit/test_logger_noise.py`

**Step 1: テストを書く**

```python
# tests/unit/test_logger_noise.py
import logging
from src.utils.logger import suppress_noise

def test_suppress_noise_sets_third_party_to_warning():
    suppress_noise()
    assert logging.getLogger("chromadb").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("streamlit").level == logging.WARNING

def test_suppress_noise_sets_google_azure_to_warning():
    suppress_noise()
    assert logging.getLogger("google.auth").level == logging.WARNING
    assert logging.getLogger("google.api_core").level == logging.WARNING
    assert logging.getLogger("azure.core").level == logging.WARNING
    assert logging.getLogger("azure.identity").level == logging.WARNING

def test_suppress_noise_filters_scriptruncontext():
    """ScriptRunContext 警告がフィルタされること"""
    suppress_noise()
    streamlit_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils")
    assert streamlit_logger.level >= logging.ERROR
```

**Step 2: テスト実行（失敗確認）**

Run: `cd rag-local && python -m pytest tests/unit/test_logger_noise.py -v`
Expected: FAIL（suppress_noise が存在しない）

**Step 3: suppress_noise() を実装**

`src/utils/logger.py` の末尾に追加:

```python
def suppress_noise():
    """サードパーティライブラリ + Streamlit のノイズログを抑制"""
    noisy_loggers = [
        "chromadb", "chromadb.config", "chromadb.telemetry",
        "httpx", "httpcore", "urllib3",
        "google.auth", "google.api_core", "google.cloud",
        "azure.core", "azure.identity",
        "streamlit", "streamlit.logger",
        "streamlit.runtime.scriptrunner_utils",
        "altair",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ScriptRunContext 警告を完全に抑制
    logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)
```

**Step 4: テスト実行（成功確認）**

Run: `cd rag-local && python -m pytest tests/unit/test_logger_noise.py -v`
Expected: PASS

**Step 5: コミット**

```bash
git add tests/unit/test_logger_noise.py src/utils/logger.py
git commit -m "feat: add suppress_noise() to logger.py"
```

---

## Task 2: logger.py に起動サマリ表示関数を追加

**Files:**
- Modify: `src/utils/logger.py`
- Test: `tests/unit/test_logger_dashboard.py`

**Step 1: テストを書く**

```python
# tests/unit/test_logger_dashboard.py
from io import StringIO
from unittest.mock import patch
from src.utils.logger import print_startup_summary

def test_print_startup_summary_all_ok(capsys):
    """全チェックOKの場合、アプリ名とReady表示が出ること"""
    checks = [
        ("DB接続", True, "naibujimu: 11,439件"),
        ("LLM接続", True, "Gemini 2.5 Flash Lite"),
    ]
    print_startup_summary("テストAI v1.0", checks)
    captured = capsys.readouterr()
    assert "テストAI" in captured.out
    assert "DB接続" in captured.out
    assert "Ready" in captured.out

def test_print_startup_summary_with_failure(capsys):
    """チェック失敗がある場合、✗が表示されること"""
    checks = [
        ("DB接続", True, "OK"),
        ("LLM接続", False, "接続失敗"),
    ]
    print_startup_summary("テストAI v1.0", checks)
    captured = capsys.readouterr()
    assert "接続失敗" in captured.out
```

**Step 2: テスト実行（失敗確認）**

Run: `cd rag-local && python -m pytest tests/unit/test_logger_dashboard.py -v`
Expected: FAIL

**Step 3: print_startup_summary() を実装**

`src/utils/logger.py` に追加:

```python
def print_startup_summary(app_name: str, checks: list):
    """起動サマリをダッシュボード形式で表示

    Args:
        app_name: アプリ名（例: "回答支援AI（類似回答検索）v1.0"）
        checks: [(label, ok, detail), ...] のリスト
    """
    console = get_console()
    if console:
        console.print()
        console.print(Rule(f" {app_name} ", style="bold cyan"))
        console.print()
        for label, ok, detail in checks:
            icon = "[green]✔[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {icon} {label} {detail}")
        console.print()
        console.print(Rule(style="dim"))
    else:
        print(f"\n{'=' * 40}")
        print(f"  {app_name}")
        print(f"{'=' * 40}")
        for label, ok, detail in checks:
            icon = "OK" if ok else "NG"
            print(f"  [{icon}] {label} {detail}")
        print(f"{'─' * 40}")
```

**Step 4: テスト実行（成功確認）**

Run: `cd rag-local && python -m pytest tests/unit/test_logger_dashboard.py -v`
Expected: PASS

**Step 5: コミット**

```bash
git add tests/unit/test_logger_dashboard.py src/utils/logger.py
git commit -m "feat: add print_startup_summary() for dashboard display"
```

---

## Task 3: logger.py にクエリパネル表示関数を追加

**Files:**
- Modify: `src/utils/logger.py`
- Modify: `tests/unit/test_logger_dashboard.py`

**Step 1: テストを追加**

```python
# tests/unit/test_logger_dashboard.py に追加
from src.utils.logger import print_query_panel

def test_print_query_panel_basic(capsys):
    """検索パネルにクエリとメタデータが表示されること"""
    print_query_panel(
        query_number=1,
        query_text="保険証が廃止された場合の手続き",
        metadata={"改定": "③", "タイプ": "hybrid"},
        results={"Azure": 45, "VertexAI": 38},
        elapsed=1.2,
    )
    captured = capsys.readouterr()
    assert "保険証" in captured.out
    assert "45" in captured.out
    assert "1.2" in captured.out

def test_print_query_panel_single_provider(capsys):
    """プロバイダーが1つの場合でも表示されること"""
    print_query_panel(
        query_number=1,
        query_text="テストクエリ",
        metadata={"業務": "内部事務"},
        results={"結果": 12},
        elapsed=0.8,
    )
    captured = capsys.readouterr()
    assert "テストクエリ" in captured.out
    assert "12" in captured.out
```

**Step 2: テスト実行（失敗確認）**

Run: `cd rag-local && python -m pytest tests/unit/test_logger_dashboard.py::test_print_query_panel_basic -v`
Expected: FAIL

**Step 3: print_query_panel() を実装**

`src/utils/logger.py` に追加:

```python
def print_query_panel(
    query_number: int,
    query_text: str,
    metadata: dict,
    results: dict,
    elapsed: float,
):
    """検索リクエストを構造化パネルで表示

    Args:
        query_number: クエリ番号
        query_text: 検索テキスト（80文字で切り詰め）
        metadata: キー=値ペアの辞書（改定番号、検索タイプ等）
        results: プロバイダー名=件数の辞書
        elapsed: 所要時間（秒）
    """
    console = get_console()
    truncated = query_text[:80] + "..." if len(query_text) > 80 else query_text
    meta_str = "  ".join(f"{k}: {v}" for k, v in metadata.items())
    result_str = "  ".join(f"{k}: {v}件" for k, v in results.items())

    if console:
        title = f"検索 #{query_number}"
        if metadata:
            title += f"  {meta_str}"
        lines = [f"  Q: {truncated}"]
        if results:
            lines.append(f"  {result_str}")
        lines.append(f"  所要: {elapsed:.1f}s")
        content = "\n".join(lines)
        console.print(Panel(content, title=title, title_align="left", border_style="cyan", padding=(0, 1)))
    else:
        print(f"\n--- 検索 #{query_number} {meta_str} ---")
        print(f"  Q: {truncated}")
        if results:
            print(f"  {result_str}")
        print(f"  所要: {elapsed:.1f}s")
        print(f"{'─' * 35}")
```

**Step 4: テスト実行（成功確認）**

Run: `cd rag-local && python -m pytest tests/unit/test_logger_dashboard.py -v`
Expected: ALL PASS

**Step 5: コミット**

```bash
git add tests/unit/test_logger_dashboard.py src/utils/logger.py
git commit -m "feat: add print_query_panel() for structured search display"
```

---

## Task 4: コアモジュールの INFO → DEBUG 降格

**Files:**
- Modify: `src/utils/vector_db.py` (6箇所)
- Modify: `src/core/search/vector_search_engine.py` (4箇所)
- Modify: `src/core/search/keyword_search_engine.py` (8箇所)
- Modify: `src/core/search/multi_stage_orchestrator.py` (4箇所)
- Modify: `src/core/search/chromadb_keyword_search.py` (logger修正 + 4箇所)
- Modify: `src/utils/gemini_embedding.py` (3箇所)

**Step 1: 各ファイルで `logger.info` → `logger.debug` に置換**

`src/utils/vector_db.py`:
- Line 40: `logger.info("LRUCache: ...")` → `logger.debug`
- Line 87: `logger.info("New ChromaDB client ...")` → `logger.debug`
- Line 96: `logger.info("Existing collection ...")` → `logger.debug`
- Line 107: `logger.info("New collection ...")` → `logger.debug`
- Line 185: `logger.info("Added {len(texts)} ...")` → `logger.debug`
- Line 276: `logger.info("Collection '...' deleted")` → `logger.debug`

`src/core/search/vector_search_engine.py`:
- Line 41: `logger.info("VectorSearchEngineを初期化")` → `logger.debug`
- Line 80: `logger.info("Search source filter")` → `logger.debug`
- Line 88: `logger.info("Vector search returned")` → `logger.debug`
- Line 95: `logger.info("Search results by source")` → `logger.debug`

`src/core/search/keyword_search_engine.py`:
- Line 43: `logger.info("Sudachi辞書")` → `logger.debug`
- Line 65: `logger.info("KeywordSearchEngine初期化")` → `logger.debug`
- Line 173: `logger.info("キャッシュ構築中")` → `logger.debug`
- Line 184: `logger.info("キャッシュ構築完了")` → `logger.debug`
- Line 205: `logger.info("キャッシュ件数不一致")` → `logger.debug`
- Line 208: `logger.info("キャッシュハッシュ不一致")` → `logger.debug`
- Line 211: `logger.info("キャッシュ読み込み")` → `logger.debug`
- Line 228: `logger.info("キャッシュ保存")` → `logger.debug`

`src/core/search/multi_stage_orchestrator.py`:
- Line 80: `logger.info("MultiStageOrchestrator初期化")` → `logger.debug`
- Line 104: `logger.info("FilterMode: ...")` → `logger.debug`
- Line 108: `logger.info("Keywords: ...")` → `logger.debug`
- Line 286: `logger.info("Both: ... Original_Only: ...")` → `logger.debug`
（Line 102, 114, 129, 142 は App 層なのでINFO維持）

`src/core/search/chromadb_keyword_search.py`:
- Line 20: `logging.getLogger(__name__)` → `setup_logger(__name__)` に変更 + import追加
- Line 91, 94, 148, 183: `logger.info` → `logger.debug`

`src/utils/gemini_embedding.py`:
- Line 77: `logger.info("singleton instance created")` → `logger.debug`
- Line 104: `logger.info("Embedding API initialized")` → `logger.debug`
- Line 177: `logger.info("Generated embeddings")` → `logger.debug`

**Step 2: 既存テスト実行**

Run: `cd rag-local && python -m pytest tests/ -v --tb=short 2>&1 | head -50`
Expected: ALL PASS（ログレベル変更は動作に影響しない）

**Step 3: コミット**

```bash
git add src/utils/vector_db.py src/core/search/vector_search_engine.py \
  src/core/search/keyword_search_engine.py src/core/search/multi_stage_orchestrator.py \
  src/core/search/chromadb_keyword_search.py src/utils/gemini_embedding.py
git commit -m "refactor: downgrade infra-layer logs from INFO to DEBUG"
```

---

## Task 5: ユーティリティ・アプリモジュールの INFO → DEBUG 降格

**Files:**
- Modify: `src/utils/dynamic_db_manager.py` (~30箇所)
- Modify: `src/core/searcher.py` (~20箇所)
- Modify: `apps/revision-eval/ui/eval_ui.py` (3箇所)

**Step 1: dynamic_db_manager.py の降格**

以下を全て `logger.info` → `logger.debug` に変更:
- Line 92, 96: クリーンアップログ
- Line 147, 149, 156: タイムスタンプ読み込み
- Line 182, 194: キー移行
- Line 258, 276, 293: クリーンアップ・移行・保存
- Line 322: タイムスタンプ更新
- Line 383, 387: バックアップ・初期化
- Line 433, 449, 464: データ検出（履歴・シナリオ）
- Line 491, 499, 504, 515, 534, 538, 550, 552, 571, 574, 578: DB更新チェック詳細
- Line 753, 771, 774: DB操作詳細
- Line 800, 814: バッチ処理中ログ
- Line 877: ファイル処理
- Line 963, 970, 988, 1000, 1018, 1036, 1051, 1086: データ準備・読み込み

以下はINFO維持（App層）:
- Line 417: `参照ファイルの分析を開始...`
- Line 468: `業務分野検出: ...`
- Line 585, 603, 608: DB更新開始/完了/最新
- Line 743, 784, 834: リセット・ベクトル化

**Step 2: searcher.py の降格**

以下を `logger.info` → `logger.debug` に変更:
- Line 35, 69, 75, 78: 初期化
- Line 134: プロンプトファイル
- Line 212, 243, 249, 255, 259: テキスト準備・キャッシュ
- Line 324, 332, 337: DB選択
- Line 368, 376, 383: 検索フィルタ・結果
- Line 513, 514, 535, 536, 537: 検索結果処理
- Line 565, 567, 575, 578: 制限・最終結果
- Line 606: DB更新チェック

以下はINFO維持:
- Line 192: `LLM API error - stopping processing`
- Line 643: DB切り替え完了

**Step 3: eval_ui.py の降格**

- Line 75: `正解IDデータを読み込み` → `logger.debug`
- Line 304: `プリウォーム完了: LLM` → `logger.debug`
- Line 312: `プリウォーム完了: キーワードキャッシュ` → `logger.debug`
（Line 590, 593, 600 は App 層なのでINFO維持）

**Step 4: 既存テスト実行**

Run: `cd rag-local && python -m pytest tests/ -v --tb=short 2>&1 | head -50`
Expected: ALL PASS

**Step 5: コミット**

```bash
git add src/utils/dynamic_db_manager.py src/core/searcher.py \
  apps/revision-eval/ui/eval_ui.py
git commit -m "refactor: downgrade infra-layer logs in utils/apps from INFO to DEBUG"
```

---

## Task 6: 回答支援AI（類似回答検索）にダッシュボード表示を統合

**Files:**
- Modify: `apps/answer-support/ui/chat.py`

**Step 1: suppress_noise + 起動サマリを統合**

chat.py の初期化部分（`initialize_session_state` 付近）に以下を追加:
1. `suppress_noise()` をインポートして最初に呼び出す
2. プリウォーム完了後に `print_startup_summary()` を呼び出す
3. 検索実行部分で `print_query_panel()` を呼び出す（既存のログ行を置換）

具体的には:
- 起動時: `suppress_noise()` 呼び出し
- 初期化完了時: DB件数・キャッシュ状態を集約して `print_startup_summary("回答支援AI（類似回答検索）v1.0", checks)` を呼び出し
- 検索時: 既存の `logger.info(f"=== 質問 {query_number} の処理開始 ===")` 〜 `=== 完了 ===` のブロックを `print_query_panel()` に置換

**Step 2: chat.py の検索ログをパネル表示に変更**

`process_query()` 内:
- 既存の `logger.info(f"=== 質問 {query_number} の処理開始 ===")` を削除
- 既存の個別ログ行（検索モード、検索対象、バランス、結果数、個別結果）を削除
- 検索完了後に `print_query_panel()` 1回で集約表示
- `logger.info(f"=== 質問 {query_number} の検索完了 ===")` を削除

**Step 3: 手動テスト**

Run: `cd rag-local && streamlit run apps/answer-support/ui/chat.py`
Expected:
- ScriptRunContext 警告が出ない
- 起動サマリがダッシュボード形式で表示される
- 検索実行時にパネル表示される

**Step 4: コミット**

```bash
git add apps/answer-support/ui/chat.py
git commit -m "feat: integrate dashboard display into answer-support UI"
```

---

## Task 7: 運用保守効率化AI（改定影響調査）（UI版）にダッシュボード表示を統合

**Files:**
- Modify: `apps/revision-eval/ui/eval_ui.py`

**Step 1: suppress_noise + 起動サマリを統合**

eval_ui.py に以下を追加:
1. `suppress_noise()` を `run_streamlit_ui()` の最初で呼び出す
2. `_prewarm()` 完了後に `print_startup_summary("運用保守効率化AI（改定影響調査）v1.0", checks)` を呼び出す
   - チェック項目: DB接続、LLM接続、キーワードキャッシュ、正解ID読み込み
3. 検索実行部分で `print_query_panel()` を呼び出す

**Step 2: 検索ログをパネル表示に変更**

`_log_search_results()` 付近（Line 588-601）:
- 既存の `logger.info(f"=== 評価クエリ ...")` と `logger.info(f"Azure検索結果数: ...")` を `print_query_panel()` に統合

**Step 3: 手動テスト**

Run: `cd rag-local && streamlit run apps/revision-eval/ui/eval_ui.py`
Expected:
- ScriptRunContext 警告が出ない
- 起動サマリがダッシュボード形式で表示
- 検索時にパネル表示

**Step 4: コミット**

```bash
git add apps/revision-eval/ui/eval_ui.py
git commit -m "feat: integrate dashboard display into revision-eval UI"
```

---

## Task 8: 運用保守効率化AI（改定影響調査）（バッチ版）の出力整理

**Files:**
- Modify: `apps/revision-eval/evaluate_revisions.py`

**Step 1: suppress_noise + 起動サマリ統合**

`main()` の先頭で `suppress_noise()` 呼び出し。

既存の `print_section("DB存在確認")` + `print_table()` + `print_section("実行設定")` + `print_status() × 12行` を `print_startup_summary()` + DB テーブルに統合:
- 起動パネル: アプリ名 + DB/LLM/入力データの OK/NG
- DBテーブル: 既存の `print_table()` をそのまま活用（エリア × プロバイダーの表は有用）
- 実行設定の12行は `--verbose` 時のみ表示（既存の verbose フラグを活用）

**Step 2: 改定ごとの結果表示をコンパクト化**

既存の `print_revision_header()` + `print_search_result()` の呼び出しはそのまま維持。
これらは既に Rich で構造化されており、設計イメージに近い。

**Step 3: 手動テスト**

Run: `cd rag-local && python -m apps.revision_eval.evaluate_revisions --help`
（実行にはデータが必要なため、起動確認のみ）

**Step 4: コミット**

```bash
git add apps/revision-eval/evaluate_revisions.py
git commit -m "feat: integrate dashboard display into revision-eval batch"
```

---

## Task 9: 最終検証

**Step 1: 全テスト実行**

Run: `cd rag-local && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: 回答支援AI（類似回答検索）の手動起動テスト**

Run: `cd rag-local && streamlit run apps/answer-support/ui/chat.py`

確認項目:
- [ ] ScriptRunContext 警告が出ない
- [ ] 起動サマリがダッシュボード形式
- [ ] 検索実行でパネル表示
- [ ] `LOG_LEVEL=DEBUG` で Infra ログが復活する

**Step 3: 運用保守効率化AI（改定影響調査）（UI版）の手動起動テスト**

Run: `cd rag-local && streamlit run apps/revision-eval/ui/eval_ui.py`

確認項目:
- [ ] ScriptRunContext 警告が出ない
- [ ] 起動サマリ（DB/LLM/キャッシュ/正解ID）
- [ ] 検索パネル表示

**Step 4: 最終コミット（もしあれば）**

```bash
git add -A
git commit -m "fix: final adjustments for terminal log redesign"
```
