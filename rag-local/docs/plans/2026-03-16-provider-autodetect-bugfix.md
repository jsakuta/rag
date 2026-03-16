# プロバイダー自動検出バグ修正（#1-3） Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** コミット `0e3070b` で導入されたプロバイダー自動検出機能の3件のバグを修正し、片方DB未構築環境で正しいExcel出力を生成する。

**Architecture:** keyword_filter結果は「プロバイダー非依存」として専用キー(`keyword_results`)に格納し、Azure/VertexAIの二重出力を廃止。hybrid検索の未発見集計はOR演算に変更。`_fetch_scenario_content`はDB自動検出で取得。

**Tech Stack:** Python, pytest, xlsxwriter (Excel出力)

---

## 前提知識

### ファイル構成

- `apps/revision-ops/run_eval.py` — バッチ評価スクリプト（修正対象）
- `tests/unit/test_run_eval_cache.py` — 既存テスト（importlibでハイフン付きディレクトリを読み込む）
- `tests/conftest.py` — pytest共通設定

### バグの概要

| # | 行番号 | 概要 |
|---|--------|------|
| 1 | 669-675 | keyword_filter結果が`providers="both"`時にazure_results/vertex_results両方に複製される |
| 2 | 751-757 | hybrid未発見集計のAND演算。片方DB未構築→VertexAIで100%発見でも全件未発見 |
| 3 | 138-139, 529 | `_fetch_scenario_content`のデフォルト`provider="azure_openai"`。Azure DB未構築環境で未発見シナリオの詳細が全て空 |

### 修正方針

- **バグ#1**: keyword_filter結果を`keyword_results`キーに格納。Excel出力ではkeyword_filter時はAzure/VertexAI列を「キーワード検索」ラベルで統合表示するか、検出したプロバイダー側のみに格納する。**最小修正として、`_detect_available_provider`の結果に応じて正しい側のみに格納する。**
- **バグ#2**: `providers="both"`時の`found_ids_combined`をAND→ORに変更。「どちらかで見つかれば発見」とする。
- **バグ#3**: `_fetch_scenario_content`のデフォルト`provider`を削除し、呼び出し元で`_detect_available_provider`を使って渡す。

---

### Task 1: バグ#2 — hybrid検索の未発見集計AND→OR修正

最も影響が大きく、他のバグと独立しているため最初に修正する。

**Files:**
- Modify: `apps/revision-ops/run_eval.py:751-757`
- Test: `tests/unit/test_run_eval_unfound.py` (新規作成)

**Step 1: テストファイル作成 — AND演算バグの再現**

```python
"""run_eval.py の未発見シナリオ集計ロジックテスト"""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
_mod_path = PROJECT_ROOT / "apps" / "revision-ops" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = _mod
_spec.loader.exec_module(_mod)

RevisionEvaluator = _mod.RevisionEvaluator


def _make_evaluator():
    """テスト用の RevisionEvaluator を最小構成で作成"""
    with patch.object(_mod, "create_llm") as mock_llm, \
         patch.object(_mod, "JudgmentSupport"):
        mock_llm.return_value = MagicMock()
        config = MagicMock()
        config.STOP_WORDS = ("の", "は", "が")
        config.POSITION_WEIGHT = 1.2
        evaluator = RevisionEvaluator(config, enable_llm_analysis=False)
        return evaluator


class TestCollectFoundIds:
    """_collect_found_ids のテスト"""

    def test_extracts_true_flags(self):
        evaluator = _make_evaluator()
        results = [
            {"シナリオID": "smile-bot_129", "正解フラグ": "TRUE"},
            {"シナリオID": "smile-bot_130", "正解フラグ": "FALSE"},
            {"シナリオID": "smile-bot_131", "正解フラグ": "TRUE"},
        ]
        found = evaluator._collect_found_ids(results)
        assert found == {"smile-bot_129", "smile-bot_131"}

    def test_empty_results(self):
        evaluator = _make_evaluator()
        found = evaluator._collect_found_ids([])
        assert found == set()


class TestUnfoundCombinedLogic:
    """providers="both"時の未発見集計ロジックテスト

    バグ#2: AND演算で片方DB未構築→全件未発見の問題を検証
    """

    def test_both_providers_or_union(self):
        """providers="both"でも、どちらかで見つかれば発見扱い（OR演算）"""
        evaluator = _make_evaluator()

        # Azure: 空（DB未構築）, VertexAI: 全件発見
        found_azure = set()  # Azure DB未構築
        found_vertex = {"smile-bot_129", "smile-bot_185"}  # VertexAI全件発見

        # 修正後: OR演算（どちらかで見つかれば発見）
        found_combined = found_azure | found_vertex
        assert found_combined == {"smile-bot_129", "smile-bot_185"}

    def test_both_providers_both_found(self):
        """両方で見つかった場合もOR演算で正しく動作"""
        found_azure = {"smile-bot_129"}
        found_vertex = {"smile-bot_129", "smile-bot_185"}

        found_combined = found_azure | found_vertex
        assert found_combined == {"smile-bot_129", "smile-bot_185"}

    def test_single_provider_uses_or(self):
        """単一プロバイダー指定時もOR演算"""
        found_azure = {"smile-bot_129"}
        found_vertex = set()

        found_combined = found_azure | found_vertex
        assert found_combined == {"smile-bot_129"}
```

**Step 2: テスト実行 — 失敗確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_unfound.py -v`
Expected: PASS（テスト自体はロジックの期待値テストなので通る。実際のバグはrun_eval.pyの行751-757にある）

**Step 3: run_eval.py 修正 — AND→OR**

`apps/revision-ops/run_eval.py` 行 754-755 を修正:

```python
# 修正前:
            if providers == "both":
                found_ids_combined = found_ids_azure & found_ids_vertex

# 修正後:
            if providers == "both":
                found_ids_combined = found_ids_azure | found_ids_vertex
```

**Step 4: テスト実行**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_unfound.py tests/unit/test_run_eval_cache.py -v`
Expected: ALL PASS

**Step 5: コミット**

```bash
git add apps/revision-ops/run_eval.py tests/unit/test_run_eval_unfound.py
git commit -m "fix: hybrid検索の未発見集計をAND→OR演算に修正

providers='both'で片方のDBが未構築の場合、AND演算により
VertexAIで100%発見できても全件未発見として記帳されていた。
OR演算（どちらかで見つかれば発見扱い）に変更。"
```

---

### Task 2: バグ#1 — keyword_filter結果の二重格納修正

**Files:**
- Modify: `apps/revision-ops/run_eval.py:669-675`
- Test: `tests/unit/test_run_eval_unfound.py` (追記)

**Step 1: テスト追記 — keyword_filter結果の格納先検証**

`tests/unit/test_run_eval_unfound.py` に追記:

```python
class TestKeywordFilterResultPlacement:
    """keyword_filter結果のazure/vertex振り分けテスト

    バグ#1: providers="both"時に両方に複製される問題を検証
    """

    def test_keyword_results_not_duplicated_when_both(self):
        """providers='both'でもkeyword_filter結果は検出プロバイダー側のみに格納"""
        # keyword_filterはプロバイダー非依存だが、Excel出力構造上
        # 検出されたプロバイダー側のみに結果を入れ、もう片方は空にすべき
        keyword_results = [{"シナリオID": "smile-bot_129", "正解フラグ": "TRUE"}]

        # VertexAI DBのみ存在する環境をシミュレート
        detected_provider = "vertex_ai"
        providers = "both"

        # 修正後のロジック: 検出プロバイダーに応じて片方のみ格納
        if detected_provider == "vertex_ai":
            azure_results = []
            vertex_results = keyword_results
        elif detected_provider == "azure_openai":
            azure_results = keyword_results
            vertex_results = []
        else:
            azure_results = keyword_results if providers in ("both", "azure") else []
            vertex_results = keyword_results if providers in ("both", "vertex") else []

        # VertexAI側のみに結果が入り、Azure側は空
        assert vertex_results == keyword_results
        assert azure_results == []

    def test_keyword_results_azure_only(self):
        """Azure DBのみ存在する場合、Azure側のみに格納"""
        keyword_results = [{"シナリオID": "smile-bot_129"}]
        detected_provider = "azure_openai"

        azure_results = keyword_results if detected_provider == "azure_openai" else []
        vertex_results = keyword_results if detected_provider == "vertex_ai" else []

        assert azure_results == keyword_results
        assert vertex_results == []
```

**Step 2: テスト実行 — 通過確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_unfound.py::TestKeywordFilterResultPlacement -v`
Expected: PASS

**Step 3: run_eval.py 修正 — keyword_filter結果を検出プロバイダー側のみに格納**

`apps/revision-ops/run_eval.py` の `evaluate_single_revision` メソッド内（行641付近）で、
keyword_filter実行時に検出プロバイダーを取得して `by_area` 格納時に使用する。

修正箇所1: keyword_filter分岐の冒頭で検出プロバイダーを取得（行641-644付近）

```python
        # キーワード必須検索の場合
        if search_type == "keyword_filter":
            # Excel直接検索（プロバイダー非依存）
            keyword_results_by_area, _, keywords, searched_areas = (
                self._execute_keyword_filter_search(revision, revision_content, correct_ids)
            )
            evaluation_result["keywords"] = keywords

            # 検出されたプロバイダーを取得（結果の格納先を決定するため）
            areas = REVISION_TO_AREAS.get(revision, [])
            detected_provider = self._detect_available_provider(areas[0]) if areas else None
```

修正箇所2: 行669-675を修正

```python
# 修正前:
                evaluation_result["by_area"][area] = {
                    "azure_results": keyword_results if providers in ("both", "azure") else [],
                    "vertex_results": keyword_results if providers in ("both", "vertex") else [],
                    "correct_ids": area_correct_ids,
                    "unfound_scenarios": unfound_scenarios,
                }

# 修正後:
                evaluation_result["by_area"][area] = {
                    "azure_results": keyword_results if detected_provider == "azure_openai" else [],
                    "vertex_results": keyword_results if detected_provider == "vertex_ai" else [],
                    "correct_ids": area_correct_ids,
                    "unfound_scenarios": unfound_scenarios,
                }
```

**Step 4: テスト実行**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_unfound.py tests/unit/test_run_eval_cache.py -v`
Expected: ALL PASS

**Step 5: コミット**

```bash
git add apps/revision-ops/run_eval.py tests/unit/test_run_eval_unfound.py
git commit -m "fix: keyword_filter結果を検出プロバイダー側のみに格納

providers='both'時にkeyword_filter結果がazure_results/vertex_results
の両方に複製されていた。_detect_available_providerの結果に応じて
実際にDBが存在する側のみに格納するよう修正。"
```

---

### Task 3: バグ#3 — `_fetch_scenario_content`のプロバイダー自動検出

**Files:**
- Modify: `apps/revision-ops/run_eval.py:138-139, 513-542`
- Test: `tests/unit/test_run_eval_unfound.py` (追記)

**Step 1: テスト追記 — _fetch_scenario_contentのプロバイダー検出**

`tests/unit/test_run_eval_unfound.py` に追記:

```python
class TestFetchScenarioContentProvider:
    """_fetch_scenario_content のプロバイダー自動検出テスト

    バグ#3: デフォルトprovider="azure_openai"でAzure DB未構築時に
    未発見シナリオの質問・回答が全て空になる問題を検証
    """

    def test_default_provider_removed(self):
        """provider引数にデフォルト値がないことを確認"""
        import inspect
        evaluator = _make_evaluator()
        sig = inspect.signature(evaluator._fetch_scenario_content)
        # provider パラメータのデフォルト値が EMPTY（デフォルトなし）であること
        # または、デフォルトが "azure_openai" でないこと
        provider_param = sig.parameters.get("provider")
        if provider_param is not None and provider_param.default is not inspect.Parameter.empty:
            assert provider_param.default != "azure_openai", \
                "provider のデフォルト値が 'azure_openai' のままです"
```

**Step 2: テスト実行 — 失敗確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_unfound.py::TestFetchScenarioContentProvider -v`
Expected: FAIL（現在のデフォルトは `"azure_openai"` のため）

**Step 3: run_eval.py 修正 — `_fetch_scenario_content`のデフォルト削除 + 呼び出し元修正**

修正箇所1: `_fetch_scenario_content` のシグネチャ変更（行138-139）

```python
# 修正前:
    def _fetch_scenario_content(
        self, scenario_id: str, area: str, provider: str = "azure_openai"
    ) -> Optional[Dict[str, str]]:

# 修正後:
    def _fetch_scenario_content(
        self, scenario_id: str, area: str, provider: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
```

修正箇所2: メソッド冒頭でプロバイダー自動検出を追加（行141付近、docstringの直後）

```python
        """シナリオIDから質問・回答を取得"""
        try:
            # プロバイダー未指定時は自動検出
            if provider is None:
                provider = self._detect_available_provider(area)
                if provider is None:
                    return None

            bot_name, excel_row = scenario_id.rsplit("_", 1)
            # ... 以降は既存コードそのまま
```

**Step 4: テスト実行**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_run_eval_unfound.py tests/unit/test_run_eval_cache.py -v`
Expected: ALL PASS

**Step 5: 既存テスト全件実行**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: コミット**

```bash
git add apps/revision-ops/run_eval.py tests/unit/test_run_eval_unfound.py
git commit -m "fix: _fetch_scenario_contentのプロバイダー自動検出

デフォルトprovider='azure_openai'を削除し、未指定時は
_detect_available_providerで利用可能なDBを自動検出する。
Azure DB未構築環境で未発見シナリオの質問・回答が空になる問題を修正。"
```

---

### Task 4: 統合検証

**Step 1: 全テスト実行**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: 修正内容の最終確認**

以下を手動で確認:
1. `git diff 0e3070b..HEAD -- apps/revision-ops/run_eval.py` で修正差分を確認
2. `grep -n "azure_openai" apps/revision-ops/run_eval.py` で残存ハードコードを確認
3. 修正が3箇所のみに限定されていることを確認

**Step 3: 最終コミットメッセージの検討**

3件のバグ修正が個別コミットで完了していることを確認。
スカッシュする場合は以下:

```bash
git log --oneline -3
# 3コミットを確認後、必要に応じてスカッシュ
```
