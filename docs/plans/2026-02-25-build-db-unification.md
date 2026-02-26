# DB構築スクリプト統合 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `build_answer_support_db.py` と `rebuild_before_scenario_db.py` を `build_db.py` に統合し、DynamicDBManager が `scenarios/revisions/` もスキャンできるようにする。

**Architecture:** DynamicDBManager に `reference_revision_scenario_path` と `_get_scenario_base_path()` ヘルパーを追加。`analyze_reference_files()` を拡張して revisions/ をスキャン。統合スクリプト `build_db.py` で両プロバイダー対応 + 差分スキップ + フィルタオプションを提供。

**Tech Stack:** Python, ChromaDB, Azure OpenAI / VertexAI Embedding

**設計書:** `docs/plans/2026-02-25-build-db-unification-design.md`

---

### Task 1: DynamicDBManager に revisions パスとヘルパーを追加

**Files:**
- Modify: `rag-local/src/utils/dynamic_db_manager.py:25-36` (`__init__`)
- Modify: `rag-local/src/utils/dynamic_db_manager.py` (新メソッド追加)
- Test: `rag-local/tests/unit/test_business_area_mapping.py`

**Step 1: テストを書く — `_get_scenario_base_path()` の振る舞い**

`tests/unit/test_business_area_mapping.py` の末尾に追加:

```python
class TestGetScenarioBasePath:
    """_get_scenario_base_path が業務分野に応じた正しいパスを返すことを検証"""

    @pytest.fixture
    def db_manager(self, mock_config):
        from src.utils.dynamic_db_manager import DynamicDBManager
        return DynamicDBManager(mock_config)

    def test_regular_area_returns_latest(self, db_manager):
        """通常業務は scenarios/latest/ パスを返す"""
        path = db_manager._get_scenario_base_path("smile")
        assert path == db_manager.reference_scenario_path
        assert "latest" in path

    def test_revision_area_returns_revisions(self, db_manager):
        """改定別は scenarios/revisions/ パスを返す"""
        path = db_manager._get_scenario_base_path("rev01_smile")
        assert path == db_manager.reference_revision_scenario_path
        assert "revisions" in path

    def test_naibujimu_returns_latest(self, db_manager):
        path = db_manager._get_scenario_base_path("naibujimu")
        assert "latest" in path

    def test_rev03_torikaku_returns_revisions(self, db_manager):
        path = db_manager._get_scenario_base_path("rev03_torikaku")
        assert "revisions" in path
```

**Step 2: テスト実行 — 失敗を確認**

Run: `cd rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestGetScenarioBasePath -v`
Expected: FAIL — `AttributeError: 'DynamicDBManager' object has no attribute '_get_scenario_base_path'`

**Step 3: DynamicDBManager に実装を追加**

`src/utils/dynamic_db_manager.py` の `__init__` に追加（line 30 付近、`reference_scenario_path` の直後）:

```python
self.reference_revision_scenario_path = os.path.join(config.base_dir, "data", "source", "scenarios", "revisions")
```

`os.makedirs` セクション（line 35 付近）に追加:

```python
os.makedirs(self.reference_revision_scenario_path, exist_ok=True)
```

新メソッドを追加（`_normalize_business_name` の近く、line 393 付近）:

```python
def _get_scenario_base_path(self, business_area: str) -> str:
    """業務分野名に応じたシナリオベースパスを返す

    Args:
        business_area: 業務分野名（正規化済み）

    Returns:
        str: scenarios/latest/ または scenarios/revisions/ のパス
    """
    if business_area.startswith("rev"):
        return self.reference_revision_scenario_path
    return self.reference_scenario_path
```

**Step 4: テスト実行 — パスを確認**

Run: `cd rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestGetScenarioBasePath -v`
Expected: PASS (4 tests)

**Step 5: コミット**

```bash
git add rag-local/src/utils/dynamic_db_manager.py rag-local/tests/unit/test_business_area_mapping.py
git commit -m "feat: DynamicDBManager に revisions パスと _get_scenario_base_path ヘルパー追加"
```

---

### Task 2: analyze_reference_files() で revisions/ もスキャンする

**Files:**
- Modify: `rag-local/src/utils/dynamic_db_manager.py:395-438` (`analyze_reference_files`)
- Test: `rag-local/tests/unit/test_business_area_mapping.py`

**Step 1: 既存テストを修正 — `TestAnalyzeReferenceFilesRevisionFilter` を2ディレクトリ構成に変更**

`test_business_area_mapping.py` の `TestAnalyzeReferenceFilesRevisionFilter` (line 200-243) を修正。
現在は rev* ファイルを `reference_scenario_path`（= latest/）に置いているが、`reference_revision_scenario_path`（= revisions/）に置く形に変更:

```python
class TestAnalyzeReferenceFilesRevisionFilter:
    """analyze_reference_files の改定別エリア除外テスト"""

    @pytest.fixture
    def db_manager_with_files(self, mock_config, tmp_path):
        """通常業務 + 改定別の参照ファイルを持つDynamicDBManager"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        db_manager = DynamicDBManager(mock_config)

        # シナリオ latest ディレクトリ（通常業務ファイル）
        scenario_latest_dir = tmp_path / "scenarios" / "latest"
        scenario_latest_dir.mkdir(parents=True)
        db_manager.reference_scenario_path = str(scenario_latest_dir)
        (scenario_latest_dir / "スマイル_シナリオデータ_20260224.xlsx").touch()

        # シナリオ revisions ディレクトリ（改定別ファイル）
        scenario_rev_dir = tmp_path / "scenarios" / "revisions"
        scenario_rev_dir.mkdir(parents=True)
        db_manager.reference_revision_scenario_path = str(scenario_rev_dir)
        (scenario_rev_dir / "rev01_smile_シナリオデータ_20260203.xlsx").touch()
        (scenario_rev_dir / "rev02_souzoku_シナリオデータ_20260203.xlsx").touch()

        # FAQディレクトリ（通常業務ファイル）
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        db_manager.reference_faq_path = str(faq_dir)
        (faq_dir / "預金_履歴データ_20250830.xlsx").touch()
        (faq_dir / "スマイル_履歴データ_20250205.xlsx").touch()

        return db_manager

    def test_default_includes_revisions(self, db_manager_with_files):
        """デフォルトでは改定別エリアを含む"""
        areas = db_manager_with_files.analyze_reference_files()
        assert "rev01_smile" in areas
        assert "deposit" in areas
        assert "smile" in areas

    def test_exclude_revisions(self, db_manager_with_files):
        """include_revisions=False で改定別エリアを除外"""
        areas = db_manager_with_files.analyze_reference_files(include_revisions=False)
        assert "deposit" in areas
        assert "smile" in areas
        assert not any(k.startswith("rev") for k in areas)

    def test_include_revisions_explicit(self, db_manager_with_files):
        """include_revisions=True で改定別エリアを含む"""
        areas = db_manager_with_files.analyze_reference_files(include_revisions=True)
        assert "rev01_smile" in areas
        assert "deposit" in areas
        assert "smile" in areas
```

**Step 2: テスト実行 — 失敗を確認**

Run: `cd rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestAnalyzeReferenceFilesRevisionFilter -v`
Expected: FAIL — `test_default_includes_revisions` で `rev01_smile` が見つからない（revisions/ をスキャンしていないため）

**Step 3: `analyze_reference_files()` を修正**

`src/utils/dynamic_db_manager.py` の `analyze_reference_files()` メソッド内、シナリオデータ分析セクション（line 421-435 付近）を以下に置換:

```python
        # シナリオデータの分析（latest/）
        scenario_files = self._get_files_in_directory(self.reference_scenario_path)
        for file in scenario_files:
            match = re.match(self.config.REFERENCE_FILE_PATTERN, file)
            if match:
                raw_business, data_type, date = match.groups()
                business = self._normalize_business_name(raw_business)
                if not include_revisions and business.startswith("rev"):
                    continue
                if business not in business_areas:
                    business_areas[business] = {"faq": [], "scenario": []}
                business_areas[business]["scenario"].append((file, date))
                logger.info(f"シナリオデータ検出: {business} - {file}")
            else:
                logger.warning(f"不正なシナリオデータファイル名: {file}")

        # シナリオデータの分析（revisions/）— include_revisions=True の場合のみ
        if include_revisions:
            revision_scenario_files = self._get_files_in_directory(self.reference_revision_scenario_path)
            for file in revision_scenario_files:
                match = re.match(self.config.REFERENCE_FILE_PATTERN, file)
                if match:
                    raw_business, data_type, date = match.groups()
                    business = self._normalize_business_name(raw_business)
                    if business not in business_areas:
                        business_areas[business] = {"faq": [], "scenario": []}
                    business_areas[business]["scenario"].append((file, date))
                    logger.info(f"シナリオデータ検出（revisions）: {business} - {file}")
                else:
                    logger.warning(f"不正なシナリオデータファイル名: {file}")
```

**Step 4: テスト実行 — パスを確認**

Run: `cd rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestAnalyzeReferenceFilesRevisionFilter -v`
Expected: PASS (3 tests)

**Step 5: 全テスト実行 — 既存テストが壊れていないことを確認**

Run: `cd rag-local && python -m pytest tests/unit/test_business_area_mapping.py -v`
Expected: ALL PASS

**Step 6: コミット**

```bash
git add rag-local/src/utils/dynamic_db_manager.py rag-local/tests/unit/test_business_area_mapping.py
git commit -m "feat: analyze_reference_files が scenarios/revisions/ もスキャンするよう拡張"
```

---

### Task 3: DynamicDBManager 内の reference_scenario_path 参照箇所を修正

**Files:**
- Modify: `rag-local/src/utils/dynamic_db_manager.py` (4箇所)

`reference_scenario_path` をハードコードで使用している箇所を `_get_scenario_base_path()` に差し替える。

修正対象（全4箇所）:

**箇所1: `_update_timestamps_after_success()` (line 313)**
```python
# Before:
scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
# After:
scenario_base = self._get_scenario_base_path(business_area)
scenario_path = os.path.join(scenario_base, latest_scenario)
```

**箇所2: `needs_update()` (line 495-498)**
```python
# Before:
scenario_needs_update = self._check_file_needs_update(
    latest_scenario, self.reference_scenario_path,
# After:
scenario_base = self._get_scenario_base_path(business_area)
scenario_needs_update = self._check_file_needs_update(
    latest_scenario, scenario_base,
```

**箇所3: `preflight_business_db()` (line 610)**
```python
# Before:
scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
# After:
scenario_base = self._get_scenario_base_path(business_area)
scenario_path = os.path.join(scenario_base, latest_scenario)
```

**箇所4: `_prepare_reference_data_for_vectorization()` (line 946)**
```python
# Before:
scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
# After:
# business_area を引数に追加する必要がある → 呼び出し元の _vectorize_data も修正
scenario_base = self._get_scenario_base_path(business_area)
scenario_path = os.path.join(scenario_base, latest_scenario)
```

**注意**: `_prepare_reference_data_for_vectorization()` に `business_area` 引数を追加する必要がある。
現在のシグネチャ: `_prepare_reference_data_for_vectorization(self, latest_scenario, latest_faq)`
変更後: `_prepare_reference_data_for_vectorization(self, latest_scenario, latest_faq, business_area=None)`

呼び出し元の `_vectorize_data()` (line 758) も修正:
```python
# Before:
reference_data = self._prepare_reference_data_for_vectorization(latest_scenario, latest_faq)
# After:
reference_data = self._prepare_reference_data_for_vectorization(latest_scenario, latest_faq, business_area)
```

**Step 1: 全テスト実行 — 修正前のベースライン**

Run: `cd rag-local && python -m pytest tests/unit/ -v`
Expected: ALL PASS

**Step 2: 上記4箇所 + `_prepare_reference_data_for_vectorization` シグネチャ変更を実施**

**Step 3: 全テスト実行 — リグレッションなし**

Run: `cd rag-local && python -m pytest tests/unit/ -v`
Expected: ALL PASS

**Step 4: コミット**

```bash
git add rag-local/src/utils/dynamic_db_manager.py
git commit -m "refactor: DynamicDBManager の scenario パス参照を _get_scenario_base_path に統一"
```

---

### Task 4: build_db.py を作成

**Files:**
- Create: `rag-local/scripts/build_db.py`

**Step 1: `build_db.py` を作成**

`build_answer_support_db.py` をベースに、以下を変更:
- 両プロバイダーループ (azure_openai + vertex_ai)
- `include_revisions=True` をデフォルトに
- `--revisions-only`, `--no-revisions` フィルタ追加
- サマリにプロバイダー列を追加

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DB構築スクリプト（Azure OpenAI + VertexAI 両プロバイダー対応）

使用方法:
    python scripts/build_db.py                          # 全業務分野（差分のみ構築）
    python scripts/build_db.py --force                   # 全業務分野（全再構築）
    python scripts/build_db.py --business naibujimu      # 特定業務分野のみ
    python scripts/build_db.py --revisions-only          # 改定別（rev*）のみ
    python scripts/build_db.py --no-revisions            # 通常業務のみ
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import SearchConfig
from src.utils.dynamic_db_manager import DynamicDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PROVIDERS = ["azure_openai", "vertex_ai"]

EMBEDDING_MODELS = {
    "azure_openai": ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
    "vertex_ai": ("VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB構築（Azure OpenAI + VertexAI）")
    parser.add_argument("--force", action="store_true", help="既存DBを削除して全再構築")
    parser.add_argument("--business", type=str, default=None,
                        help="構築対象の業務分野（例: naibujimu, smile, rev02_souzoku）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revisions-only", action="store_true", help="改定別（rev*）のみ")
    group.add_argument("--no-revisions", action="store_true", help="通常業務のみ（rev*除外）")
    return parser.parse_args()


def delete_existing_dbs(db_base: Path, target_areas: list[str]) -> bool:
    for area in target_areas:
        db_path = db_base / area
        if not db_path.exists():
            continue
        try:
            shutil.rmtree(db_path)
            logger.info(f"削除完了: {db_path}")
        except Exception as e:
            logger.error(f"削除エラー: {db_path} - {e}")
            return False
    return True


def reset_timestamps(timestamp_file: Path, target_areas: list[str]) -> bool:
    if not timestamp_file.exists():
        return True
    try:
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamps = json.load(f)
        keys_to_remove = [k for k in timestamps
                          if any(k.startswith(f"{area}_") for area in target_areas)]
        for key in keys_to_remove:
            del timestamps[key]
            logger.info(f"タイムスタンプ削除: {key}")
        with open(timestamp_file, "w", encoding="utf-8") as f:
            json.dump(timestamps, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"タイムスタンプ更新エラー: {e}")
        return False


def build_dbs(args: argparse.Namespace) -> None:
    all_results = []
    include_revisions = not args.no_revisions

    for provider in PROVIDERS:
        print(f"\n{'=' * 60}")
        print(f"=== {provider} ===")
        print(f"{'=' * 60}")

        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        config.embedding_provider = provider
        env_key, default_model = EMBEDDING_MODELS[provider]
        config.embedding_model = os.getenv(env_key, default_model)

        with DynamicDBManager(config) as db_manager:
            business_areas = db_manager.analyze_reference_files(
                include_revisions=include_revisions)

            # --revisions-only: rev* のみに絞る
            if args.revisions_only:
                business_areas = {k: v for k, v in business_areas.items()
                                  if k.startswith("rev")}

            # --business: 指定分野のみ
            if args.business:
                if args.business not in business_areas:
                    print(f"エラー: '{args.business}' が見つかりません")
                    print(f"利用可能: {list(business_areas.keys())}")
                    sys.exit(1)
                business_areas = {args.business: business_areas[args.business]}

            target_areas = list(business_areas.keys())
            logger.info(f"対象業務分野: {target_areas}")

            # --force: DB削除 + タイムスタンプリセット
            if args.force:
                db_base = PROJECT_ROOT / "data" / "vector_db"
                timestamp_file = db_base / "update_timestamps.json"
                print(f"  [Force] 既存DB削除（{len(target_areas)}件）...")
                if not delete_existing_dbs(db_base, target_areas):
                    print("エラー: DB削除失敗。Streamlit UIが実行中の場合は停止してください")
                    sys.exit(1)
                print("  [Force] タイムスタンプリセット...")
                if not reset_timestamps(timestamp_file, target_areas):
                    print("エラー: タイムスタンプリセット失敗")
                    sys.exit(1)

            for area in target_areas:
                print(f"\n--- {area} ({provider}) ---")
                start_time = time.time()
                try:
                    db_manager.update_business_db(area, business_areas[area])
                    elapsed = time.time() - start_time
                    db_path = db_manager.get_db_path_for_business(area)
                    sqlite_path = os.path.join(db_path, "chroma.sqlite3")
                    all_results.append({
                        "area": area, "provider": provider,
                        "status": "OK" if os.path.exists(sqlite_path) else "WARN",
                        "elapsed": elapsed,
                    })
                except Exception as e:
                    elapsed = time.time() - start_time
                    all_results.append({
                        "area": area, "provider": provider,
                        "status": "ERROR", "elapsed": elapsed, "error": str(e),
                    })
                    logger.error(f"{area} ({provider}) エラー: {e}")

        print(f"\n=== {provider} 完了 ===")

    # サマリ
    print("\n" + "=" * 60)
    print("構築結果サマリ")
    print("=" * 60)
    print(f"{'業務分野':<20} {'プロバイダー':<15} {'ステータス':<10} {'所要時間':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['area']:<20} {r['provider']:<15} {r['status']:<10} {r['elapsed']:.1f}秒")
    print("=" * 60)

    if any(r["status"] == "ERROR" for r in all_results):
        sys.exit(1)


def main() -> None:
    args = parse_args()
    mode = "全再構築（--force）" if args.force else "差分構築"
    scope = (args.business if args.business
             else "改定別のみ" if args.revisions_only
             else "通常業務のみ" if args.no_revisions
             else "全業務分野")
    print("=" * 60)
    print(f"DB構築スクリプト | モード: {mode} | 対象: {scope}")
    print("=" * 60)
    build_dbs(args)


if __name__ == "__main__":
    main()
```

**Step 2: 動作確認（ドライラン）**

Run: `cd rag-local && python scripts/build_db.py --help`
Expected: ヘルプメッセージが表示される

**Step 3: コミット**

```bash
git add rag-local/scripts/build_db.py
git commit -m "feat: 統合DB構築スクリプト build_db.py を追加"
```

---

### Task 5: 旧スクリプトを削除し、ドキュメントを更新

**Files:**
- Delete: `rag-local/scripts/build_answer_support_db.py`
- Delete: `rag-local/scripts/rebuild_before_scenario_db.py`
- Delete: `rag-local/scripts/rebuild_faq_db.py` (build_db.py に包含)
- Modify: `rag-local/docs/DB_BUILD_GUIDE.md`
- Modify: `rag-local/CLAUDE.md` (スクリプト名の参照更新)

**Step 1: 旧スクリプトを削除**

```bash
git rm rag-local/scripts/build_answer_support_db.py
git rm rag-local/scripts/rebuild_before_scenario_db.py
git rm rag-local/scripts/rebuild_faq_db.py
```

**Step 2: DB_BUILD_GUIDE.md を更新**

`build_answer_support_db.py` → `build_db.py` の参照を更新。使用例を統合スクリプトに合わせる。

**Step 3: CLAUDE.md を更新**

`scripts/build_answer_support_db.py` の参照を `scripts/build_db.py` に更新。

**Step 4: ドキュメント内の参照を検索して漏れがないか確認**

Run: `grep -r "build_answer_support_db\|rebuild_before_scenario_db\|rebuild_faq_db" rag-local/ --include="*.md" --include="*.py" -l`

**Step 5: コミット**

```bash
git add -A rag-local/scripts/ rag-local/docs/DB_BUILD_GUIDE.md rag-local/CLAUDE.md
git commit -m "refactor: 旧DB構築スクリプト3本を削除し build_db.py に統合"
```

---

### Task 6: 全テスト実行 + 最終確認

**Files:** なし（テスト実行のみ）

**Step 1: 全ユニットテスト実行**

Run: `cd rag-local && python -m pytest tests/unit/ -v`
Expected: ALL PASS

**Step 2: build_db.py の動作確認（--no-revisions で通常業務のみ）**

Run: `cd rag-local && python scripts/build_db.py --no-revisions`
Expected: naibujimu, smile が差分チェック → DBは最新ですでスキップ

**Step 3: build_db.py の動作確認（--revisions-only で改定別のみ）**

Run: `cd rag-local && python scripts/build_db.py --revisions-only`
Expected: rev01〜rev06 が差分チェック → DBは最新ですでスキップ（または構築）

**Step 4: 最終コミット（必要な場合のみ）**

テストで修正が必要だった場合のみコミット。
