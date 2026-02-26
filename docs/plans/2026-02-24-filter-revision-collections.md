# 回答支援AIから改定別コレクションを除外する修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 回答支援AI（answer-support）が改定別コレクション（rev*）にアクセスしないよう、DynamicDBManagerにフィルタリングを追加する

**Architecture:** `DynamicDBManager` に `include_revisions` パラメータを追加し、`get_all_business_areas()` と `analyze_reference_files()` で改定別エリア（`rev` プレフィックス）を除外可能にする。回答支援AI側の呼び出し箇所で `include_revisions=False` を指定する。事務改定評価AI（revision-eval）は DynamicDBManager を使用していないため影響なし。

**Tech Stack:** Python, pytest, ChromaDB

---

### Task 1: `get_all_business_areas()` に `include_revisions` パラメータ追加

**Files:**
- Modify: `rag-local/src/utils/dynamic_db_manager.py:867-898`
- Test: `rag-local/tests/unit/test_business_area_mapping.py`

**Step 1: テストを書く**

`rag-local/tests/unit/test_business_area_mapping.py` の末尾に追加:

```python
class TestGetAllBusinessAreasRevisionFilter:
    """get_all_business_areas の改定別コレクション除外テスト"""

    @pytest.fixture
    def db_manager_with_dirs(self, mock_config, tmp_path):
        """通常業務 + 改定別のDBディレクトリを持つDynamicDBManager"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        db_manager = DynamicDBManager(mock_config)
        db_manager.base_db_path = str(tmp_path)

        # 通常業務: smile/azure_openai/chroma.sqlite3
        for area in ["smile", "deposit", "general"]:
            db_dir = tmp_path / area / "azure_openai"
            db_dir.mkdir(parents=True)
            (db_dir / "chroma.sqlite3").touch()

        # 改定別: rev01_smile/azure_openai/chroma.sqlite3
        for area in ["rev01_smile", "rev02_souzoku", "rev03_naibujimu"]:
            db_dir = tmp_path / area / "azure_openai"
            db_dir.mkdir(parents=True)
            (db_dir / "chroma.sqlite3").touch()

        return db_manager

    def test_default_includes_revisions(self, db_manager_with_dirs):
        """デフォルトでは改定別コレクションを含む"""
        areas = db_manager_with_dirs.get_all_business_areas()
        assert "rev01_smile" in areas
        assert "smile" in areas

    def test_exclude_revisions(self, db_manager_with_dirs):
        """include_revisions=False で改定別コレクションを除外"""
        areas = db_manager_with_dirs.get_all_business_areas(include_revisions=False)
        assert "smile" in areas
        assert "deposit" in areas
        assert "general" in areas
        assert not any(a.startswith("rev") for a in areas)

    def test_include_revisions_explicit(self, db_manager_with_dirs):
        """include_revisions=True で改定別コレクションを含む"""
        areas = db_manager_with_dirs.get_all_business_areas(include_revisions=True)
        assert "rev01_smile" in areas
        assert "rev02_souzoku" in areas
        assert "smile" in areas
```

**Step 2: テストが失敗することを確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestGetAllBusinessAreasRevisionFilter -v`
Expected: FAIL — `get_all_business_areas() got an unexpected keyword argument 'include_revisions'`

**Step 3: 実装**

`rag-local/src/utils/dynamic_db_manager.py` の `get_all_business_areas` を修正:

```python
def get_all_business_areas(self, include_revisions: bool = True) -> List[str]:
    """全業務分野の一覧を取得（新旧両構造対応）

    Args:
        include_revisions: Falseの場合、改定別コレクション（rev*）を除外

    新構造: {business}/{provider}/chroma.sqlite3
    旧構造: {business}_DB/chroma.sqlite3
    """
    business_areas = set()

    if not os.path.exists(self.base_db_path):
        return list(business_areas)

    for item in os.listdir(self.base_db_path):
        item_path = os.path.join(self.base_db_path, item)
        if not os.path.isdir(item_path):
            continue

        # 旧構造: {business}_DB/
        if item.endswith('_DB'):
            business_area = item[:-3]  # "_DB"を除去
            if include_revisions or not business_area.startswith("rev"):
                business_areas.add(business_area)
            continue

        # 新構造: {business}/{provider}/chroma.sqlite3
        if not include_revisions and item.startswith("rev"):
            continue

        # プロバイダーサブディレクトリをチェック
        for provider in self.config.VALID_EMBEDDING_PROVIDERS:
            provider_path = os.path.join(item_path, provider)
            db_file = os.path.join(provider_path, "chroma.sqlite3")
            if os.path.exists(db_file):
                business_areas.add(item)
                break

    return sorted(list(business_areas))
```

**Step 4: テストが通ることを確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestGetAllBusinessAreasRevisionFilter -v`
Expected: PASS（3件全て）

**Step 5: 既存テストの回帰確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_business_area_mapping.py -v`
Expected: 全PASS

**Step 6: コミット**

```bash
git add rag-local/src/utils/dynamic_db_manager.py rag-local/tests/unit/test_business_area_mapping.py
git commit -m "feat: get_all_business_areas に include_revisions フィルタ追加"
```

---

### Task 2: `analyze_reference_files()` に `include_revisions` パラメータ追加

**Files:**
- Modify: `rag-local/src/utils/dynamic_db_manager.py:395-430`
- Test: `rag-local/tests/unit/test_business_area_mapping.py`

**Step 1: テストを書く**

`rag-local/tests/unit/test_business_area_mapping.py` に追加:

```python
class TestAnalyzeReferenceFilesRevisionFilter:
    """analyze_reference_files の改定別エリア除外テスト"""

    @pytest.fixture
    def db_manager_with_files(self, mock_config, tmp_path):
        """通常業務 + 改定別の参照ファイルを持つDynamicDBManager"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        db_manager = DynamicDBManager(mock_config)

        # シナリオディレクトリ（改定別ファイル）
        scenario_dir = tmp_path / "scenarios"
        scenario_dir.mkdir()
        db_manager.reference_scenario_path = str(scenario_dir)
        (scenario_dir / "rev01_smile_シナリオデータ_20260203.xlsx").touch()
        (scenario_dir / "rev02_souzoku_シナリオデータ_20260203.xlsx").touch()

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
```

**Step 2: テストが失敗することを確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestAnalyzeReferenceFilesRevisionFilter -v`
Expected: FAIL

**Step 3: 実装**

`rag-local/src/utils/dynamic_db_manager.py` の `analyze_reference_files` を修正:

```python
def analyze_reference_files(self, include_revisions: bool = True) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """参照ファイルを業務分野ごとに分類（DB互換名をキーとして返す）

    Args:
        include_revisions: Falseの場合、改定別エリア（rev*）を除外
    """
    logger.info("参照ファイルの分析を開始...")

    business_areas = {}

    # 履歴データの分析
    faq_files = self._get_files_in_directory(self.reference_faq_path)
    for file in faq_files:
        match = re.match(self.config.REFERENCE_FILE_PATTERN, file)
        if match:
            raw_business, data_type, date = match.groups()
            business = self._normalize_business_name(raw_business)
            if not include_revisions and business.startswith("rev"):
                continue
            if business not in business_areas:
                business_areas[business] = {"faq": [], "scenario": []}
            business_areas[business]["faq"].append((file, date))
            logger.info(f"履歴データ検出: {business} - {file}")
        else:
            logger.warning(f"不正な履歴データファイル名: {file}")

    # シナリオデータの分析
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

    logger.info(f"業務分野検出: {list(business_areas.keys())}")
    return business_areas
```

**Step 4: テストが通ることを確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestAnalyzeReferenceFilesRevisionFilter -v`
Expected: PASS（3件全て）

**Step 5: 全テスト回帰確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_business_area_mapping.py -v`
Expected: 全PASS

**Step 6: コミット**

```bash
git add rag-local/src/utils/dynamic_db_manager.py rag-local/tests/unit/test_business_area_mapping.py
git commit -m "feat: analyze_reference_files に include_revisions フィルタ追加"
```

---

### Task 3: 回答支援AI の呼び出し箇所を修正

**Files:**
- Modify: `rag-local/apps/answer-support/ui/chat.py:36-46`（UI業務分野リスト）
- Modify: `rag-local/apps/answer-support/ui/chat.py:69-126`（参照データ読み込み）
- Modify: `rag-local/apps/answer-support/main.py:92-126`（バッチDB更新）

**Step 1: `chat.py` の `get_available_business_areas()` を修正**

```python
@st.cache_data(ttl=60)
def get_available_business_areas() -> list:
    """利用可能な業務分野を動的に取得（改定別コレクションを除外）"""
    try:
        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        db_manager = DynamicDBManager(config)
        areas = db_manager.get_all_business_areas(include_revisions=False)
        if areas:
            return areas
    except Exception as e:
        logger.warning(f"業務分野一覧の取得に失敗: {e}")
    return DEFAULT_BUSINESS_AREAS
```

**Step 2: `chat.py` の `_load_reference_data_for_business()` を修正**

行72の `analyze_reference_files()` 呼び出しに `include_revisions=False` を追加:

```python
            business_areas = db_manager.analyze_reference_files(include_revisions=False)
```

**Step 3: `main.py` の `run_db_update()` を修正**

行99の `analyze_reference_files()` 呼び出しに `include_revisions=False` を追加:

```python
        reference_files = db_manager.analyze_reference_files(include_revisions=False)
```

**Step 4: `main.py` の `run_preflight()` を修正**

行60の `analyze_reference_files()` 呼び出しに `include_revisions=False` を追加:

```python
        reference_files = db_manager.analyze_reference_files(include_revisions=False)
```

**Step 5: 全テスト回帰確認**

Run: `cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v`
Expected: 全PASS

**Step 6: コミット**

```bash
git add rag-local/apps/answer-support/ui/chat.py rag-local/apps/answer-support/main.py
git commit -m "fix: 回答支援AIから改定別コレクション(rev*)を除外"
```
