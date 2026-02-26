# keyword_filter ChromaDB化 + 影響調査モード Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Excel直読みのkeyword_filter検索をChromaDB検索に置換し、影響調査モードを追加する

**Architecture:** `src/core/search/chromadb_keyword_search.py` に共通検索モジュールを新設。バッチ版（`evaluate_revisions.py`）とUI版（`eval_ui.py`）の両方からこのモジュールを呼ぶ。影響調査モードはUI版のみに追加。

**Tech Stack:** ChromaDB (PersistentClient), KeywordSearchEngine (Sudachi), TextCombiner (既存パーサー)

**Design Doc:** `docs/plans/2026-02-26-keyword-filter-chromadb-design.md`

---

## 重要な前提知識

### ChromaDB ディレクトリ構造
```
data/vector_db/
├── chroma.sqlite3          ← トップレベル（naibujimu, smile用）
├── naibujimu/azure_openai/ ← プロバイダー別サブDB
├── smile/azure_openai/
├── rev02_souzoku/azure_openai/  ← 改定別
├── rev03_smile/azure_openai/
└── ...
```
各 `azure_openai/` ディレクトリが独立した ChromaDB インスタンス（`chroma.sqlite3` + コレクション `default`）。

### 結果フォーマットの違い
- **バッチ版**: 日本語キー `{"順位", "シナリオID", "類似度", "マッチ種別", "正解フラグ", "質問", "回答", ...}`
- **UI版**: 英語キー `{"Similarity", "Search_Result_Q", "Search_Result_A", "Search_Category", "Sheet_Name", "Row_Index", ...}`
- **共通モジュール**: 生データ（`MatchResult` dataclass）を返し、フォーマット変換は呼び出し側で行う

### 既存ユーティリティ（再利用）
- `TextCombiner.parse()`: `分類: X | 質問: Y | 回答: Z` → `ParsedCombinedText(hierarchy, query, answer)`
- `KeywordSearchEngine.extract_keywords()`: Sudachiでキーワード抽出
- `AREA_TO_BOT`: area名 → bot名マッピング（`settings.yaml:192`）
- `AREA_TO_CATEGORY`: area名 → 日本語カテゴリ名（`settings.yaml:202`）

---

## Task 1: 共通モジュールのテスト作成

**Files:**
- Create: `rag-local/tests/unit/test_chromadb_keyword_search.py`

**Step 1: テストファイルを作成**

```python
"""ChromaDBKeywordSearcher のユニットテスト"""
import pytest
import chromadb
from unittest.mock import MagicMock
from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher, MatchResult


@pytest.fixture
def in_memory_db(tmp_path):
    """テスト用ChromaDBをtmp_pathに作成し、サンプルデータを投入"""
    db_path = str(tmp_path / "test_db")
    client = chromadb.PersistentClient(path=db_path)

    # rev02_souzoku/azure_openai 相当のコレクション
    import os
    sub_db_path = str(tmp_path / "test_db" / "rev02_souzoku" / "azure_openai")
    os.makedirs(sub_db_path, exist_ok=True)
    sub_client = chromadb.PersistentClient(path=sub_db_path)
    col = sub_client.get_or_create_collection("default")
    col.add(
        documents=[
            "分類: 預金 > 少額払い | 質問: 相続の届出印は全て押印するのか | 回答: はい、全て押印してください",
            "分類: 預金 > 解約 | 質問: 通帳を持参していない場合の対応 | 回答: 本人確認書類で代替可能です",
            "分類: 預金 > 新規 | 質問: 口座開設の必要書類 | 回答: 身分証明書と印鑑が必要です",
        ],
        metadatas=[
            {"source": "scenario", "sheet_name": "預金", "row_index": 10, "hierarchy": "預金 > 少額払い"},
            {"source": "scenario", "sheet_name": "預金", "row_index": 20, "hierarchy": "預金 > 解約"},
            {"source": "scenario", "sheet_name": "預金", "row_index": 30, "hierarchy": "預金 > 新規"},
        ],
        ids=["doc_1", "doc_2", "doc_3"],
    )
    return tmp_path / "test_db"


@pytest.fixture
def mock_keyword_engine():
    """KeywordSearchEngine のモック"""
    engine = MagicMock()
    engine.extract_keywords.return_value = ["届出印", "押印"]
    return engine


class TestChromaDBKeywordSearcher:
    def test_single_collection_search(self, in_memory_db, mock_keyword_engine):
        """単一コレクションでキーワードヒットあり"""
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=mock_keyword_engine,
            area_to_bot={"souzoku": "souzoku-bot"},
            area_to_category={"souzoku": "相続"},
        )
        results = searcher.search(
            collection_names=["rev02_souzoku"],
            query="届出印の押印",
            provider="azure_openai",
        )
        assert len(results) >= 1
        # 「届出印」「押印」の両方がヒットするドキュメントが最上位
        assert results[0].match_count == 2
        assert results[0].row_index == 10
        assert results[0].scenario_id == "souzoku-bot_12"  # 10 + 2

    def test_no_keywords_returns_empty(self, in_memory_db):
        """キーワード抽出結果が空 → 空リスト"""
        engine = MagicMock()
        engine.extract_keywords.return_value = []
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=engine,
            area_to_bot={},
            area_to_category={},
        )
        results = searcher.search(["rev02_souzoku"], "あ", provider="azure_openai")
        assert results == []

    def test_collection_not_found(self, in_memory_db, mock_keyword_engine):
        """存在しないコレクション → 警告ログ、例外なし"""
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=mock_keyword_engine,
            area_to_bot={},
            area_to_category={},
        )
        results = searcher.search(["nonexistent_collection"], "届出印", provider="azure_openai")
        assert results == []

    def test_results_sorted_by_match_count(self, in_memory_db):
        """マッチ数降順ソート"""
        engine = MagicMock()
        engine.extract_keywords.return_value = ["通帳", "届出印"]
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=engine,
            area_to_bot={"souzoku": "souzoku-bot"},
            area_to_category={"souzoku": "相続"},
        )
        results = searcher.search(["rev02_souzoku"], "通帳 届出印", provider="azure_openai")
        # 各ドキュメントのマッチ数が降順であること
        for i in range(len(results) - 1):
            assert results[i].match_count >= results[i + 1].match_count

    def test_scenario_id_excel_row_compatibility(self, in_memory_db, mock_keyword_engine):
        """row_index + 2 = Excel行番号 のID互換性"""
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=mock_keyword_engine,
            area_to_bot={"souzoku": "souzoku-bot"},
            area_to_category={"souzoku": "相続"},
        )
        results = searcher.search(["rev02_souzoku"], "届出印", provider="azure_openai")
        for r in results:
            assert r.scenario_id == f"souzoku-bot_{r.row_index + 2}"

    def test_match_result_has_required_fields(self, in_memory_db, mock_keyword_engine):
        """MatchResult に必要なフィールドがすべて含まれること"""
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=mock_keyword_engine,
            area_to_bot={"souzoku": "souzoku-bot"},
            area_to_category={"souzoku": "相続"},
        )
        results = searcher.search(["rev02_souzoku"], "届出印", provider="azure_openai")
        assert len(results) >= 1
        r = results[0]
        # 必須フィールド確認
        assert hasattr(r, "question")
        assert hasattr(r, "answer")
        assert hasattr(r, "similarity")
        assert hasattr(r, "scenario_id")
        assert hasattr(r, "sheet_name")
        assert hasattr(r, "row_index")
        assert hasattr(r, "match_count")
        assert hasattr(r, "collection_name")
```

**Step 2: テスト実行 → FAIL を確認**

```bash
cd rag-local && python -m pytest tests/unit/test_chromadb_keyword_search.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.core.search.chromadb_keyword_search'`

**Step 3: コミット**

```bash
git add tests/unit/test_chromadb_keyword_search.py
git commit -m "test: ChromaDBKeywordSearcher のユニットテスト追加"
```

---

## Task 2: 共通モジュール実装

**Files:**
- Create: `rag-local/src/core/search/chromadb_keyword_search.py`
- Modify: `rag-local/src/core/search/__init__.py` — exports に追加

**Step 1: `chromadb_keyword_search.py` を作成**

```python
"""ChromaDB全件取得 + キーワードマッチング

評価モード（rev*コレクション）と影響調査モード（naibujimu, smile）の
両方で使用する共通キーワード検索モジュール。

ChromaDB の collection.get() で全件取得し、Python側でキーワードマッチングを行う。
"""
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError as ChromaNotFoundError

from src.core.search.keyword_search_engine import KeywordSearchEngine
from src.core.search.text_combiner import get_text_combiner

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """キーワードマッチ結果"""
    question: str
    answer: str
    hierarchy: str
    similarity: float
    match_count: int
    scenario_id: str
    sheet_name: str
    row_index: int
    collection_name: str
    source: str  # "scenario" or "history_data"


class ChromaDBKeywordSearcher:
    """ChromaDB全件取得 + キーワードマッチング

    Args:
        base_db_path: vector_db ディレクトリのパス（例: data/vector_db）
        keyword_engine: KeywordSearchEngine インスタンス
        area_to_bot: area名 → bot名マッピング
        area_to_category: area名 → 日本語カテゴリ名マッピング
    """

    def __init__(
        self,
        base_db_path: str,
        keyword_engine: KeywordSearchEngine,
        area_to_bot: Dict[str, str],
        area_to_category: Dict[str, str],
    ):
        self.base_db_path = base_db_path
        self.keyword_engine = keyword_engine
        self.area_to_bot = area_to_bot
        self.area_to_category = area_to_category
        self._text_combiner = get_text_combiner()

    def search(
        self,
        collection_names: List[str],
        query: str,
        provider: str = "azure_openai",
        max_results: int = 50,
    ) -> List[MatchResult]:
        """ChromaDBからキーワード検索を実行

        Args:
            collection_names: 検索対象コレクション名リスト
                評価モード: ["rev02_souzoku"] 等
                影響調査モード: ["naibujimu", "smile"] 等
            query: 検索クエリ
            provider: 埋め込みプロバイダー（DBディレクトリ構造用）
            max_results: 最大結果件数

        Returns:
            MatchResult のリスト（マッチ数降順）
        """
        keywords = self.keyword_engine.extract_keywords(query)
        if not keywords:
            logger.info("キーワード抽出結果が空です")
            return []

        logger.info(f"抽出キーワード: {keywords}")
        all_results: List[MatchResult] = []

        for col_name in collection_names:
            results = self._search_collection(col_name, keywords, provider)
            all_results.extend(results)

        all_results.sort(key=lambda r: r.match_count, reverse=True)
        return all_results[:max_results]

    def _search_collection(
        self,
        collection_name: str,
        keywords: List[str],
        provider: str,
    ) -> List[MatchResult]:
        """単一コレクションからキーワード検索"""
        db_path = os.path.join(self.base_db_path, collection_name, provider)
        if not os.path.exists(db_path):
            logger.warning(f"DBパスが存在しません: {db_path}")
            return []

        try:
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection("default")
        except (ChromaNotFoundError, Exception) as e:
            logger.warning(f"コレクション取得エラー ({collection_name}/{provider}): {e}")
            return []

        result = collection.get(include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not documents:
            logger.info(f"{collection_name}: ドキュメントなし")
            return []

        # area名を抽出（rev02_souzoku → souzoku, naibujimu → naibujimu）
        area = self._extract_area(collection_name)
        bot_name = self._resolve_bot_name(area)
        total_keywords = len(keywords)

        matched: List[MatchResult] = []
        for doc, meta in zip(documents, metadatas):
            doc_lower = doc.lower()
            match_count = sum(1 for kw in keywords if kw.lower() in doc_lower)
            if match_count == 0:
                continue

            parsed = self._text_combiner.parse(doc)
            row_index = meta.get("row_index", 0)
            excel_row = int(row_index) + 2
            scenario_id = f"{bot_name}_{excel_row}"

            matched.append(MatchResult(
                question=parsed.query,
                answer=parsed.answer,
                hierarchy=parsed.hierarchy,
                similarity=round(match_count / total_keywords, 4),
                match_count=match_count,
                scenario_id=scenario_id,
                sheet_name=meta.get("sheet_name", ""),
                row_index=int(row_index),
                collection_name=collection_name,
                source=meta.get("source", "unknown"),
            ))

        logger.info(f"{collection_name}: {len(matched)}件ヒット（キーワード検索）")
        return matched

    def _extract_area(self, collection_name: str) -> str:
        """コレクション名からarea名を抽出（rev02_souzoku → souzoku）"""
        # revXX_ プレフィックスを除去
        parts = collection_name.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("rev"):
            return parts[1]
        return collection_name

    def _resolve_bot_name(self, area: str) -> str:
        """area名からbot名を解決"""
        area_lower = area.lower()
        for keyword, bot_name in self.area_to_bot.items():
            if keyword in area_lower:
                return bot_name
        return "unknown-bot"
```

**Step 2: `__init__.py` に export 追加**

`rag-local/src/core/search/__init__.py` の `__all__` リストと lazy import に `ChromaDBKeywordSearcher`, `MatchResult` を追加。

**Step 3: テスト実行 → PASS を確認**

```bash
cd rag-local && python -m pytest tests/unit/test_chromadb_keyword_search.py -v
```
Expected: 全テスト PASS

**Step 4: コミット**

```bash
git add src/core/search/chromadb_keyword_search.py src/core/search/__init__.py
git commit -m "feat: ChromaDBKeywordSearcher 共通モジュール追加"
```

---

## Task 3: バッチ版の置換

**Files:**
- Modify: `rag-local/apps/revision-eval/evaluate_revisions.py:273-397`

**Step 1: `_execute_keyword_filter_search` を置換**

変更内容:
1. `_load_scenario_excel()` メソッド（273-283行）を削除
2. `_execute_keyword_filter_search()` メソッド（285-397行）を書き換え:

```python
def _execute_keyword_filter_search(
    self, revision: str, query: str, correct_ids: List[str]
) -> Tuple[Dict[str, List[Dict]], str, List[str], List[str]]:
    """キーワード必須検索（ChromaDB）"""
    areas = REVISION_TO_AREAS.get(revision, [])
    if not areas:
        logger.warning(f"改定 {revision} に対応するエリアがありません")
        return {}, "", [], []

    keyword_engine = KeywordSearchEngine(
        stop_words=self.config.STOP_WORDS,
        position_weight=self.config.POSITION_WEIGHT,
    )
    keywords = keyword_engine.extract_keywords(query)
    logger.info(f"  抽出キーワード: {keywords}")

    searcher = ChromaDBKeywordSearcher(
        base_db_path=str(VECTOR_DB_BASE),
        keyword_engine=keyword_engine,
        area_to_bot=AREA_TO_BOT,
        area_to_category=AREA_TO_CATEGORY,
    )

    collection_names = [f"rev{revision}_{area}" for area in areas]
    # 注意: revisionは既に "02" 等の形式。collection_nameは "rev02_souzoku" になる
    # REVISION_TO_AREAS の設定に依存するので、実際のコレクション名と一致するか要確認
    all_matches = searcher.search(collection_names, query, provider="azure_openai")

    # 結果をバッチ版フォーマットに変換（area別に分割）
    results_by_area = {}
    searched_areas = []
    for area in areas:
        area_matches = [m for m in all_matches if self._extract_area(m.collection_name) == area]
        if not area_matches:
            continue

        area_results = []
        for i, m in enumerate(area_matches[:MAX_RESULTS]):
            source_file = self._get_source_file(revision, m.scenario_id.rsplit("_", 1)[0], "")
            area_results.append({
                "順位": i + 1,
                "シナリオID": m.scenario_id,
                "類似度": m.similarity,
                "マッチ種別": "Keyword",
                "正解フラグ": "TRUE" if m.scenario_id in correct_ids else "FALSE",
                "質問": m.question,
                "回答": m.answer,
                "関連性判定": "",
                "判定根拠": "",
                "ソースファイル": source_file,
            })
        results_by_area[area] = area_results
        searched_areas.append(area)
        logger.info(f"  {area}: {len(area_results)}件取得（キーワード検索）")

    return results_by_area, "", keywords, searched_areas

@staticmethod
def _extract_area(collection_name: str) -> str:
    """コレクション名からarea名を抽出"""
    parts = collection_name.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("rev"):
        return parts[1]
    return collection_name
```

3. import追加: `from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher`
4. `SCENARIO_DIR` 定数（99行）を削除

**Step 2: 既存テスト実行**

```bash
cd rag-local && python -m pytest tests/unit/ -v
```
Expected: 既存テスト PASS（keyword_filter の直接テストがない場合は変更による退行なし）

**Step 3: コミット**

```bash
git add apps/revision-eval/evaluate_revisions.py
git commit -m "refactor: バッチ版keyword_filterをChromaDB化"
```

---

## Task 4: UI版の置換

**Files:**
- Modify: `rag-local/apps/revision-eval/ui/eval_ui.py:175-267`

**Step 1: `_execute_keyword_filter_search` を置換**

変更内容:
1. `_execute_keyword_filter_search()` 関数（175-267行）を書き換え:

```python
def _execute_keyword_filter_search(query: str, revision: str, areas: List[str]) -> List[Dict]:
    """キーワード検索（ChromaDB、LLM不使用）"""
    from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher
    from src.core.search.keyword_search_engine import KeywordSearchEngine

    config = st.session_state.config
    VECTOR_DB_BASE = PROJECT_ROOT / "data" / "vector_db"

    keyword_engine = KeywordSearchEngine(
        stop_words=config.STOP_WORDS,
        position_weight=config.POSITION_WEIGHT,
    )

    searcher = ChromaDBKeywordSearcher(
        base_db_path=str(VECTOR_DB_BASE),
        keyword_engine=keyword_engine,
        area_to_bot=AREA_TO_BOT,
        area_to_category=AREA_TO_CATEGORY,
    )

    collection_names = [f"rev{revision}_{area}" for area in areas]
    matches = searcher.search(collection_names, query, provider="azure_openai")

    # UI版フォーマットに変換
    return [
        {
            "Similarity": m.similarity,
            "Search_Result_Q": m.question,
            "Search_Result_A": m.answer,
            "Search_Category": "Keyword",
            "Sheet_Name": AREA_TO_CATEGORY.get(m.collection_name.split("_", 1)[-1] if "_" in m.collection_name else m.collection_name, ""),
            "Row_Index": m.row_index,
            "Search_Query": "",
            "_area": m.collection_name.split("_", 1)[-1] if m.collection_name.startswith("rev") else m.collection_name,
        }
        for m in matches
    ]
```

2. `SCENARIO_DIR` 定数（180行付近）を削除
3. 不要になった `import pandas as pd` の Excel 読み込み用途を確認（他で使用中なら残す）

**Step 2: テスト実行**

```bash
cd rag-local && python -m pytest tests/unit/ -v
```
Expected: PASS

**Step 3: コミット**

```bash
git add apps/revision-eval/ui/eval_ui.py
git commit -m "refactor: UI版keyword_filterをChromaDB化"
```

---

## Task 5: 影響調査モード追加（UI版）

**Files:**
- Modify: `rag-local/apps/revision-eval/ui/eval_ui.py` — 検索タイプ選択UI + 影響調査実行ロジック

**Step 1: 検索タイプ選択UIを拡張**

`eval_ui.py` の検索タイプ選択部分（Streamlitウィジェット）に `"impact_analysis"` を追加。
影響調査モード選択時は:
- revision 選択を非表示
- カテゴリ選択（「全て」「内部事務」「スマイル」）を表示

```python
# 検索タイプ選択を見つけて拡張
SEARCH_TYPE_OPTIONS = {
    "multi_stage": "ハイブリッド検索",
    "keyword_filter": "キーワード検索（改定前データ）",
    "impact_analysis": "影響調査（通常業務データ）",
}
```

**Step 2: 影響調査検索ロジックを追加**

```python
def _execute_impact_analysis_search(query: str, categories: List[str]) -> List[Dict]:
    """影響調査モード: 通常業務データの横断キーワード検索"""
    from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher
    from src.core.search.keyword_search_engine import KeywordSearchEngine

    config = st.session_state.config
    VECTOR_DB_BASE = PROJECT_ROOT / "data" / "vector_db"

    keyword_engine = KeywordSearchEngine(
        stop_words=config.STOP_WORDS,
        position_weight=config.POSITION_WEIGHT,
    )

    searcher = ChromaDBKeywordSearcher(
        base_db_path=str(VECTOR_DB_BASE),
        keyword_engine=keyword_engine,
        area_to_bot=AREA_TO_BOT,
        area_to_category=AREA_TO_CATEGORY,
    )

    # カテゴリ → コレクション名
    collection_names = categories  # ["naibujimu"], ["smile"], ["naibujimu", "smile"]
    matches = searcher.search(collection_names, query, provider="azure_openai")

    return [
        {
            "Similarity": m.similarity,
            "Search_Result_Q": m.question,
            "Search_Result_A": m.answer,
            "Search_Category": "Keyword",
            "Sheet_Name": AREA_TO_CATEGORY.get(m.collection_name, m.collection_name),
            "Row_Index": m.row_index,
            "Search_Query": "",
            "_area": m.collection_name,
            "_source": m.source,  # "scenario" or "history_data" でFAQ/シナリオ判別
        }
        for m in matches
    ]
```

**Step 3: `execute_dual_provider_search` に影響調査モード分岐を追加**

```python
# execute_dual_provider_search 内
if search_type == "impact_analysis":
    # 影響調査モードは別関数で処理
    categories = st.session_state.get("impact_categories", ["naibujimu", "smile"])
    impact_results = _execute_impact_analysis_search(query, categories)
    return impact_results, [], ""
```

**Step 4: 手動テスト**

```bash
cd rag-local && streamlit run apps/revision-eval/ui/eval_ui.py
```
- 影響調査モードを選択
- カテゴリ「全て」で検索 → naibujimu + smile の結果が表示されること
- FAQ と シナリオ の両方がヒットすること

**Step 5: コミット**

```bash
git add apps/revision-eval/ui/eval_ui.py
git commit -m "feat: 影響調査モード追加（通常業務データ横断キーワード検索）"
```

---

## Task 6: クリーンアップ・最終確認

**Files:**
- Modify: 不要なインポート・定数の削除
- Check: 全テスト実行

**Step 1: 不要コード削除**

- `evaluate_revisions.py`: `SCENARIO_DIR` 定数、`_load_scenario_excel()` が残っていれば削除
- `eval_ui.py`: `SCENARIO_DIR` 相当の定数が残っていれば削除
- 両ファイル: 不要な `import pandas as pd` 用途確認（他で使用中なら残す）

**Step 2: 全テスト実行**

```bash
cd rag-local && python -m pytest tests/unit/ -v
```
Expected: 全テスト PASS

**Step 3: コミット**

```bash
git add -A
git commit -m "chore: keyword_filter ChromaDB化のクリーンアップ"
```

---

## 検証チェックリスト

- [ ] `test_chromadb_keyword_search.py` 全テスト PASS
- [ ] 既存ユニットテスト 全 PASS
- [ ] UI版: keyword_filter で検索 → 結果表示（正解ID照合）
- [ ] UI版: 影響調査モードで検索 → FAQ+シナリオ横断結果
- [ ] バッチ版: `evaluate_revisions.py` 実行 → Excel出力のフォーマット互換
