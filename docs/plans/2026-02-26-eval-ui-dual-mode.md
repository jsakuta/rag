# eval_ui.py 2モード再構成 + レビュー指摘修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** eval_ui.py を「評価モード」「影響調査モード」の2モード構成に再設計し、レビュー指摘7件を修正する。

**Architecture:** ChromaDBKeywordSearcher に source_filter / extract_area public化 / 例外処理修正を施し、eval_ui.py のサイドバーをモード切替型に再構成する。影響調査モードでは通常業務コレクション(naibujimu, smile)に対して hybrid/keyword 両方の検索を提供し、metadata.source でシナリオ/FAQフィルタを行う。

**Tech Stack:** Python, Streamlit, ChromaDB, pytest

**設計書:** `docs/plans/2026-02-26-eval-ui-dual-mode-design.md`

---

### Task 1: ChromaDBKeywordSearcher のバグ修正 + 機能追加

BUG-2(例外握りつぶし), BUG-6(extract_area public化), source_filter 追加, キャッシュ追加を一括で行う。

**Files:**
- Modify: `rag-local/src/core/search/chromadb_keyword_search.py`
- Modify: `rag-local/src/core/search/__init__.py`

**Step 1: chromadb_keyword_search.py を修正**

```python
"""ChromaDB全件取得 + キーワードマッチング

評価モード（rev*コレクション）と影響調査モード（naibujimu, smile）の
両方で使用する共通キーワード検索モジュール。

ChromaDB の collection.get() で全件取得し、Python側でキーワードマッチングを行う。
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

    @property
    def area(self) -> str:
        """コレクション名からarea名を返す"""
        return ChromaDBKeywordSearcher.extract_area(self.collection_name)


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
        self._collection_cache: Dict[Tuple[str, str], Tuple[list, list]] = {}

    def search(
        self,
        collection_names: List[str],
        query: str,
        provider: str = "azure_openai",
        max_results: int = 50,
        source_filter: Optional[str] = None,
    ) -> List[MatchResult]:
        """ChromaDBからキーワード検索を実行

        Args:
            collection_names: 検索対象コレクション名リスト
                評価モード: ["rev02_souzoku"] 等
                影響調査モード: ["naibujimu", "smile"] 等
            query: 検索クエリ
            provider: 埋め込みプロバイダー（DBディレクトリ構造用）
            max_results: 最大結果件数
            source_filter: データソースフィルタ（"scenario" | "history_data" | None=全て）

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
            results = self._search_collection(col_name, keywords, provider, source_filter)
            all_results.extend(results)

        all_results.sort(key=lambda r: r.match_count, reverse=True)
        return all_results[:max_results]

    def _get_collection_data(
        self, collection_name: str, provider: str
    ) -> Tuple[list, list]:
        """コレクションデータを取得（キャッシュ付き）"""
        cache_key = (collection_name, provider)
        if cache_key in self._collection_cache:
            return self._collection_cache[cache_key]

        db_path = os.path.join(self.base_db_path, collection_name, provider)
        if not os.path.exists(db_path):
            logger.warning(f"DBパスが存在しません: {db_path}")
            return [], []

        try:
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection("default")
        except ChromaNotFoundError:
            logger.warning(f"コレクションが見つかりません: {collection_name}/{provider}")
            return [], []
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"DB読み込みエラー ({collection_name}/{provider}): {e}")
            return [], []

        result = collection.get(include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        self._collection_cache[cache_key] = (documents, metadatas)
        return documents, metadatas

    def _search_collection(
        self,
        collection_name: str,
        keywords: List[str],
        provider: str,
        source_filter: Optional[str] = None,
    ) -> List[MatchResult]:
        """単一コレクションからキーワード検索"""
        documents, metadatas = self._get_collection_data(collection_name, provider)

        if not documents:
            logger.info(f"{collection_name}: ドキュメントなし")
            return []

        area = self.extract_area(collection_name)
        bot_name = self._resolve_bot_name(area)
        total_keywords = len(keywords)

        matched: List[MatchResult] = []
        for doc, meta in zip(documents, metadatas):
            if source_filter and meta.get("source") != source_filter:
                continue

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

    @staticmethod
    def extract_area(collection_name: str) -> str:
        """コレクション名からarea名を抽出（rev02_souzoku → souzoku）"""
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

**Step 2: __init__.py を更新（_extract_area → extract_area）**

`rag-local/src/core/search/__init__.py` — `__all__` リストは変更不要（`ChromaDBKeywordSearcher` は既に含まれている）。

**Step 3: テスト実行**

Run: `cd rag-local && python -m pytest tests/unit/test_chromadb_keyword_search.py -v`
Expected: 既存6テスト全パス（public extract_area は後方互換）

**Step 4: コミット**

```bash
git add rag-local/src/core/search/chromadb_keyword_search.py rag-local/src/core/search/__init__.py
git commit -m "fix: ChromaDBKeywordSearcher 例外処理修正 + source_filter + キャッシュ + extract_area public化"
```

---

### Task 2: テスト追加・修正（BUG-7 + source_filter テスト）

**Files:**
- Modify: `rag-local/tests/unit/test_chromadb_keyword_search.py`

**Step 1: テストファイルを書き換え**

fixture に source が異なるデータ（scenario + history_data）を追加し、source_filter テストと BUG-7 修正テストを追加。

```python
"""ChromaDBKeywordSearcher のユニットテスト"""
import os
import pytest
import chromadb
from chromadb.config import Settings
from unittest.mock import MagicMock
from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher, MatchResult


@pytest.fixture
def in_memory_db(tmp_path):
    """テスト用ChromaDBをtmp_pathに作成し、サンプルデータを投入"""
    # rev02_souzoku/azure_openai コレクション
    sub_db_path = str(tmp_path / "test_db" / "rev02_souzoku" / "azure_openai")
    os.makedirs(sub_db_path, exist_ok=True)
    sub_client = chromadb.PersistentClient(
        path=sub_db_path,
        settings=Settings(anonymized_telemetry=False),
    )
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

    # naibujimu/azure_openai コレクション（シナリオ + FAQ 混在）
    mixed_db_path = str(tmp_path / "test_db" / "naibujimu" / "azure_openai")
    os.makedirs(mixed_db_path, exist_ok=True)
    mixed_client = chromadb.PersistentClient(
        path=mixed_db_path,
        settings=Settings(anonymized_telemetry=False),
    )
    mixed_col = mixed_client.get_or_create_collection("default")
    mixed_col.add(
        documents=[
            "分類: 預金 > 届出印 | 質問: 届出印の届出方法 | 回答: 窓口で届出書を提出してください",
            "分類: 預金 > 届出印 | 質問: 届出印を変更する手続き | 回答: 届出印変更届を提出してください",
            "分類: 預金 > 解約 | 質問: 通帳の解約方法 | 回答: 窓口で解約届を提出してください",
        ],
        metadatas=[
            {"source": "scenario", "sheet_name": "預金", "row_index": 100, "hierarchy": "預金 > 届出印"},
            {"source": "history_data", "sheet_name": "預金", "row_index": 200, "hierarchy": "預金 > 届出印"},
            {"source": "scenario", "sheet_name": "預金", "row_index": 300, "hierarchy": "預金 > 解約"},
        ],
        ids=["mixed_1", "mixed_2", "mixed_3"],
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
        assert results[0].match_count == 2
        assert results[0].row_index == 10
        assert results[0].scenario_id == "souzoku-bot_12"

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

    def test_source_field_values(self, in_memory_db, mock_keyword_engine):
        """source フィールドが有効な値であること"""
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=mock_keyword_engine,
            area_to_bot={"souzoku": "souzoku-bot"},
            area_to_category={"souzoku": "相続"},
        )
        results = searcher.search(["rev02_souzoku"], "届出印", provider="azure_openai")
        assert len(results) >= 1
        valid_sources = {"scenario", "history_data", "unknown"}
        for r in results:
            assert r.source in valid_sources

    def test_source_filter_scenario_only(self, in_memory_db):
        """source_filter="scenario" でシナリオのみ返却"""
        engine = MagicMock()
        engine.extract_keywords.return_value = ["届出印"]
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=engine,
            area_to_bot={"naibujimu": "naibujimu-bot"},
            area_to_category={"naibujimu": "内部事務"},
        )
        results = searcher.search(
            ["naibujimu"], "届出印", provider="azure_openai",
            source_filter="scenario",
        )
        assert len(results) >= 1
        for r in results:
            assert r.source == "scenario"

    def test_source_filter_faq_only(self, in_memory_db):
        """source_filter="history_data" でFAQのみ返却"""
        engine = MagicMock()
        engine.extract_keywords.return_value = ["届出印"]
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=engine,
            area_to_bot={"naibujimu": "naibujimu-bot"},
            area_to_category={"naibujimu": "内部事務"},
        )
        results = searcher.search(
            ["naibujimu"], "届出印", provider="azure_openai",
            source_filter="history_data",
        )
        assert len(results) >= 1
        for r in results:
            assert r.source == "history_data"

    def test_source_filter_none_returns_all(self, in_memory_db):
        """source_filter=None で全件返却"""
        engine = MagicMock()
        engine.extract_keywords.return_value = ["届出印"]
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=engine,
            area_to_bot={"naibujimu": "naibujimu-bot"},
            area_to_category={"naibujimu": "内部事務"},
        )
        results = searcher.search(
            ["naibujimu"], "届出印", provider="azure_openai",
            source_filter=None,
        )
        sources = {r.source for r in results}
        assert "scenario" in sources
        assert "history_data" in sources

    def test_extract_area_public(self):
        """extract_area が public メソッドとして呼べること"""
        assert ChromaDBKeywordSearcher.extract_area("rev02_souzoku") == "souzoku"
        assert ChromaDBKeywordSearcher.extract_area("naibujimu") == "naibujimu"
        assert ChromaDBKeywordSearcher.extract_area("rev03_naibujimu") == "naibujimu"

    def test_match_result_area_property(self, in_memory_db, mock_keyword_engine):
        """MatchResult.area プロパティが正しく動作すること"""
        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(in_memory_db),
            keyword_engine=mock_keyword_engine,
            area_to_bot={"souzoku": "souzoku-bot"},
            area_to_category={"souzoku": "相続"},
        )
        results = searcher.search(["rev02_souzoku"], "届出印", provider="azure_openai")
        assert len(results) >= 1
        assert results[0].area == "souzoku"
```

**Step 2: テスト実行**

Run: `cd rag-local && python -m pytest tests/unit/test_chromadb_keyword_search.py -v`
Expected: 12テスト全パス

**Step 3: コミット**

```bash
git add rag-local/tests/unit/test_chromadb_keyword_search.py
git commit -m "test: source_filter + extract_area + MatchResult.area テスト追加"
```

---

### Task 3: evaluate_revisions.py バグ修正（BUG-1, BUG-5）

**Files:**
- Modify: `rag-local/apps/revision-eval/evaluate_revisions.py:265-323`

**Step 1: BUG-1（2重キーワード抽出）と BUG-5（O(N×M)）を修正**

`_execute_keyword_filter_search` メソッドを以下に書き換え:

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
        # キーワード抽出は searcher.search() 内で1回だけ行われる

        searcher = ChromaDBKeywordSearcher(
            base_db_path=str(VECTOR_DB_BASE),
            keyword_engine=keyword_engine,
            area_to_bot=AREA_TO_BOT,
            area_to_category=AREA_TO_CATEGORY,
        )

        # area ごとに MAX_RESULTS 件制限するため、全体上限は十分大きく
        all_matches = searcher.search(areas, query, provider="azure_openai", max_results=MAX_RESULTS * len(areas))

        # BUG-5修正: defaultdict で O(N+M) グルーピング
        from collections import defaultdict
        grouped: Dict[str, List] = defaultdict(list)
        for m in all_matches:
            grouped[m.area].append(m)

        results_by_area = {}
        searched_areas = []
        keywords = keyword_engine.extract_keywords(query)  # ログ用（searcher内で既に抽出済み）

        for area in areas:
            area_name = ChromaDBKeywordSearcher.extract_area(area)
            area_matches = grouped.get(area_name, [])
            if not area_matches:
                continue

            bot_name = self._extract_bot_name_from_area(area)
            area_results = []
            for i, m in enumerate(area_matches[:MAX_RESULTS]):
                lv1 = m.hierarchy.split(" > ")[0].strip() if m.hierarchy else ""
                source_file = self._get_source_file(revision, bot_name, lv1)
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
```

**注意**: `_extract_area` → `extract_area` の参照更新。`from collections import defaultdict` は関数内 import でOK（バッチ版でのみ使用される軽量 import）。

**Step 2: テスト実行**

Run: `cd rag-local && python -m pytest tests/unit/ -v --timeout=30`
Expected: 全テストパス

**Step 3: コミット**

```bash
git add rag-local/apps/revision-eval/evaluate_revisions.py
git commit -m "fix: バッチ版 キーワード2重抽出削除 + O(N×M)→O(N+M) グルーピング"
```

---

### Task 4: eval_ui.py 2モード再構成

最大の変更。サイドバーを「評価モード」「影響調査モード」に分離し、影響調査モードに hybrid/keyword 両対応を追加。

**Files:**
- Modify: `rag-local/apps/revision-eval/ui/eval_ui.py`

**Step 1: session_state 初期化を拡張（BUG-4 修正）**

`initialize_session_state()` を修正:

```python
def initialize_session_state():
    initialize_common_session_state()
    if "correct_ids" not in st.session_state:
        st.session_state.correct_ids = []
    if "selected_revision" not in st.session_state:
        st.session_state.selected_revision = None
    if "azure_results" not in st.session_state:
        st.session_state.azure_results = []
    if "vertex_results" not in st.session_state:
        st.session_state.vertex_results = []
    # 影響調査モード用
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "evaluation"
    if "impact_categories" not in st.session_state:
        st.session_state.impact_categories = ["naibujimu", "smile"]
    if "impact_source_filter" not in st.session_state:
        st.session_state.impact_source_filter = None  # None=全て
```

**Step 2: execute_dual_provider_search を影響調査 hybrid 対応に拡張**

```python
def execute_dual_provider_search(query: str, revision: str) -> Tuple[List[Dict], List[Dict], str]:
    """Azure/VertexAI両方で検索を実行"""
    config = st.session_state.config
    app_mode = st.session_state.get("app_mode", "evaluation")

    # 影響調査モード
    if app_mode == "impact_analysis":
        search_type = getattr(config, "search_type", "hybrid")
        categories = st.session_state.get("impact_categories", ["naibujimu", "smile"])
        source_filter = st.session_state.get("impact_source_filter")

        if search_type == "keyword_filter":
            results = _execute_impact_keyword_search(query, categories, source_filter)
            return results, [], ""
        else:
            # hybrid: 意味検索（Azure/VertexAI）
            vector_weight = getattr(config, "vector_weight", DEFAULT_VECTOR_WEIGHT)
            azure_results = _search_with_provider(query, "", "azure_openai", categories, vector_weight)
            vertex_results = _search_with_provider(query, "", "vertex_ai", categories, vector_weight)
            # source_filter を Python 側で適用
            if source_filter:
                azure_results = [r for r in azure_results if r.get("_source", "") == source_filter or "_source" not in r]
                vertex_results = [r for r in vertex_results if r.get("_source", "") == source_filter or "_source" not in r]
            llm_query = azure_results[0].get("Search_Query", query) if azure_results else ""
            return azure_results, vertex_results, llm_query

    # 評価モード
    search_type = getattr(config, "search_type", None) or REVISION_SEARCH_TYPES.get(revision, "hybrid")
    areas = REVISION_TO_AREAS.get(revision, [])
    vector_weight = REVISION_VECTOR_WEIGHTS.get(revision, DEFAULT_VECTOR_WEIGHT)

    if not areas:
        logger.warning(f"改定 {revision} に対応するエリアがありません")
        return [], [], ""

    if search_type == "keyword_filter":
        azure_results = _execute_keyword_filter_search(query, revision, areas)
        return azure_results, [], ""

    azure_results = _search_with_provider(query, revision, "azure_openai", areas, vector_weight)
    vertex_results = _search_with_provider(query, revision, "vertex_ai", areas, vector_weight)

    llm_query = azure_results[0].get("Search_Query", query) if azure_results else ""

    return azure_results, vertex_results, llm_query
```

**Step 3: 影響調査キーワード検索関数のリネーム・修正**

既存の `_execute_impact_analysis_search` → `_execute_impact_keyword_search` にリネームし、`source_filter` パラメータを追加:

```python
def _execute_impact_keyword_search(query: str, categories: List[str], source_filter: Optional[str] = None) -> List[Dict]:
    """影響調査モード: キーワード検索"""
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

    matches = searcher.search(
        categories, query, provider="azure_openai",
        max_results=10000, source_filter=source_filter,
    )

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
            "_source": m.source,
        }
        for m in matches
    ]
```

**Step 4: _execute_keyword_filter_search で extract_area を public 参照に更新**

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

    matches = searcher.search(areas, query, provider="azure_openai", max_results=10000)

    return [
        {
            "Similarity": m.similarity,
            "Search_Result_Q": m.question,
            "Search_Result_A": m.answer,
            "Search_Category": "Keyword",
            "Sheet_Name": AREA_TO_CATEGORY.get(m.area, m.collection_name),
            "Row_Index": m.row_index,
            "Search_Query": "",
            "_area": m.area,
        }
        for m in matches
    ]
```

**Step 5: process_query を影響調査モード対応に修正**

```python
def process_query(query: str):
    st.session_state.processing_query = True
    try:
        query_number = len(st.session_state.chat_history) // 2 + 1
        app_mode = st.session_state.get("app_mode", "evaluation")
        revision = st.session_state.selected_revision

        if app_mode == "impact_analysis":
            logger.info(f"=== 影響調査クエリ {query_number} ===")
        else:
            search_type = REVISION_SEARCH_TYPES.get(revision, "hybrid")
            logger.info(f"=== 評価クエリ {query_number}: 改定番号={revision}, 検索タイプ={search_type} ===")

        azure_results, vertex_results, llm_query = execute_dual_provider_search(query, revision)

        st.session_state.azure_results = azure_results
        st.session_state.vertex_results = vertex_results

        logger.info(f"Azure検索結果数: {len(azure_results)}件, VertexAI検索結果数: {len(vertex_results)}件")

        st.session_state.chat_history.append({
            "type": "bot",
            "text": {
                "mode": "dual_provider",
                "azure": azure_results,
                "vertex": vertex_results,
                "llm_query": llm_query,
            }
        })

    except Exception as e:
        escaped_error = html.escape(str(e))
        error_message = f"エラーが発生しました: {escaped_error}"
        st.error(error_message)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({"type": "bot", "text": error_message})
    finally:
        st.session_state.processing_query = False
```

**Step 6: _render_provider_results の search_type 判定を修正**

```python
def _render_provider_results(results: List[Dict], correct_ids: List[str], is_vertex: bool = False) -> None:
    """プロバイダー検索結果を表示"""
    if not results:
        if is_vertex:
            app_mode = st.session_state.get("app_mode", "evaluation")
            search_type = getattr(st.session_state.config, "search_type", "hybrid")
            if app_mode == "impact_analysis" and search_type == "keyword_filter":
                st.info("キーワード検索のためスキップ（Azureタブの結果をご確認ください）")
            elif app_mode == "evaluation" and search_type == "keyword_filter":
                st.info("キーワード検索のためスキップ（Azureタブの結果をご確認ください）")
            else:
                st.info("該当する結果がありません")
        else:
            st.info("該当する結果がありません")
        return
    # ... (以降変更なし)
```

**Step 7: run_streamlit_ui のサイドバーを2モード構成に再設計**

`run_streamlit_ui()` 内の `with st.sidebar:` ブロックを以下に書き換え:

```python
def run_streamlit_ui():
    st.set_page_config(page_title="事務改定評価", layout="wide", initial_sidebar_state="expanded")
    apply_common_styles()
    initialize_session_state()

    with st.sidebar:
        st.title("事務改定 AI")

        # モード選択（最上部）
        app_mode = st.radio(
            "モード",
            options=["evaluation", "impact_analysis"],
            format_func=lambda x: {"evaluation": "評価モード", "impact_analysis": "影響調査モード"}[x],
            key="app_mode_radio",
            horizontal=True,
        )
        st.session_state.app_mode = app_mode

        st.markdown("---")

        if app_mode == "evaluation":
            # === 評価モード ===
            revision_data = load_revision_correct_ids()
            revision_options = list(revision_data.keys())

            if not revision_options:
                st.warning("正解IDデータが見つかりません")
                st.session_state.selected_revision = None
                st.session_state.correct_ids = []
            else:
                current_revision_idx = 0
                if st.session_state.selected_revision in revision_options:
                    current_revision_idx = revision_options.index(st.session_state.selected_revision)

                selected_revision = st.selectbox(
                    "改定番号",
                    revision_options,
                    index=current_revision_idx,
                    key="revision_select",
                    help="改定番号を選択すると、Azure/VertexAI両方で検索し、正解IDとマッチした結果にバッジを表示します"
                )

                st.session_state.selected_revision = selected_revision

                if selected_revision in revision_data:
                    content, correct_ids = revision_data[selected_revision]
                    st.session_state.correct_ids = correct_ids
                    st.success(f"正解ID: {len(correct_ids)}件")

                    areas = REVISION_TO_AREAS.get(selected_revision, [])
                    if areas:
                        st.caption(f"対象エリア: {', '.join(areas)}")

            st.markdown("---")
            st.subheader("検索設定")

            default_search_type = REVISION_SEARCH_TYPES.get(
                st.session_state.get("selected_revision", ""), "hybrid"
            )
            eval_search_type_labels = {
                "hybrid": "意味検索",
                "keyword_filter": "キーワード検索",
            }
            eval_selected_search_type = st.radio(
                "検索タイプ",
                options=["hybrid", "keyword_filter"],
                format_func=lambda x: eval_search_type_labels[x],
                index=0 if default_search_type == "hybrid" else 1,
                key="eval_search_type_radio",
                horizontal=True,
            )
            st.session_state.config.search_type = eval_selected_search_type

            if eval_selected_search_type == "hybrid":
                default_vector_weight = REVISION_VECTOR_WEIGHTS.get(
                    st.session_state.get("selected_revision", ""), DEFAULT_VECTOR_WEIGHT
                )
                weight = render_vector_weight_slider(default_vector_weight, key="eval_vector_weight")
                st.session_state.config.vector_weight = weight
                st.session_state.config.keyword_weight = 1.0 - weight

                st.markdown("---")
                eval_top_k = st.number_input(
                    "候補数",
                    min_value=10,
                    max_value=200,
                    value=max(10, st.session_state.config.top_k),
                    step=10,
                    key="eval_top_k",
                    help="検索結果の最大件数（評価用に多めに設定）"
                )
                st.session_state.config.top_k = eval_top_k
            else:
                st.caption("キーワード検索: マッチする全件を返却します")

        else:
            # === 影響調査モード ===
            st.session_state.correct_ids = []  # 正解判定なし

            impact_category_options = {
                "all": "全て（内部事務 + スマイル）",
                "naibujimu": "内部事務",
                "smile": "スマイル",
            }
            impact_category = st.radio(
                "対象カテゴリ",
                options=list(impact_category_options.keys()),
                format_func=lambda x: impact_category_options[x],
                key="impact_category_radio",
            )
            if impact_category == "all":
                st.session_state.impact_categories = ["naibujimu", "smile"]
            else:
                st.session_state.impact_categories = [impact_category]

            st.markdown("---")
            source_options = {
                "all": "全て（シナリオ + FAQ）",
                "scenario": "シナリオのみ",
                "history_data": "FAQのみ",
            }
            source_selection = st.radio(
                "データソース",
                options=list(source_options.keys()),
                format_func=lambda x: source_options[x],
                key="impact_source_radio",
            )
            st.session_state.impact_source_filter = None if source_selection == "all" else source_selection

            st.markdown("---")
            st.subheader("検索設定")

            impact_search_type_labels = {
                "hybrid": "意味検索",
                "keyword_filter": "キーワード検索",
            }
            impact_search_type = st.radio(
                "検索タイプ",
                options=["hybrid", "keyword_filter"],
                format_func=lambda x: impact_search_type_labels[x],
                key="impact_search_type_radio",
                horizontal=True,
            )
            st.session_state.config.search_type = impact_search_type

            if impact_search_type == "hybrid":
                weight = render_vector_weight_slider(DEFAULT_VECTOR_WEIGHT, key="impact_vector_weight")
                st.session_state.config.vector_weight = weight
                st.session_state.config.keyword_weight = 1.0 - weight

                impact_top_k = st.number_input(
                    "候補数",
                    min_value=10,
                    max_value=200,
                    value=max(10, st.session_state.config.top_k),
                    step=10,
                    key="impact_top_k",
                )
                st.session_state.config.top_k = impact_top_k
            else:
                st.caption("キーワード検索: マッチする全件を返却します")

        st.markdown("---")
        if st.button("チャット履歴を保存", use_container_width=True, key="save_chat_history_button"):
            save_chat_history()

    # メインエリア タイトル
    if st.session_state.get("app_mode") == "impact_analysis":
        cats = st.session_state.get("impact_categories", [])
        cat_label = " + ".join(AREA_TO_CATEGORY.get(c, c) for c in cats)
        st.title(f"影響調査【{cat_label}】")
    elif st.session_state.selected_revision:
        st.title(f"事務改定評価【改定{st.session_state.selected_revision}】")
    else:
        st.title("事務改定評価")

    # チャット表示・検索フォームは既存のまま
    chat_container = st.container()
    # ... (以降は既存コードを維持、ただし検索ボタン押下時の改定番号チェックを修正)
```

**Step 8: 検索ボタンの改定番号チェックを修正**

```python
    if submit_button and query.strip():
        app_mode = st.session_state.get("app_mode", "evaluation")
        if app_mode == "evaluation" and not st.session_state.selected_revision:
            st.warning("改定番号を選択してください。")
        else:
            st.session_state.chat_history.append({"type": "user", "text": query})
            process_query(query.strip())
```

**Step 9: 手動テスト**

Run: `cd rag-local && streamlit run apps/revision-eval/ui/eval_ui.py`

確認項目:
1. サイドバー最上部にモード切替（評価/影響調査）が表示される
2. 評価モード: 改定番号選択 → hybrid/keyword_filter → 検索 → Azure/VertexAI タブ + 正解バッジ
3. 影響調査モード: カテゴリ + データソース + 検索タイプ → 検索 → 結果表示
4. 影響調査モード keyword_filter: シナリオのみ / FAQのみ でフィルタ確認
5. 影響調査モード hybrid: 意味検索結果が表示される

**Step 10: コミット**

```bash
git add rag-local/apps/revision-eval/ui/eval_ui.py
git commit -m "feat: eval_ui 2モード構成（評価モード + 影響調査モード）"
```

---

### Task 5: 最終統合テスト + 設計書コミット

**Files:**
- Existing: `docs/plans/2026-02-26-eval-ui-dual-mode-design.md`
- Existing: `docs/plans/2026-02-26-eval-ui-dual-mode.md`

**Step 1: 全ユニットテスト実行**

Run: `cd rag-local && python -m pytest tests/unit/ -v --timeout=30`
Expected: 全テストパス（既存 + 新規12テスト）

**Step 2: 設計書・計画書をコミット**

```bash
git add docs/plans/2026-02-26-eval-ui-dual-mode-design.md docs/plans/2026-02-26-eval-ui-dual-mode.md
git commit -m "docs: eval_ui 2モード再構成 設計書 + 実装計画"
```

---

## タスク一覧

| Task | 内容 | 修正するBUG |
|------|------|------------|
| 1 | ChromaDBKeywordSearcher バグ修正 + 機能追加 | BUG-2, BUG-3, BUG-6 |
| 2 | テスト追加・修正 | BUG-7 + source_filter |
| 3 | evaluate_revisions.py バグ修正 | BUG-1, BUG-5 |
| 4 | eval_ui.py 2モード再構成 | BUG-4 + 新機能 |
| 5 | 統合テスト + ドキュメントコミット | — |
