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
