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
    db_path = str(tmp_path / "test_db")

    # rev02_souzoku/azure_openai 相当のコレクション
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
