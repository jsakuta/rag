"""SearchStrategyパターンのユニットテスト"""
import pytest
from unittest.mock import MagicMock, patch

from src.core.search.search_strategy import (
    create_strategy,
    OriginalSearchStrategy,
    LLMEnhancedSearchStrategy,
    MultiStageSearchStrategy,
    KeywordFilterSearchStrategy,
)


class TestCreateStrategy:
    """create_strategy ファクトリ関数のテスト"""

    def _make_mock_searcher(self, search_type="hybrid", search_mode="original"):
        """モックSearcherを生成"""
        searcher = MagicMock()
        searcher.config = MagicMock()
        searcher.config.search_type = search_type
        searcher.config.search_mode = search_mode
        return searcher

    def test_original_strategy(self):
        """search_mode=original → OriginalSearchStrategy"""
        searcher = self._make_mock_searcher(search_mode="original")
        strategy = create_strategy(searcher)
        assert isinstance(strategy, OriginalSearchStrategy)

    def test_llm_enhanced_strategy(self):
        """search_mode=llm_enhanced → LLMEnhancedSearchStrategy"""
        searcher = self._make_mock_searcher(search_mode="llm_enhanced")
        strategy = create_strategy(searcher)
        assert isinstance(strategy, LLMEnhancedSearchStrategy)

    def test_multi_stage_strategy(self):
        """search_mode=multi_stage → MultiStageSearchStrategy"""
        searcher = self._make_mock_searcher(search_mode="multi_stage")
        strategy = create_strategy(searcher)
        assert isinstance(strategy, MultiStageSearchStrategy)

    def test_keyword_filter_strategy(self):
        """search_type=keyword_filter → KeywordFilterSearchStrategy"""
        searcher = self._make_mock_searcher(search_type="keyword_filter")
        strategy = create_strategy(searcher)
        assert isinstance(strategy, KeywordFilterSearchStrategy)

    def test_keyword_filter_overrides_search_mode(self):
        """search_type=keyword_filter は search_mode より優先"""
        searcher = self._make_mock_searcher(
            search_type="keyword_filter", search_mode="multi_stage"
        )
        strategy = create_strategy(searcher)
        assert isinstance(strategy, KeywordFilterSearchStrategy)


class TestOriginalSearchStrategy:
    """OriginalSearchStrategyのテスト"""

    def test_execute_calls_shared_methods(self):
        """共有メソッドが正しく呼ばれることを検証"""
        searcher = MagicMock()
        searcher.config = MagicMock()
        searcher.config.search_mode = "original"
        searcher.config.search_type = "hybrid"
        searcher._keyword_engine.extract_keywords.return_value = ["口座", "開設"]
        searcher._execute_vector_search.return_value = []
        searcher._calculate_and_merge_scores.return_value = []
        searcher._format_final_results.return_value = [{"Similarity": 0.9}]

        strategy = OriginalSearchStrategy(searcher)
        result = strategy.execute("1", "口座開設の方法", "窓口で申込")

        searcher._keyword_engine.extract_keywords.assert_called_once_with("口座開設の方法")
        searcher._execute_vector_search.assert_called_once_with("口座開設の方法")
        searcher._calculate_and_merge_scores.assert_called_once()
        searcher._format_final_results.assert_called_once()
        assert len(result) == 1


class TestLLMEnhancedSearchStrategy:
    """LLMEnhancedSearchStrategyのテスト"""

    def test_execute_uses_summarize_text(self):
        """LLMで拡張されたクエリが使用されることを検証"""
        searcher = MagicMock()
        searcher.config = MagicMock()
        searcher.config.search_mode = "llm_enhanced"
        searcher.config.search_type = "hybrid"
        searcher.summarize_text.return_value = "口座開設手続き"
        searcher._keyword_engine.extract_keywords.return_value = ["口座", "開設"]
        searcher._execute_vector_search.return_value = []
        searcher._calculate_and_merge_scores.return_value = []
        searcher._format_final_results.return_value = []

        strategy = LLMEnhancedSearchStrategy(searcher)
        strategy.execute("1", "口座開設の方法を教えてください", "")

        searcher.summarize_text.assert_called_once_with("口座開設の方法を教えてください")
        searcher._execute_vector_search.assert_called_once_with("口座開設手続き")


class TestKeywordFilterSearchStrategy:
    """KeywordFilterSearchStrategyのテスト"""

    def test_execute_no_keywords(self):
        """キーワード抽出失敗時は空リストを返す"""
        searcher = MagicMock()
        searcher.config = MagicMock()
        searcher.config.search_type = "keyword_filter"
        searcher._keyword_engine.extract_keywords.return_value = []

        strategy = KeywordFilterSearchStrategy(searcher)
        result = strategy.execute("1", "", "")

        assert result == []

    def test_execute_no_matches(self):
        """マッチなし時は空リストを返す"""
        searcher = MagicMock()
        searcher.config = MagicMock()
        searcher.config.search_type = "keyword_filter"
        searcher._keyword_engine.extract_keywords.return_value = ["振込"]
        searcher._reference_keywords_cache = {
            0: {"口座", "開設"},
            1: {"残高", "照会"},
        }

        strategy = KeywordFilterSearchStrategy(searcher)
        result = strategy.execute("1", "振込方法", "")

        assert result == []
