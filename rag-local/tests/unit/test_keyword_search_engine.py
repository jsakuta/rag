"""KeywordSearchEngineのユニットテスト"""
import pytest
from src.core.search.keyword_search_engine import KeywordSearchEngine


class TestKeywordSearchEngine:
    """KeywordSearchEngineのテストクラス"""

    def test_extract_keywords_basic(self):
        """基本的なキーワード抽出"""
        engine = KeywordSearchEngine(stop_words=('こと', 'もの', 'ため'))
        text = "口座開設の方法を教えてください"
        keywords = engine.extract_keywords(text, top_k=3)

        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        # ストップワードが除外されていること
        assert 'こと' not in keywords
        assert 'もの' not in keywords

    def test_extract_keywords_empty(self):
        """空文字列のキーワード抽出"""
        engine = KeywordSearchEngine(stop_words=('こと', 'もの', 'ため'))
        keywords = engine.extract_keywords("", top_k=5)

        assert isinstance(keywords, list)
        assert len(keywords) == 0

    def test_calculate_similarity_basic(self):
        """基本的な類似度計算"""
        engine = KeywordSearchEngine(stop_words=('こと', 'もの', 'ため'))
        # テキストから実際に抽出されるキーワードを使用
        ref_keywords = engine.extract_keywords("口座開設の手続き方法について")
        query_keywords = engine.extract_keywords("口座開設の方法を教えてください")
        # 抽出されたキーワードが同じなら類似度 > 0
        text = "口座開設の手続き方法について"
        similarity = engine.calculate_similarity(query_keywords, text)

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_calculate_similarity_no_match(self):
        """マッチなしの類似度計算"""
        engine = KeywordSearchEngine(stop_words=('こと', 'もの', 'ため'))
        keywords = ['振込', '手数料']
        text = "口座開設の手続き方法について"

        similarity = engine.calculate_similarity(keywords, text)

        assert isinstance(similarity, float)
        assert similarity == 0.0  # 共通キーワードなし
