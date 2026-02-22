"""_calculate_keyword_similarities のDB/参照データ不整合テスト

バグ再現: Vector DBのドキュメントIDが参照データの範囲外のとき、
ValueErrorでクラッシュする問題を修正するためのTDDテスト。
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_searcher_with_cache(reference_count: int, cache_indices: dict = None):
    """テスト用のモックSearcherを構築

    Args:
        reference_count: reference_queriesの件数
        cache_indices: _reference_keywords_cache の内容（{index: set(keywords)}）
    """
    searcher = MagicMock()
    searcher.reference_queries = [f"質問{i}" for i in range(reference_count)]
    searcher._reference_keywords_cache = cache_indices or {
        i: {f"kw{i}"} for i in range(reference_count)
    }
    # _extract_keywords のモック（ドキュメントテキストからキーワード抽出用）
    searcher._extract_keywords = MagicMock(return_value=["個人事業主", "カード"])
    # _calculate_keyword_similarity のモック
    searcher._calculate_keyword_similarity = MagicMock(return_value=0.5)
    return searcher


class TestCalculateKeywordSimilaritiesOutOfSync:
    """Vector DBとreference_dataが不整合のときの振る舞いテスト"""

    def test_doc_id_exceeds_reference_count_should_not_crash(self):
        """doc_idが参照データ範囲外でも、クラッシュせず類似度を返す"""
        from src.core.searcher import Searcher

        searcher = _make_searcher_with_cache(reference_count=100)

        # Vector DB検索結果: doc_200 は参照データ(100件)の範囲外
        search_results = [
            {
                'id': 'doc_200',
                'document': '分類: カード認証 | 質問: 個人事業主のカード認証 | 回答: 認めます',
                'metadata': {'source': 'history_data'},
                'similarity': 0.85,
            }
        ]
        keywords = ["個人事業主", "カード", "認証"]

        # 修正前: ValueErrorが発生する
        # 修正後: クラッシュせずに類似度のリストを返す
        result = Searcher._calculate_keyword_similarities(searcher, search_results, keywords)

        assert isinstance(result, list)
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0

    def test_multiple_results_mixed_in_and_out_of_range(self):
        """範囲内と範囲外のドキュメントが混在する場合、全て正しく処理"""
        from src.core.searcher import Searcher

        searcher = _make_searcher_with_cache(
            reference_count=100,
            cache_indices={
                10: {"口座", "開設"},
                50: {"残高", "照会"},
            }
        )

        search_results = [
            # 範囲内・キャッシュヒット
            {
                'id': 'doc_10',
                'document': '分類: 口座 | 質問: 口座開設方法 | 回答: 窓口へ',
                'metadata': {},
                'similarity': 0.90,
            },
            # 範囲外（doc_500, 参照データ100件）
            {
                'id': 'doc_500',
                'document': '分類: カード | 質問: カード認証 | 回答: 可能',
                'metadata': {},
                'similarity': 0.80,
            },
            # 範囲内・キャッシュヒット
            {
                'id': 'doc_50',
                'document': '分類: 残高 | 質問: 残高照会 | 回答: ATMで',
                'metadata': {},
                'similarity': 0.75,
            },
        ]
        keywords = ["口座", "開設"]

        result = Searcher._calculate_keyword_similarities(searcher, search_results, keywords)

        assert len(result) == 3
        assert all(0.0 <= sim <= 1.0 for sim in result)

    def test_out_of_range_uses_document_text_for_keyword_matching(self):
        """範囲外ドキュメントは、検索結果のdocumentテキストからキーワード抽出して類似度を計算"""
        from src.core.searcher import Searcher

        searcher = _make_searcher_with_cache(reference_count=10)
        # ドキュメントテキストからの抽出を模擬
        searcher._extract_keywords = MagicMock(return_value=["個人事業主", "カード"])

        search_results = [
            {
                'id': 'doc_999',
                'document': '分類: 認証 | 質問: 個人事業主のカード認証 | 回答: 認可',
                'metadata': {},
                'similarity': 0.85,
            }
        ]
        keywords = ["個人事業主", "カード", "認証"]

        result = Searcher._calculate_keyword_similarities(searcher, search_results, keywords)

        # _extract_keywords がドキュメントテキストで呼ばれたことを確認
        searcher._extract_keywords.assert_called()
        assert len(result) == 1
        assert result[0] > 0.0  # キーワードが一致するので0より大きい

    def test_cache_miss_within_range_falls_back_to_reference_query(self):
        """キャッシュミスだが参照データ範囲内の場合、reference_queriesを使用"""
        from src.core.searcher import Searcher

        # キャッシュにインデックス5がない
        searcher = _make_searcher_with_cache(
            reference_count=10,
            cache_indices={0: {"口座"}, 1: {"残高"}}
        )

        search_results = [
            {
                'id': 'doc_5',
                'document': '分類: 振込 | 質問: 振込方法 | 回答: ATM',
                'metadata': {},
                'similarity': 0.80,
            }
        ]
        keywords = ["振込"]

        result = Searcher._calculate_keyword_similarities(searcher, search_results, keywords)

        # reference_queries[5]を使って_calculate_keyword_similarityが呼ばれる
        searcher._calculate_keyword_similarity.assert_called_once_with(["振込"], "質問5")
        assert len(result) == 1
