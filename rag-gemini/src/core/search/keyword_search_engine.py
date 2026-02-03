# --- src/core/search/keyword_search_engine.py ---
"""キーワード検索エンジン

Sudachiトークナイザーを使用したキーワード抽出とJaccard類似度計算。
"""

import threading
from collections import Counter
from typing import List, Set, Dict, Tuple

from sudachipy import Dictionary, tokenizer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class KeywordSearchEngine:
    """キーワード検索エンジン

    Sudachiトークナイザーを使用してキーワードを抽出し、
    Jaccard類似度に基づくキーワードマッチングを行う。

    Attributes:
        stop_words: 除外する一般的な単語のセット
        position_weight: テキスト前半マッチ時の重み係数
    """

    # パフォーマンス: Sudachi辞書をクラス変数として共有（メモリ節約）
    _shared_tokenizer = None
    _tokenizer_lock = threading.Lock()

    @classmethod
    def _get_shared_tokenizer(cls):
        """スレッドセーフな共有トークナイザーを取得"""
        if cls._shared_tokenizer is None:
            with cls._tokenizer_lock:
                if cls._shared_tokenizer is None:  # Double-checked locking
                    cls._shared_tokenizer = Dictionary().create()
                    logger.info("Sudachi辞書を共有インスタンスとして初期化")
        return cls._shared_tokenizer

    def __init__(
        self,
        stop_words: Tuple[str, ...],
        position_weight: float = 1.2
    ):
        """KeywordSearchEngineを初期化

        Args:
            stop_words: 除外する一般的な単語のタプル
            position_weight: テキスト前半マッチ時の重み係数
        """
        self.tokenizer = self._get_shared_tokenizer()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.stop_words = set(stop_words)
        self.position_weight = position_weight

        # パフォーマンス: キーワードキャッシュ（N+1問題解消）
        self._keyword_cache: Dict[int, Set[str]] = {}

        logger.info("KeywordSearchEngineを初期化しました")

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """テキストからキーワードを抽出

        名詞（固有名詞、一般名詞）を抽出し、重要度順にソート。

        Args:
            text: 抽出対象のテキスト
            top_k: 返すキーワードの最大数

        Returns:
            List[str]: 重要度順のキーワードリスト
        """
        morphemes = self.tokenizer.tokenize(text, self.mode)
        keywords = []

        for m in morphemes:
            if m.part_of_speech()[0] == '名詞':
                important_types = ['固有名詞', '一般']
                weight = 2 if m.part_of_speech()[1] in important_types else 1
                word = m.dictionary_form()
                if len(word) > 1:
                    keywords.extend([word] * weight)

        filtered_words = {
            word: count
            for word, count in Counter(keywords).items()
            if word not in self.stop_words
        }
        return [word for word, _ in Counter(filtered_words).most_common(top_k)]

    def calculate_similarity(
        self,
        query_keywords: List[str],
        reference_text: str
    ) -> float:
        """キーワード類似度を計算（Jaccard-like正規化）

        Args:
            query_keywords: クエリから抽出されたキーワードリスト
            reference_text: 参照テキスト

        Returns:
            float: 0.0〜1.0の類似度スコア
        """
        ref_keywords = set(self.extract_keywords(reference_text))
        query_keywords_set = set(query_keywords)

        if not ref_keywords or not query_keywords_set:
            return 0.0

        intersection = ref_keywords.intersection(query_keywords_set)
        union = ref_keywords.union(query_keywords_set)

        if not union:
            return 0.0

        # 交差したキーワードにのみ位置の重みを適用
        weighted_score = sum(
            self.position_weight if reference_text.find(kw) < len(reference_text) // 2 else 1.0
            for kw in intersection
        )
        # 分母は素のunionサイズ（Jaccard-like正規化）
        normalized_score = weighted_score / len(union)
        return min(normalized_score, 1.0)

    def calculate_similarity_fast(
        self,
        query_keywords_set: Set[str],
        ref_keywords: Set[str]
    ) -> float:
        """キャッシュされたキーワードセットを使用した高速類似度計算

        Args:
            query_keywords_set: クエリキーワードのセット
            ref_keywords: 参照キーワードのセット（キャッシュから取得）

        Returns:
            float: 0.0〜1.0の類似度スコア
        """
        if not ref_keywords or not query_keywords_set:
            return 0.0

        intersection = ref_keywords.intersection(query_keywords_set)
        union = ref_keywords.union(query_keywords_set)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def build_cache(self, queries: List[str]) -> None:
        """参照クエリのキーワードキャッシュを構築

        Args:
            queries: キャッシュ対象のクエリリスト
        """
        logger.info("キーワードキャッシュを構築中...")
        self._keyword_cache = {}

        for i, query in enumerate(queries):
            self._keyword_cache[i] = set(self.extract_keywords(query))

        logger.info(f"キーワードキャッシュ構築完了: {len(self._keyword_cache)}件")

    def get_cached_keywords(self, index: int) -> Set[str]:
        """キャッシュからキーワードセットを取得

        Args:
            index: キャッシュのインデックス

        Returns:
            Set[str]: キーワードセット

        Raises:
            KeyError: キャッシュにインデックスが存在しない場合
        """
        if index not in self._keyword_cache:
            raise KeyError(f"キーワードキャッシュにインデックス {index} が存在しません")
        return self._keyword_cache[index]

    def has_cached_keywords(self, index: int) -> bool:
        """キャッシュにキーワードが存在するか確認

        Args:
            index: 確認するインデックス

        Returns:
            bool: 存在する場合True
        """
        return index in self._keyword_cache

    def clear_cache(self) -> None:
        """キーワードキャッシュをクリア"""
        self._keyword_cache.clear()
        logger.info("キーワードキャッシュをクリアしました")

    def filter_by_keywords(
        self,
        query_keywords: List[str],
        texts: List[str]
    ) -> List[Tuple[int, int]]:
        """キーワードを含むテキストをフィルタリング

        入力キーワードを1つでも含むテキストのインデックスとマッチ数を返す。
        結果はマッチ数の降順、同数の場合は元の順序でソートされる。

        Args:
            query_keywords: 検索キーワードのリスト
            texts: フィルタリング対象のテキストリスト

        Returns:
            List[Tuple[int, int]]: (インデックス, マッチ数)のリスト（マッチ数降順）
        """
        if not query_keywords:
            return []

        results = []
        for idx, text in enumerate(texts):
            # 各キーワードがテキストに含まれるかチェック（部分一致）
            match_count = sum(1 for kw in query_keywords if kw in text)
            if match_count > 0:
                results.append((idx, match_count))

        # マッチ数の降順でソート、同数の場合は元の順序（idx昇順）
        results.sort(key=lambda x: (-x[1], x[0]))
        return results

    def filter_by_keywords_from_cache(
        self,
        query_keywords: Set[str]
    ) -> List[Tuple[int, int]]:
        """キャッシュされたキーワードセットを使用してフィルタリング

        Args:
            query_keywords: 検索キーワードのセット

        Returns:
            List[Tuple[int, int]]: (インデックス, マッチ数)のリスト（マッチ数降順）
        """
        if not query_keywords:
            return []

        results = []
        for idx, ref_keywords in self._keyword_cache.items():
            # キーワードの共通部分をカウント
            match_count = len(query_keywords.intersection(ref_keywords))
            if match_count > 0:
                results.append((idx, match_count))

        # マッチ数の降順でソート、同数の場合は元の順序（idx昇順）
        results.sort(key=lambda x: (-x[1], x[0]))
        return results
