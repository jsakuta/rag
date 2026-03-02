# --- src/core/search/keyword_search_engine.py ---
"""キーワード検索エンジン

Sudachiトークナイザーを使用したキーワード抽出とJaccard類似度計算。
"""

import hashlib
import json
import threading
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set, Dict, Tuple

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
                    logger.debug("Sudachi辞書を共有インスタンスとして初期化")
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

        logger.debug("KeywordSearchEngineを初期化しました")

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """テキストからキーワードを抽出

        名詞（固有名詞、一般名詞）を抽出し、重要度順にソート。

        Args:
            text: 抽出対象のテキスト
            top_k: 返すキーワードの最大数

        Returns:
            List[str]: 重要度順のキーワードリスト
        """
        with self._tokenizer_lock:
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
        half_len = len(reference_text) // 2
        weighted_score = sum(
            self.position_weight if 0 <= reference_text.find(kw) < half_len else 1.0
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

    def build_cache(
        self,
        queries: List[str],
        cache_path: Optional[Path] = None,
        index_mapping: Optional[List[int]] = None,
    ) -> None:
        """参照クエリのキーワードキャッシュを構築（ディスクキャッシュ対応）

        cache_path が指定され、ファイルが存在し、件数が一致する場合はディスクから読み込む。
        なければ構築してディスクに保存する。

        Args:
            queries: キャッシュ対象のクエリリスト
            cache_path: キャッシュファイルパス（省略時はディスクキャッシュなし）
            index_mapping: キャッシュキーとして使うインデックスのリスト。
                指定時は queries[i] を index_mapping[i] のキーで格納する。
                省略時は 0, 1, 2, ... の連番。
        """
        # index_mapping 使用時はハッシュに含めて旧キャッシュを無効化
        hash_queries = queries
        if index_mapping:
            hash_queries = [f"{index_mapping[i]}:{q}" for i, q in enumerate(queries)]
        content_hash = self._compute_content_hash(hash_queries)
        if cache_path and self._try_load_cache(cache_path, len(queries), content_hash):
            return

        logger.debug("キーワードキャッシュを構築中...")
        self._keyword_cache = {}

        active_count = 0
        for i, query in enumerate(queries):
            key = index_mapping[i] if index_mapping else i
            if query:
                self._keyword_cache[key] = set(self.extract_keywords(query))
                active_count += 1
            else:
                self._keyword_cache[key] = set()

        logger.debug(f"キーワードキャッシュ構築完了: {active_count}/{len(queries)}件")

        if cache_path:
            self._save_cache(cache_path, len(queries), content_hash)

    @staticmethod
    def _compute_content_hash(queries: List[str]) -> str:
        """クエリリストのコンテンツハッシュを計算（キャッシュ無効化用）"""
        hasher = hashlib.md5()
        for q in queries:
            hasher.update((q or "").encode("utf-8"))
        return hasher.hexdigest()

    def _try_load_cache(self, cache_path: Path, expected_count: int, content_hash: str) -> bool:
        """ディスクキャッシュを読み込み。件数またはハッシュ不一致なら False を返す。"""
        try:
            if not cache_path.exists():
                return False
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if raw.get("count") != expected_count:
                logger.debug(f"キャッシュ件数不一致 ({raw.get('count')} != {expected_count}), 再構築")
                return False
            if raw.get("hash") != content_hash:
                logger.debug("キャッシュコンテンツハッシュ不一致, 再構築")
                return False
            self._keyword_cache = {int(k): set(v) for k, v in raw["data"].items()}
            logger.debug(f"キーワードキャッシュをディスクから読み込み: {len(self._keyword_cache)}件")
            return True
        except Exception as e:
            logger.warning(f"キャッシュ読み込み失敗: {e}")
            return False

    def _save_cache(self, cache_path: Path, count: int, content_hash: str) -> None:
        """キーワードキャッシュをディスクに保存"""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            raw = {
                "count": count,
                "hash": content_hash,
                "data": {str(k): sorted(v) for k, v in self._keyword_cache.items()},
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            logger.debug(f"キーワードキャッシュをディスクに保存: {cache_path}")
        except Exception as e:
            logger.warning(f"キャッシュ保存失敗: {e}")

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

