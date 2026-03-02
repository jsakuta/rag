# --- src/core/search/vector_search_engine.py ---
"""ベクトル検索エンジン

埋め込みモデルを使用したセマンティック検索。
"""

from typing import List, Dict, Any, Optional

from src.utils.base_embedding import BaseEmbeddingModel
from src.utils.vector_db import MetadataVectorDB
from src.utils.logger import setup_logger
from src.types.search_types import VectorSearchResultDict

logger = setup_logger(__name__)


class VectorSearchEngine:
    """ベクトル検索エンジン

    埋め込みモデルとVectorDBを使用してセマンティック検索を行う。

    Attributes:
        embedding_model: 埋め込みモデル
        vector_db: ベクトルデータベース
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_db: Optional[MetadataVectorDB] = None
    ):
        """VectorSearchEngineを初期化

        Args:
            embedding_model: 埋め込みモデル
            vector_db: ベクトルデータベース（後から設定可能）
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db

        logger.debug("VectorSearchEngineを初期化しました")

    def encode_query(self, query: str) -> List[float]:
        """クエリをベクトルに変換

        Args:
            query: 検索クエリ

        Returns:
            List[float]: クエリベクトル
        """
        embeddings = self.embedding_model.encode([query], normalize_embeddings=True)
        return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])

    def search(
        self,
        query: str,
        n_results: int,
        filter_metadata: Optional[Dict[str, str]] = None
    ) -> List[VectorSearchResultDict]:
        """ベクトル検索を実行

        Args:
            query: 検索クエリ
            n_results: 返す結果の数
            filter_metadata: メタデータフィルタ（オプション）

        Returns:
            List[VectorSearchResultDict]: 検索結果のリスト

        Raises:
            RuntimeError: VectorDBが設定されていない場合
        """
        if self.vector_db is None:
            raise RuntimeError("VectorDBが設定されていません。set_vector_db()を呼び出してください。")

        query_vector = self.encode_query(query)

        if filter_metadata:
            logger.debug(f"  Search source filter: {filter_metadata}")

        search_results = self.vector_db.search(
            query_embedding=query_vector,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        logger.debug(f"  Vector search returned {len(search_results)} results")

        # 検索結果のソース分布をログ出力
        source_counts: Dict[str, int] = {}
        for result in search_results:
            source = result['metadata'].get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        logger.debug(f"  Search results by source: {source_counts}")

        return search_results

    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得

        Returns:
            Dict[str, Any]: コレクション情報

        Raises:
            RuntimeError: VectorDBが設定されていない場合
        """
        if self.vector_db is None:
            raise RuntimeError("VectorDBが設定されていません。")

        return self.vector_db.get_collection_info()
