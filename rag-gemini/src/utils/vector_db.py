import os
import threading
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError as ChromaNotFoundError
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import numpy as np
from datetime import datetime
import uuid
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LRUCache:
    """スレッドセーフなLRUキャッシュ（Memory Leak防止）"""

    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                # アクセスされたら最後尾に移動（LRU）
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    # 最も古いエントリを削除
                    oldest_key, oldest_value = self._cache.popitem(last=False)
                    logger.info(f"LRUCache: Evicting oldest entry: {oldest_key}")
                    # ChromaDBクライアントの場合はクローズを試みる
                    if hasattr(oldest_value, '_server') and oldest_value._server:
                        try:
                            oldest_value._server = None
                        except Exception as e:
                            logger.warning(f"Failed to cleanup ChromaDB client: {e}")
                self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache


class MetadataVectorDB:
    """メタデータ対応のベクトルデータベースクラス"""

    # パフォーマンス + Memory Leak防止: LRUキャッシュを使用（最大10エントリ）
    _client_cache = LRUCache(max_size=10)
    _cache_lock = threading.Lock()

    def __init__(self, base_dir: str = ".", collection_name: str = None, batch_size: int = 100, db_path: str = None):
        self.collection_name = collection_name
        self.batch_size = batch_size  # 設定可能なバッチサイズ

        # db_pathが直接指定されている場合はそれを使用、そうでない場合は従来通りbase_dirから生成
        if db_path is not None:
            self.db_path = db_path
            self.base_dir = None  # db_path直接指定時はbase_dirは使用しない
        else:
            self.base_dir = base_dir
            self.db_path = os.path.join(base_dir, "data", "vector_db")

        os.makedirs(self.db_path, exist_ok=True)

        # パフォーマンス + Memory Leak防止: LRUキャッシュを使用
        with self._cache_lock:
            cached_client = self._client_cache.get(self.db_path)
            if cached_client is None:
                cached_client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                self._client_cache.put(self.db_path, cached_client)
                logger.info(f"New ChromaDB client created for {self.db_path}")
            self.client = cached_client
        
        # コレクションの取得または作成
        if self.collection_name is None:
            raise ValueError("collection_name must be specified")
            
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Existing collection '{self.collection_name}' loaded")
        except (ValueError, ChromaNotFoundError) as e:
            # コレクションが存在しない場合（ChromaDBのバージョンにより例外型が異なる）
            logger.debug(f"Collection not found, creating new one: {e}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": f"RAG system vector database for {self.collection_name}",
                    "hnsw:space": "cosine"  # コサイン距離を明示指定
                }
            )
            logger.info(f"New collection '{self.collection_name}' created")
        except Exception as e:
            logger.error(f"Unexpected error accessing collection '{self.collection_name}': {e}")
            raise
    
    def add_documents(self,
                     texts: List[str],
                     embeddings,
                     metadatas: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> None:
        """ドキュメントとメタデータをベクトルDBに追加

        Args:
            texts: ドキュメントテキストのリスト
            embeddings: 埋め込みベクトル（List[List[float]] または numpy.ndarray）
            metadatas: メタデータのリスト
            ids: ドキュメントIDのリスト（省略時は自動生成）
        """
        # 安全性: 空リストチェック（Empty Batch Handling）
        if not texts:
            logger.warning("add_documents called with empty texts list, skipping")
            return

        # 安全性: ユニークID生成（Duplicate ID Collision防止）
        if ids is None:
            # UUIDを使用してコレクション間でも一意なIDを生成
            batch_uuid = uuid.uuid4().hex[:8]
            ids = [f"doc_{batch_uuid}_{i}" for i in range(len(texts))]

        # パフォーマンス: numpy配列の場合はtolist()を呼び出す
        # （ChromaDBはlistを期待するため変換が必要）
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()

        # メタデータの正規化（型保持を改善）
        normalized_metadatas = []
        for metadata in metadatas:
            normalized_metadata = {}
            for key, value in metadata.items():
                # 日付の場合は文字列に変換
                if isinstance(value, datetime):
                    normalized_metadata[key] = value.strftime("%Y/%m/%d")
                # リストの場合は文字列に結合（要素を文字列に変換）
                elif isinstance(value, list):
                    normalized_metadata[key] = " | ".join(str(v) for v in value if v is not None) if value else ""
                # ブーリアン: boolはintのサブクラスなので先にチェック
                # ChromaDBはboolをサポートしないためint変換
                elif isinstance(value, bool):
                    normalized_metadata[key] = 1 if value else 0
                # 数値: ChromaDBがサポートする型はそのまま保持
                elif isinstance(value, (int, float)):
                    normalized_metadata[key] = value
                elif value is None:
                    normalized_metadata[key] = ""
                else:
                    normalized_metadata[key] = str(value)
            normalized_metadatas.append(normalized_metadata)

        # バッチサイズで分割して追加（ChromaDBの制限を回避）
        # パフォーマンス: 設定可能なバッチサイズを使用
        for i in range(0, len(texts), self.batch_size):
            end_idx = min(i + self.batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = normalized_metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]

            # 安全性: 空バッチをスキップ
            if not batch_texts:
                continue

            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

        logger.info(f"Added {len(texts)} documents to vector database")
    
    # セキュリティ: 許可されたメタデータキーと期待される型のホワイトリスト
    ALLOWED_METADATA_KEYS = {
        'source': (str, int, float),
        'tags': (str, int, float),
        'date': (str,),
        'hierarchy': (str,),
        'sheet_name': (str,),
        'row_index': (int, str),
        'scenario': (str,),
        'faq': (str,),
        'category': (str,),
        'type': (str,),
        'business_area': (str,)
    }

    def search(self,
               query_embedding: List[float],
               n_results: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """メタデータフィルタリング付きベクトル検索"""

        # フィルタリング条件の正規化（セキュリティ強化: ホワイトリスト検証）
        where_filter = None
        if filter_metadata:
            where_filter = {}
            for key, value in filter_metadata.items():
                # セキュリティ: 許可されたキーのみ受け入れ
                if key not in self.ALLOWED_METADATA_KEYS:
                    logger.warning(f"Invalid metadata key rejected: {key}")
                    continue

                expected_types = self.ALLOWED_METADATA_KEYS[key]

                # セキュリティ: dict型は演算子インジェクション防止のため拒否
                if isinstance(value, dict):
                    # $in 演算子のみ許可（内部で生成する場合のみ）
                    if len(value) == 1 and "$in" in value and isinstance(value["$in"], list):
                        sanitized_list = [v for v in value["$in"] if isinstance(v, expected_types)]
                        if sanitized_list:
                            where_filter[key] = {"$in": sanitized_list}
                    else:
                        logger.warning(f"Complex filter operators not allowed: {key}={value}")
                    continue

                # セキュリティ: リストは $in 演算子に変換
                if isinstance(value, list):
                    sanitized_list = [v for v in value if isinstance(v, expected_types)]
                    if sanitized_list:
                        where_filter[key] = {"$in": sanitized_list}
                elif isinstance(value, expected_types):
                    where_filter[key] = value
                else:
                    logger.warning(f"Type mismatch for {key}: expected {expected_types}, got {type(value)}")
        
        # 検索実行
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # 結果の整形
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                # コサイン距離の場合: distance = 1 - cos_similarity
                # 類似度を0〜1の範囲にクリップ
                'similarity': max(0.0, min(1.0, 1.0 - results['distances'][0][i]))
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクションの情報を取得"""
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'document_count': count,
            'path': self.db_path
        }
    
    def delete_collection(self) -> None:
        """コレクションを削除"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted")
    
    def reset_collection(self) -> None:
        """コレクションをリセット（削除して再作成）"""
        try:
            self.delete_collection()
        except ValueError:
            # コレクションが存在しない場合は無視
            logger.debug(f"Collection '{self.collection_name}' did not exist, creating new one")
        except Exception as e:
            logger.warning(f"Error deleting collection '{self.collection_name}': {e}")

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "RAG system vector database with metadata"}
        )
        logger.info(f"Collection '{self.collection_name}' reset") 