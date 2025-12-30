import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MetadataVectorDB:
    """メタデータ対応のベクトルデータベースクラス"""
    
    def __init__(self, base_dir: str = ".", collection_name: str = None):
        self.base_dir = base_dir
        self.collection_name = collection_name
        self.db_path = os.path.join(base_dir, "reference", "vector_db")
        os.makedirs(self.db_path, exist_ok=True)
        
        # ChromaDBクライアントの初期化
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # コレクションの取得または作成
        if self.collection_name is None:
            raise ValueError("collection_name must be specified")
            
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Existing collection '{self.collection_name}' loaded")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"RAG system vector database for {self.collection_name}"}
            )
            logger.info(f"New collection '{self.collection_name}' created")
    
    def add_documents(self, 
                     texts: List[str], 
                     embeddings: List[List[float]], 
                     metadatas: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> None:
        """ドキュメントとメタデータをベクトルDBに追加"""
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        # メタデータの正規化
        normalized_metadatas = []
        for metadata in metadatas:
            normalized_metadata = {}
            for key, value in metadata.items():
                # 日付の場合は文字列に変換
                if isinstance(value, datetime):
                    normalized_metadata[key] = value.strftime("%Y/%m/%d")
                # リストの場合は文字列に結合
                elif isinstance(value, list):
                    normalized_metadata[key] = " | ".join(value) if value else ""
                else:
                    normalized_metadata[key] = str(value) if value is not None else ""
            normalized_metadatas.append(normalized_metadata)
        
        # バッチサイズで分割して追加（ChromaDBの制限を回避）
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = normalized_metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        
        logger.info(f"Added {len(texts)} documents to vector database")
    
    def search(self, 
               query_embedding: List[float], 
               n_results: int = 10,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """メタデータフィルタリング付きベクトル検索"""
        
        # フィルタリング条件の正規化
        where_filter = None
        if filter_metadata:
            where_filter = {}
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    # リストの場合はOR条件
                    where_filter[key] = {"$in": value}
                else:
                    where_filter[key] = value
        
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
                'similarity': 1.0 - results['distances'][0][i]  # 距離を類似度に変換
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
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "RAG system vector database with metadata"}
        )
        logger.info(f"Collection '{self.collection_name}' reset") 