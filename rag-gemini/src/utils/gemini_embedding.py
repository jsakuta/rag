import numpy as np
from typing import List, Union, Optional
from vertexai.language_models import TextEmbeddingModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.utils.logger import setup_logger
from src.utils.auth import initialize_vertex_ai

logger = setup_logger(__name__)

class GeminiEmbeddingModel:
    """Gemini Embedding APIを使用する埋め込みモデルクラス（シングルトン対応）"""

    _instance: Optional['GeminiEmbeddingModel'] = None

    @classmethod
    def get_instance(cls, config) -> 'GeminiEmbeddingModel':
        """シングルトンインスタンスを取得（パフォーマンス向上）"""
        if cls._instance is None:
            cls._instance = cls(config)
            logger.info("GeminiEmbeddingModel singleton instance created")
        return cls._instance

    def __init__(self, config):
        self.config = config
        self.model = self._setup_model()
    
    def _setup_model(self):
        """Gemini Embedding APIの初期化"""
        try:
            initialize_vertex_ai(self.config)

            # gemini-embedding-001モデルの初期化
            model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
            
            logger.info("Gemini Embedding API initialized successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Embedding API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _get_embeddings_with_retry(self, batch_texts: List[str]):
        """リトライ付きでEmbedding APIを呼び出す

        Args:
            batch_texts: テキストのバッチ

        Returns:
            埋め込み結果のリスト
        """
        return self.model.get_embeddings(batch_texts)

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        """
        テキストをベクトル化
        
        Args:
            texts: 単一テキストまたはテキストのリスト
            normalize_embeddings: ベクトルを正規化するかどうか
        
        Returns:
            numpy.ndarray: 埋め込みベクトル
        """
        try:
            # 単一テキストの場合はリストに変換
            if isinstance(texts, str):
                texts = [texts]
            
            # バッチサイズで分割（API制限を回避）
            batch_size = self.config.EMBEDDING_BATCH_SIZE
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # 埋め込み生成（リトライ付き）
                embeddings = self._get_embeddings_with_retry(batch_texts)

                # ベクトルを抽出
                batch_vectors = []
                for embedding in embeddings:
                    vector = embedding.values
                    if normalize_embeddings:
                        # L2正規化
                        norm = np.linalg.norm(vector)
                        if norm > 0:
                            vector = vector / norm
                    batch_vectors.append(vector)
                
                all_embeddings.extend(batch_vectors)
            
            # numpy配列に変換
            result = np.array(all_embeddings)
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_single(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        単一テキストをベクトル化
        
        Args:
            text: ベクトル化するテキスト
            normalize_embeddings: ベクトルを正規化するかどうか
        
        Returns:
            numpy.ndarray: 埋め込みベクトル
        """
        return self.encode([text], normalize_embeddings)[0] 