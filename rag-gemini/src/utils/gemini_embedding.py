# --- utils/gemini_embedding.py ---
"""Gemini (Vertex AI) 埋め込みモデル"""
import threading
import numpy as np
from typing import List, Union, Optional, TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.base_embedding import BaseEmbeddingModel
from src.utils.logger import setup_logger
from src.utils.auth import initialize_vertex_ai

if TYPE_CHECKING:
    from config import SearchConfig

logger = setup_logger(__name__)

# Vertex AI SDK のインポート
try:
    from vertexai.language_models import TextEmbeddingModel
    from google.api_core.exceptions import (
        ServiceUnavailable,
        TooManyRequests,
        DeadlineExceeded,
        PermissionDenied,
        InvalidArgument
    )
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    # フォールバック用
    ServiceUnavailable = Exception
    TooManyRequests = Exception
    DeadlineExceeded = Exception
    PermissionDenied = Exception
    InvalidArgument = Exception
    logger.warning("Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")


class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini Embedding API を使用する埋め込みモデルクラス

    Vertex AI の gemini-embedding-001 モデルを使用してテキストを
    ベクトル化します。デフォルト3072次元の埋め込みを生成します。
    （768, 1536, 3072 から選択可能）
    """

    # gemini-embedding-001 の次元数（デフォルト）
    EMBEDDING_DIM = 3072

    _instance: Optional['GeminiEmbeddingModel'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: 'SearchConfig') -> 'GeminiEmbeddingModel':
        """スレッドセーフなシングルトンインスタンスを取得（Double-checked locking）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = cls(config)
                    # 初期化時の設定をprimitive valuesとして保存（比較の確実性のため）
                    cls._instance._init_model_name = config.embedding_model
                    cls._instance._init_project_id = config.gemini_project_id
                    logger.info("GeminiEmbeddingModel singleton instance created")
        else:
            # 異なる設定で呼び出された場合は警告（primitive values で比較）
            if hasattr(cls._instance, '_init_model_name'):
                if (cls._instance._init_model_name != config.embedding_model or
                    cls._instance._init_project_id != config.gemini_project_id):
                    logger.warning(
                        "GeminiEmbeddingModel singleton called with different config. "
                        f"Using existing instance (model: {cls._instance._init_model_name})"
                    )
        return cls._instance

    def _setup_model(self):
        """Gemini Embedding API の初期化"""
        if not VERTEX_AI_AVAILABLE:
            raise ImportError(
                "Vertex AI SDK is not installed. "
                "Run: pip install google-cloud-aiplatform"
            )

        try:
            initialize_vertex_ai(self.config)

            # gemini-embedding-001 モデルの初期化
            model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

            logger.info("Gemini Embedding API initialized successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to initialize Gemini Embedding API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((ServiceUnavailable, TooManyRequests, DeadlineExceeded)),
        reraise=True
    )
    def _get_embeddings_with_retry(self, batch_texts: List[str]):
        """リトライ付きで Embedding API を呼び出す

        リトライ対象:
        - ServiceUnavailable (503): サービス一時不可
        - TooManyRequests (429): レート制限/クォータ超過
        - DeadlineExceeded: タイムアウト

        リトライ対象外（即座に失敗）:
        - PermissionDenied (403): 権限エラー
        - InvalidArgument (400): 無効なリクエスト

        Args:
            batch_texts: テキストのバッチ

        Returns:
            埋め込み結果のリスト
        """
        try:
            return self.model.get_embeddings(batch_texts)
        except (PermissionDenied, InvalidArgument) as e:
            # 回復不可能なエラーは即座に失敗させる
            logger.error(f"回復不可能なAPIエラー: {type(e).__name__}")
            raise

    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        """テキストをベクトル化

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
                for embedding in embeddings:
                    vector = np.array(embedding.values)
                    if normalize_embeddings:
                        vector = self.normalize_vector(vector)
                    all_embeddings.append(vector)

            # numpy配列に変換
            result = np.array(all_embeddings)

            logger.info(f"Generated embeddings for {len(texts)} texts using Gemini")
            return result

        except Exception as e:
            logger.error(f"Error generating embeddings with Gemini: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""
        return self.EMBEDDING_DIM

    @property
    def provider_name(self) -> str:
        """プロバイダー名を返す"""
        return "vertex_ai"
