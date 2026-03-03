# --- utils/gemini_embedding.py ---
"""Gemini (Vertex AI) 埋め込みモデル"""
import threading
import numpy as np
from typing import List, Union, Optional, TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from src.utils.base_embedding import BaseEmbeddingModel
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from config import SearchConfig

logger = setup_logger(__name__)

# google-genai SDK のインポート
try:
    from google.genai.errors import ClientError, ServerError
    from google.genai.types import EmbedContentConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    # Retry Logic Fallback: SDKがない場合でもリトライが機能するよう、
    # 独自の例外クラスを定義（Exceptionへのフォールバックを避ける）
    class ClientError(Exception):
        """google-genai SDK not available - placeholder"""
        def __init__(self, *args, code=None, **kwargs):
            self.code = code
            super().__init__(*args, **kwargs)
    class ServerError(Exception):
        """google-genai SDK not available - placeholder"""
        pass
    EmbedContentConfig = None
    logger.warning("google-genai SDK not installed. Run: pip install google-genai")


def _is_retryable(error):
    """リトライ対象の例外かどうかを判定"""
    if isinstance(error, ServerError):
        return True       # 5xx は全てリトライ
    if isinstance(error, ClientError) and getattr(error, 'code', 0) == 429:
        return True       # Rate limit のみリトライ
    return False


class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini Embedding API を使用する埋め込みモデルクラス

    Vertex AI の gemini-embedding-001 モデルを使用してテキストを
    ベクトル化します。デフォルト3072次元の埋め込みを生成します。
    （768, 1536, 3072 から選択可能）
    """

    # gemini-embedding-001 の次元数（デフォルト）
    EMBEDDING_DIM = 3072

    # google-genai SDK のバッチ上限
    GENAI_SDK_BATCH_LIMIT = 100

    _instance: Optional['GeminiEmbeddingModel'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: 'SearchConfig') -> 'GeminiEmbeddingModel':
        """スレッドセーフなシングルトンインスタンスを取得（Double-checked locking）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    # スレッドセーフ: 一時変数を使用して完全に初期化してから代入（Race Condition防止）
                    instance = cls(config)
                    # 初期化時の設定をprimitive valuesとして保存（比較の確実性のため）
                    instance._init_model_name = config.embedding_model
                    instance._init_project_id = config.gemini_project_id
                    # 完全に初期化が完了してからクラス変数に代入
                    cls._instance = instance
                    logger.debug("GeminiEmbeddingModel singleton instance created")
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
                "google-genai SDK is not installed. "
                "Run: pip install google-genai"
            )

        try:
            from src.utils.auth import create_genai_client
            self._model_name = self.config.embedding_model or "gemini-embedding-001"
            client = create_genai_client(self.config)
            logger.debug(f"Gemini Embedding API initialized (model: {self._model_name})")
            return client  # → base class が self.model に格納

        except Exception as e:
            logger.error(f"Failed to initialize Gemini Embedding API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception(_is_retryable),
        reraise=True
    )
    def _get_embeddings_with_retry(self, batch_texts: List[str]):
        """リトライ付きで Embedding API を呼び出す

        リトライ対象:
        - ServerError (5xx): サービス一時不可
        - ClientError (429): レート制限/クォータ超過

        リトライ対象外（即座に失敗）:
        - ClientError (403): 権限エラー
        - ClientError (400): 無効なリクエスト

        Args:
            batch_texts: テキストのバッチ

        Returns:
            埋め込み結果のリスト
        """
        try:
            response = self.model.models.embed_content(
                model=self._model_name,
                contents=batch_texts,
                config=EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIM),
            )
            return response.embeddings
        except ClientError as e:
            if getattr(e, 'code', 0) != 429:
                logger.error(f"回復不可能なAPIエラー (HTTP {getattr(e, 'code', '?')}): {e}")
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
            batch_size = min(self.config.EMBEDDING_BATCH_SIZE, self.GENAI_SDK_BATCH_LIMIT)
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

            logger.debug(f"Generated embeddings for {len(texts)} texts using Gemini")
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
