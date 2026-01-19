# --- utils/azure_embedding.py ---
"""Azure OpenAI text-embedding-3-large 埋め込みモデル"""
import os
import threading
import numpy as np
from typing import List, Union, Optional, TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.base_embedding import BaseEmbeddingModel
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from config import SearchConfig

logger = setup_logger(__name__)

# Azure OpenAI SDK のインポート（遅延インポートで依存関係を軽減）
try:
    from openai import AzureOpenAI, RateLimitError, APIConnectionError, APIStatusError
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    RateLimitError = Exception  # フォールバック用
    APIConnectionError = Exception
    APIStatusError = Exception
    logger.warning("Azure OpenAI SDK not installed. Run: pip install openai")


class AzureOpenAIEmbeddingModel(BaseEmbeddingModel):
    """Azure OpenAI text-embedding-3-large を使用する埋め込みモデルクラス

    Azure OpenAI の text-embedding-3-large モデルを使用してテキストを
    ベクトル化します。3072次元の高品質な埋め込みを生成します。
    """

    # text-embedding-3-large の次元数
    EMBEDDING_DIM = 3072

    _instance: Optional['AzureOpenAIEmbeddingModel'] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: 'SearchConfig') -> 'AzureOpenAIEmbeddingModel':
        """スレッドセーフなシングルトンインスタンスを取得（Double-checked locking）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = cls(config)
                    # 初期化時の設定をprimitive valuesとして保存（比較の確実性のため）
                    cls._instance._init_deployment = config.azure_openai_embedding_deployment
                    cls._instance._init_endpoint = config.azure_openai_embedding_endpoint
                    logger.info("AzureOpenAIEmbeddingModel singleton instance created")
        else:
            # 異なる設定で呼び出された場合は警告（primitive values で比較）
            if hasattr(cls._instance, '_init_deployment'):
                if (cls._instance._init_deployment != config.azure_openai_embedding_deployment or
                    cls._instance._init_endpoint != config.azure_openai_embedding_endpoint):
                    logger.warning(
                        "AzureOpenAIEmbeddingModel singleton called with different config. "
                        f"Using existing instance (deployment: {cls._instance._init_deployment})"
                    )
        return cls._instance

    def _setup_model(self):
        """Azure OpenAI クライアントの初期化"""
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError(
                "Azure OpenAI SDK is not installed. "
                "Run: pip install openai azure-identity"
            )

        try:
            # 環境変数から設定を取得
            endpoint = self.config.azure_openai_embedding_endpoint
            api_key = self.config.azure_openai_embedding_api_key
            api_version = self.config.azure_openai_embedding_api_version

            if not endpoint:
                raise ValueError("AZURE_OPENAI_EMBEDDING_ENDPOINT is not set")
            if not api_key:
                raise ValueError("AZURE_OPENAI_EMBEDDING_API_KEY is not set")

            # Azure OpenAI クライアントの初期化
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key
            )

            # セキュリティ: エンドポイントURLをマスクしてログ出力（URL解析版）
            from urllib.parse import urlparse
            try:
                parsed = urlparse(endpoint)
                if parsed.netloc:
                    parts = parsed.netloc.split('.')
                    # 最後のTLDのみ表示（例: https://***...com）
                    masked_endpoint = f"{parsed.scheme}://***...{parts[-1]}" if len(parts) >= 2 else f"{parsed.scheme}://***"
                else:
                    masked_endpoint = "***"
            except Exception:
                masked_endpoint = "***"
            logger.info(f"Azure OpenAI Embedding API initialized successfully (endpoint: {masked_endpoint})")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI Embedding API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        reraise=True
    )
    def _get_embeddings_with_retry(self, batch_texts: List[str]) -> List[List[float]]:
        """リトライ付きで Azure OpenAI Embedding API を呼び出す

        リトライ対象:
        - RateLimitError (429): レート制限エラー
        - APIConnectionError: 接続エラー

        リトライ対象外（即座に失敗）:
        - 認証エラー (401)
        - 無効なリクエスト (400)
        - モデル不在 (404)

        Args:
            batch_texts: テキストのバッチ

        Returns:
            埋め込み結果のリスト
        """
        try:
            deployment = self.config.azure_openai_embedding_deployment
            response = self.model.embeddings.create(
                input=batch_texts,
                model=deployment
            )
            # 埋め込みベクトルを抽出
            return [item.embedding for item in response.data]
        except APIStatusError as e:
            # 4xx系エラーは即座に失敗させる（リトライしても回復しない）
            if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                logger.error(f"回復不可能なAPIエラー（ステータス: {e.status_code}）")
                raise
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
                batch_vectors = self._get_embeddings_with_retry(batch_texts)

                # 正規化処理
                for vector in batch_vectors:
                    vec_array = np.array(vector)
                    if normalize_embeddings:
                        vec_array = self.normalize_vector(vec_array)
                    all_embeddings.append(vec_array)

            # numpy配列に変換
            result = np.array(all_embeddings)

            logger.info(f"Generated embeddings for {len(texts)} texts using Azure OpenAI")
            return result

        except Exception as e:
            logger.error(f"Error generating embeddings with Azure OpenAI: {e}")
            raise

    @property
    def embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""
        return self.EMBEDDING_DIM

    @property
    def provider_name(self) -> str:
        """プロバイダー名を返す"""
        return "azure_openai"
