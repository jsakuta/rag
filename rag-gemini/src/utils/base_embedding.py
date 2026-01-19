# --- utils/base_embedding.py ---
"""埋め込みモデルの抽象基底クラス"""
from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from config import SearchConfig


class BaseEmbeddingModel(ABC):
    """埋め込みモデルの抽象基底クラス

    Gemini (Vertex AI) と Azure OpenAI text-embedding-3-large の
    両方をサポートするための共通インターフェースを定義します。
    """

    def __init__(self, config: 'SearchConfig'):
        """
        Args:
            config: SearchConfig インスタンス
        """
        self.config = config
        self.model = self._setup_model()

    @abstractmethod
    def _setup_model(self):
        """モデルの初期化（サブクラスで実装）

        Returns:
            初期化されたモデルインスタンス
        """
        pass

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True) -> np.ndarray:
        """テキストをベクトル化

        Args:
            texts: 単一テキストまたはテキストのリスト
            normalize_embeddings: ベクトルを正規化するかどうか

        Returns:
            numpy.ndarray: 埋め込みベクトル（2次元配列）
        """
        pass

    def encode_single(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """単一テキストをベクトル化

        Args:
            text: ベクトル化するテキスト
            normalize_embeddings: ベクトルを正規化するかどうか

        Returns:
            numpy.ndarray: 埋め込みベクトル（1次元配列）
        """
        return self.encode([text], normalize_embeddings)[0]

    # Division by Zero防止: ノルムの最小閾値
    NORM_EPSILON = 1e-10

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """ベクトルをL2正規化（数値安定性を考慮）

        Args:
            vector: 正規化するベクトル

        Returns:
            numpy.ndarray: 正規化されたベクトル

        Note:
            ノルムが非常に小さい場合（NORM_EPSILON以下）は、
            数値的な不安定性を避けるためゼロベクトルを返します。
        """
        norm = np.linalg.norm(vector)
        # Division by Zero + 数値安定性: ノルムが極めて小さい場合はゼロベクトルを返す
        if norm > BaseEmbeddingModel.NORM_EPSILON:
            return vector / norm
        # ゼロまたは極めて小さいノルムの場合は元のベクトルを返す
        # （ゼロベクトルの正規化を試みた可能性があるため）
        return vector

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """プロバイダー名を返す（例: "vertex_ai", "azure_openai"）"""
        pass
