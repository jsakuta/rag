# --- utils/auth.py ---
"""Google Cloud認証処理の共通モジュール + 埋め込みモデルファクトリー"""
import os
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, Type

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from src.utils.logger import setup_logger

# 遅延インポート: Vertex AI関連（インストールされていない場合もエラーにしない）
vertexai = None
service_account = None
ChatVertexAI = None

if TYPE_CHECKING:
    from config import SearchConfig
    from src.utils.base_embedding import BaseEmbeddingModel

logger = setup_logger(__name__)

# LLMプロバイダー設定マッピング
LLM_PROVIDER_CONFIG: Dict[str, Tuple[str, Type, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", ChatAnthropic, "anthropic_api_key"),
    "openai": ("OPENAI_API_KEY", ChatOpenAI, "api_key"),
}


def _load_vertex_ai_modules():
    """Vertex AI関連モジュールを遅延ロード"""
    global vertexai, service_account, ChatVertexAI
    if vertexai is None:
        import vertexai as _vertexai
        from google.oauth2 import service_account as _service_account
        from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
        vertexai = _vertexai
        service_account = _service_account
        ChatVertexAI = _ChatVertexAI


def get_google_credentials(config: 'SearchConfig'):
    """Google Cloud認証情報を取得

    Args:
        config: SearchConfig インスタンス

    Returns:
        service_account.Credentials: 認証情報

    Raises:
        FileNotFoundError: 認証ファイルが存在しない場合
    """
    _load_vertex_ai_modules()
    credentials_path = os.path.join(config.base_dir, config.gemini_credentials_path)

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f'認証ファイルが見つかりません: {credentials_path}')

    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


def initialize_vertex_ai(
    config: 'SearchConfig',
    credentials = None
) -> None:
    """Vertex AIを初期化

    Args:
        config: SearchConfig インスタンス
        credentials: 認証情報（省略時は自動取得）

    Raises:
        FileNotFoundError: 認証ファイルが存在しない場合
    """
    _load_vertex_ai_modules()
    if credentials is None:
        credentials = get_google_credentials(config)

    vertexai.init(
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials
    )
    logger.info("Vertex AI initialized successfully")


def create_llm(config: 'SearchConfig'):
    """LLMプロバイダーに応じたLLMインスタンスを生成

    Args:
        config: SearchConfig インスタンス

    Returns:
        LLMインスタンス（ChatAnthropic, ChatOpenAI, または ChatVertexAI）

    Raises:
        ValueError: サポートされていないプロバイダーまたはAPI キーが未設定の場合
    """
    provider = config.llm_provider

    # Vertex AI (Gemini) の場合は専用の認証を使用
    if provider == "gemini":
        _load_vertex_ai_modules()
        if not config.gemini_project_id:
            raise ValueError("GEMINI_PROJECT_ID environment variable is not set")
        credentials = get_google_credentials(config)
        initialize_vertex_ai(config, credentials)
        return ChatVertexAI(
            model=config.llm_model,
            temperature=0,
            project=config.gemini_project_id,
            location=config.gemini_location,
            credentials=credentials,
        )

    # その他のプロバイダー
    if provider not in LLM_PROVIDER_CONFIG:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    env_key, llm_class, api_param = LLM_PROVIDER_CONFIG[provider]
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError(f"{env_key} environment variable is not set")

    return llm_class(**{api_param: api_key, "model": config.llm_model, "temperature": 0})


def create_embedding_model(config: 'SearchConfig', use_singleton: bool = True) -> 'BaseEmbeddingModel':
    """埋め込みモデルのファクトリー関数

    config.embedding_provider に応じて適切な埋め込みモデルを生成します。

    Args:
        config: SearchConfig インスタンス
        use_singleton: シングルトンインスタンスを使用するかどうか（デフォルト: True）

    Returns:
        BaseEmbeddingModel: 埋め込みモデルインスタンス

    Raises:
        ValueError: サポートされていないプロバイダーの場合

    使用例:
        >>> from config import SearchConfig
        >>> from src.utils.auth import create_embedding_model
        >>>
        >>> # Gemini (Vertex AI) を使用
        >>> config = SearchConfig(embedding_provider="vertex_ai")
        >>> model = create_embedding_model(config)
        >>>
        >>> # Azure OpenAI を使用
        >>> config = SearchConfig(embedding_provider="azure_openai")
        >>> model = create_embedding_model(config)
    """
    provider = config.embedding_provider

    if provider == "vertex_ai":
        from src.utils.gemini_embedding import GeminiEmbeddingModel
        if use_singleton:
            return GeminiEmbeddingModel.get_instance(config)
        return GeminiEmbeddingModel(config)

    elif provider == "azure_openai":
        from src.utils.azure_embedding import AzureOpenAIEmbeddingModel
        if use_singleton:
            return AzureOpenAIEmbeddingModel.get_instance(config)
        return AzureOpenAIEmbeddingModel(config)

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: vertex_ai, azure_openai"
        )
