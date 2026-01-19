# --- utils/auth.py ---
"""Google Cloud認証処理の共通モジュール"""
import os
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, Type

import vertexai
from google.oauth2 import service_account
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from config import SearchConfig

logger = setup_logger(__name__)

# LLMプロバイダー設定マッピング
LLM_PROVIDER_CONFIG: Dict[str, Tuple[str, Type, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", ChatAnthropic, "anthropic_api_key"),
    "openai": ("OPENAI_API_KEY", ChatOpenAI, "api_key"),
}


def get_google_credentials(config: 'SearchConfig') -> service_account.Credentials:
    """Google Cloud認証情報を取得

    Args:
        config: SearchConfig インスタンス

    Returns:
        service_account.Credentials: 認証情報

    Raises:
        FileNotFoundError: 認証ファイルが存在しない場合
    """
    credentials_path = os.path.join(config.base_dir, config.gemini_credentials_path)

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f'認証ファイルが見つかりません: {credentials_path}')

    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


def initialize_vertex_ai(
    config: 'SearchConfig',
    credentials: Optional[service_account.Credentials] = None
) -> None:
    """Vertex AIを初期化

    Args:
        config: SearchConfig インスタンス
        credentials: 認証情報（省略時は自動取得）

    Raises:
        FileNotFoundError: 認証ファイルが存在しない場合
    """
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
