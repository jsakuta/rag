# --- utils/auth.py ---
"""Google Cloud認証処理の共通モジュール + 埋め込みモデルファクトリー"""
import json
import os
from typing import TYPE_CHECKING

from src.utils.logger import setup_logger

# 遅延インポート: 認証モジュール
service_account = None

if TYPE_CHECKING:
    from config import SearchConfig
    from src.utils.base_embedding import BaseEmbeddingModel

logger = setup_logger(__name__)


def _load_auth_modules():
    """認証モジュールのみ遅延ロード（軽量・高速）"""
    global service_account
    if service_account is None:
        from google.oauth2 import service_account as _sa
        service_account = _sa


def _get_credentials_local(config: 'SearchConfig'):
    """ローカルのサービスアカウント JSON ファイルから認証情報を取得

    Args:
        config: SearchConfig インスタンス

    Returns:
        service_account.Credentials: 認証情報

    Raises:
        FileNotFoundError: 認証ファイルが存在しない場合
    """
    _load_auth_modules()
    credentials_path = os.path.join(config.base_dir, config.gemini_credentials_path)
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f'認証ファイルが見つかりません: {credentials_path}')

    logger.info(f"Using local credentials file: {config.gemini_credentials_path}")
    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


def _get_credentials_key_vault(config: 'SearchConfig'):
    """Azure Key Vault からサービスアカウント認証情報を取得

    Key Vault に格納された Google サービスアカウント JSON をシークレットとして取得し、
    認証情報オブジェクトを生成する。

    Args:
        config: SearchConfig インスタンス

    Returns:
        service_account.Credentials: 認証情報

    Raises:
        ValueError: Key Vault の設定が不完全な場合
    """
    _load_auth_modules()
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=config.azure_key_vault_url, credential=credential)

    secret = client.get_secret(config.azure_key_vault_secret_name)
    service_account_info = json.loads(secret.value)

    scopes = [config.azure_key_vault_scopes]
    logger.info("Google credentials retrieved from Azure Key Vault")
    return service_account.Credentials.from_service_account_info(
        service_account_info, scopes=scopes
    )


# credential_source → 取得関数のマッピング
_CREDENTIAL_HANDLERS = {
    "local": _get_credentials_local,
    "key_vault": _get_credentials_key_vault,
}


def get_google_credentials(config: 'SearchConfig'):
    """Google Cloud認証情報を取得

    config.credential_source の設定に応じて認証方式を切り替える:
    - "local"（デフォルト）: ローカルのサービスアカウント JSON ファイル
    - "key_vault": Azure Key Vault に格納されたサービスアカウント JSON

    Args:
        config: SearchConfig インスタンス

    Returns:
        service_account.Credentials: 認証情報
    """
    handler = _CREDENTIAL_HANDLERS.get(config.credential_source)
    if handler is None:
        raise ValueError(f"Unknown credential_source: {config.credential_source}")

    return handler(config)


def create_genai_client(config: 'SearchConfig', credentials=None):
    """google-genai Client を生成

    Args:
        config: SearchConfig インスタンス
        credentials: 認証情報（省略時は自動取得）

    Returns:
        genai.Client インスタンス
    """
    from google import genai
    if credentials is None:
        credentials = get_google_credentials(config)
    client = genai.Client(
        vertexai=True,
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials,
    )
    logger.info("google-genai Client initialized successfully")
    return client


def create_llm(config: 'SearchConfig'):
    """Gemini LLMインスタンスを生成

    Args:
        config: SearchConfig インスタンス

    Returns:
        ChatGoogleGenerativeAI インスタンス

    Raises:
        ValueError: プロバイダーが gemini でない、または設定が不足している場合
    """
    if config.llm_provider != "gemini":
        raise ValueError(
            f"Unsupported LLM provider: {config.llm_provider}. "
            f"Currently only 'gemini' is supported"
        )

    if not config.gemini_project_id:
        raise ValueError("GEMINI_PROJECT_ID environment variable is not set")

    from langchain_google_genai import ChatGoogleGenerativeAI
    credentials = get_google_credentials(config)
    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=0,
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials,
    )


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
