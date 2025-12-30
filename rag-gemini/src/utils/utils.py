# --- utils/utils.py ---
import os
import json
from typing import Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from google.oauth2 import service_account
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def get_secret_from_key_vault(vault_url: str, secret_name: str) -> Optional[str]:
    """
    Azure Key Vaultからシークレットを取得
    
    Args:
        vault_url: Key VaultのURL
        secret_name: シークレット名
    
    Returns:
        シークレット値（取得できない場合はNone）
    """
    try:
        # Azure認証情報を取得
        credential = DefaultAzureCredential()
        
        # Key Vaultクライアントを作成
        secret_client = SecretClient(vault_url=vault_url, credential=credential)
        
        # シークレットを取得
        secret = secret_client.get_secret(secret_name)
        secret_value = secret.value
        
        logger.info(f"Successfully retrieved secret from Key Vault: {vault_url}/{secret_name}")
        return secret_value
        
    except Exception as e:
        logger.error(f"Failed to retrieve secret from Key Vault: {e}")
        return None

def get_google_credentials_from_key_vault(vault_url: str, secret_name: str, scopes: str) -> Optional[service_account.Credentials]:
    """
    Key VaultからGoogle Cloud認証情報を取得
    
    Args:
        vault_url: Key VaultのURL
        secret_name: シークレット名
        scopes: 認証スコープ
    
    Returns:
        Google Cloud認証情報（取得できない場合はNone）
    """
    try:
        # Key Vaultから認証情報を取得
        google_credentials_json = get_secret_from_key_vault(vault_url, secret_name)
        
        if not google_credentials_json:
            return None
        
        # JSON形式の認証情報を解析
        credentials_info = json.loads(google_credentials_json)
        
        # サービスアカウント認証情報を作成
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info, 
            scopes=[scopes]
        )
        
        logger.info("Successfully created Google Cloud credentials from Key Vault")
        return credentials
        
    except Exception as e:
        logger.error(f"Failed to create Google Cloud credentials: {e}")
        return None

def get_google_api_key_from_key_vault(vault_url: str, secret_name: str) -> Optional[str]:
    """
    Key VaultからGoogle APIキーを取得
    
    Args:
        vault_url: Key VaultのURL
        secret_name: シークレット名
    
    Returns:
        APIキー（取得できない場合はNone）
    """
    # Key Vaultから取得を試行
    api_key = get_secret_from_key_vault(vault_url, secret_name)
    if api_key:
        return api_key
    
    # 環境変数から取得を試行
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        logger.info("Using API key from environment variable")
        return api_key
    
    logger.warning("No API key found in Key Vault or environment variables")
    return None 