# --- utils/auth.py ---
"""Google Cloud認証処理の共通モジュール"""
import os
from typing import TYPE_CHECKING, Optional

import vertexai
from google.oauth2 import service_account
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from config import SearchConfig

logger = setup_logger(__name__)


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
