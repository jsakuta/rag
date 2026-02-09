"""pytest共通設定とフィクスチャ"""
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SearchConfig


@pytest.fixture
def mock_config():
    """モックSearchConfig"""
    return SearchConfig(
        base_dir=str(PROJECT_ROOT),
        top_k=5,
        vector_weight=0.9,
        search_mode="original",
        embedding_provider="azure_openai",
        embedding_model="text-embedding-3-large"
    )


@pytest.fixture
def sample_texts():
    """テスト用サンプルテキスト"""
    return [
        "質問: 口座開設の方法 | 回答: 窓口で申込書を記入してください",
        "質問: 残高照会の方法 | 回答: ATMまたはアプリで確認できます",
        "質問: 振込手数料について | 回答: 同一銀行は無料、他行は220円です"
    ]


@pytest.fixture
def sample_query():
    """テスト用クエリ"""
    return "口座開設の手続き方法を教えてください"
