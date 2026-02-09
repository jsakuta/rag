# 設定ガイド

このドキュメントでは、RAG-Geminiシステムの全環境変数と設定オプションを詳しく説明します。

## 目次

- [環境変数一覧](#環境変数一覧)
- [LLM設定](#llm設定)
- [埋め込みモデル設定](#埋め込みモデル設定)
- [検索設定](#検索設定)
- [データベース設定](#データベース設定)
- [UI設定](#ui設定)
- [ログ設定](#ログ設定)
- [設定ファイル例](#設定ファイル例)

---

## 環境変数一覧

### 必須環境変数

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `DEFAULT_LLM_PROVIDER` | LLMプロバイダー | **必須** | `gemini` |
| `DEFAULT_LLM_MODEL` | LLMモデル名 | **必須** | `gemini-2.5-flash-lite` |
| `DEFAULT_EMBEDDING_PROVIDER` | 埋め込みプロバイダー | **必須** | `azure_openai` |
| `DEFAULT_EMBEDDING_MODEL` | 埋め込みモデル名 | **必須** | `text-embedding-3-large` |

### Google Cloud / VertexAI 設定

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `GEMINI_CREDENTIALS_PATH` | 認証ファイルパス | `gemini_credentials.json` | `./secrets/credentials.json` |
| `GEMINI_PROJECT_ID` | Google Cloud プロジェクトID | **必須** | `pj-cbk001` |
| `GEMINI_LOCATION` | リージョン | `us-central1` | `asia-northeast1` |
| `VERTEX_AI_EMBEDDING_MODEL` | VertexAI埋め込みモデル | `gemini-embedding-001` | `text-embedding-004` |

### Azure OpenAI 設定

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `AZURE_OPENAI_API_KEY` | API キー | **必須** | `sk-...` |
| `AZURE_OPENAI_ENDPOINT` | エンドポイントURL | **必須** | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | 埋め込みデプロイメント名 | `text-embedding-3-large` | `your-deployment-name` |
| `AZURE_OPENAI_API_VERSION` | APIバージョン | `2024-12-01-preview` | `2024-12-01-preview` |

### Anthropic (Claude) 設定

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `ANTHROPIC_API_KEY` | API キー | - | `sk-ant-...` |

### OpenAI (ChatGPT) 設定

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `OPENAI_API_KEY` | API キー | - | `sk-...` |

### Azure Key Vault 設定（オプション）

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `AZURE_KEY_VAULT_URL` | Key Vault URL | - | `https://your-vault.vault.azure.net/` |
| `AZURE_KEY_VAULT_SCOPES` | スコープ | - | `https://www.googleapis.com/auth/cloud-platform` |

### その他

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `LOG_LEVEL` | ログレベル | `INFO` | `DEBUG` |
| `ENABLE_LLM_ANALYSIS` | LLM分析の有効化 | `true` | `false` |

---

## LLM設定

### プロバイダーの選択

#### Gemini（推奨）

```env
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

# 必須: Google Cloud認証
GEMINI_CREDENTIALS_PATH=gemini_credentials.json
GEMINI_PROJECT_ID=your-project-id
GEMINI_LOCATION=us-central1
```

**特徴:**
- 高速レスポンス
- 日本語対応
- コスト効率が良い

**利用可能なモデル:**
- `gemini-2.5-flash-lite` - 最速、低コスト（推奨）
- `gemini-2.5-flash` - バランス型
- `gemini-2.5-pro` - 高精度

#### Anthropic Claude

```env
DEFAULT_LLM_PROVIDER=anthropic
DEFAULT_LLM_MODEL=claude-3-5-sonnet-20241022

# 必須: Anthropic API キー
ANTHROPIC_API_KEY=sk-ant-...
```

**特徴:**
- 高精度分析
- 長文対応
- 高コスト

**利用可能なモデル:**
- `claude-3-5-sonnet-20241022` - 最新（推奨）
- `claude-3-5-haiku-20241022` - 高速・低コスト
- `claude-opus-4-6` - 最高精度

#### OpenAI ChatGPT

```env
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o

# 必須: OpenAI API キー
OPENAI_API_KEY=sk-...
```

**特徴:**
- 汎用性が高い
- 広範な用途

**利用可能なモデル:**
- `gpt-4o` - 最新（推奨）
- `gpt-4-turbo` - 高速
- `gpt-3.5-turbo` - 低コスト

---

## 埋め込みモデル設定

### プロバイダーの選択

#### Azure OpenAI（推奨）

```env
DEFAULT_EMBEDDING_PROVIDER=azure_openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-large

# 必須: Azure OpenAI認証
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

**特徴:**
- 高精度（3072次元）
- 安定性が高い
- エンタープライズ向け

**性能:**
- バッチサイズ: 16
- API上限: 2048トークン/リクエスト

#### VertexAI Gemini

```env
DEFAULT_EMBEDDING_PROVIDER=vertex_ai
DEFAULT_EMBEDDING_MODEL=gemini-embedding-001

# 必須: Google Cloud認証
GEMINI_CREDENTIALS_PATH=gemini_credentials.json
GEMINI_PROJECT_ID=your-project-id
VERTEX_AI_EMBEDDING_MODEL=gemini-embedding-001
```

**特徴:**
- Google Cloud統合
- MRL（Matryoshka Representation Learning）対応
- 柔軟な次元数選択

**性能:**
- バッチサイズ: 5
- API上限: 250テキスト/リクエスト
- 次元数: 3072（デフォルト）/ 1536 / 768

---

## 検索設定

### SearchConfig パラメータ

**ファイル:** `config.py`

```python
from config import SearchConfig

config = SearchConfig(
    # 検索パラメータ
    top_k=4,                    # 返却する結果数
    vector_weight=0.9,          # ベクトル検索の重み
    keyword_weight=0.1,         # キーワード検索の重み

    # 検索モード: original | llm_enhanced | multi_stage
    search_mode="original",

    # 参照データ形式
    reference_type="multi_folder",   # excel | hierarchical_excel | multi_folder
)
```

### 検索モード

#### 原文検索モード（デフォルト）

```python
search_mode="original"
```

**特徴:**
- 高速（LLM API呼び出しなし）
- 直接的な類似性検索
- コスト効率が良い

#### LLM拡張検索モード

```python
search_mode="llm_enhanced"
```

**特徴:**
- 高精度（意図理解）
- LLMがクエリを最適化

#### 多段階検索モード

```python
search_mode="multi_stage"
```

**特徴:**
- 原文検索 + LLM拡張検索のOR結合
- 最高の網羅性
- 事務改定評価に適する

### 重み調整

#### ベクトル重視（推奨）

```python
vector_weight=0.9
keyword_weight=0.1
```

**用途:** 意味的類似性を重視

#### バランス型

```python
vector_weight=0.7
keyword_weight=0.3
```

**用途:** 意味とキーワード両方を考慮

#### キーワード重視

```python
vector_weight=0.5
keyword_weight=0.5
```

**用途:** 専門用語の完全一致を重視

---

## データベース設定

### ディレクトリ構造

```env
# デフォルト設定
DB_BASE_DIR=reference/vector_db
REFERENCE_SCENARIO_DIR=reference/scenario
REFERENCE_FAQ_DIR=reference/faq_data
INPUT_DIR=input
OUTPUT_DIR=output
```

### コレクション命名規則

| 日本語 | コレクション名 |
|--------|----------------|
| 預金 | deposit_DB |
| 融資 | loan_DB |
| 外貨 | foreign_currency_DB |
| 投信 | investment_trust_DB |
| 住宅ローン | housing_loan_DB |
| カード | card_DB |
| 保険 | insurance_DB |
| 年金 | pension_DB |
| 総則 | general_DB |

### タイムスタンプ管理

**ファイル:** `reference/vector_db/update_timestamps.json`

```json
{
  "faq": {
    "deposit": 1735567200.0,
    "loan": 1735567200.0
  },
  "scenario": {
    "deposit": 1735567200.0,
    "loan": 1735567200.0
  }
}
```

**動作:**
- ファイル更新時刻とタイムスタンプを比較
- 変更があればDB再構築
- 変更なければ既存DBを使用

---

## UI設定

### Streamlit 設定

```bash
# デフォルトポート
streamlit run ui/chat.py

# カスタムポート
streamlit run ui/chat.py --server.port 8502

# 外部アクセス許可
streamlit run ui/chat.py --server.address 0.0.0.0
```

### Docker 設定

```dockerfile
# Dockerfile
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## ログ設定

### ログレベル

```env
# 環境変数
LOG_LEVEL=INFO  # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

```python
# コード内
import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__, level=logging.DEBUG)
```

### ログフォーマット

```
2025-01-01 12:00:00 - INFO - module_name - メッセージ
```

### ログファイル

```
logs/
├── app.log          # メインログ
├── error.log        # エラーログ（将来実装予定）
└── access.log       # アクセスログ（将来実装予定）
```

---

## 設定ファイル例

### .env.example（開発環境）

```env
# ===== LLM設定 =====
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

# ===== 埋め込みモデル設定 =====
DEFAULT_EMBEDDING_PROVIDER=azure_openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-large

# ===== Google Cloud / VertexAI =====
GEMINI_CREDENTIALS_PATH=gemini_credentials.json
GEMINI_PROJECT_ID=your-project-id
GEMINI_LOCATION=us-central1
VERTEX_AI_EMBEDDING_MODEL=gemini-embedding-001

# ===== Azure OpenAI =====
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# ===== Anthropic (オプション) =====
# ANTHROPIC_API_KEY=sk-ant-...

# ===== OpenAI (オプション) =====
# OPENAI_API_KEY=sk-...

# ===== Azure Key Vault (オプション) =====
# AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net/
# AZURE_KEY_VAULT_SCOPES=https://www.googleapis.com/auth/cloud-platform

# ===== その他 =====
LOG_LEVEL=INFO
ENABLE_LLM_ANALYSIS=true
```

### .env（本番環境）

```env
# ===== LLM設定 =====
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

# ===== 埋め込みモデル設定 =====
DEFAULT_EMBEDDING_PROVIDER=azure_openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-large

# ===== Google Cloud / VertexAI =====
GEMINI_CREDENTIALS_PATH=/app/secrets/gemini_credentials.json
GEMINI_PROJECT_ID=prod-project-id
GEMINI_LOCATION=asia-northeast1

# ===== Azure OpenAI =====
AZURE_OPENAI_API_KEY=${KEY_VAULT:azure-openai-key}
AZURE_OPENAI_ENDPOINT=https://prod-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large-prod
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# ===== Azure Key Vault =====
AZURE_KEY_VAULT_URL=https://prod-vault.vault.azure.net/
AZURE_KEY_VAULT_SCOPES=https://www.googleapis.com/auth/cloud-platform

# ===== その他 =====
LOG_LEVEL=WARNING
ENABLE_LLM_ANALYSIS=true
```

---

## 設定検証

### 起動時チェック

```python
# config.py で実装済み
def validate_config():
    """設定の妥当性をチェック"""

    required_vars = [
        "DEFAULT_LLM_PROVIDER",
        "DEFAULT_LLM_MODEL",
        "DEFAULT_EMBEDDING_PROVIDER",
        "DEFAULT_EMBEDDING_MODEL",
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Required environment variable {var} is not set")

    # プロバイダー別の必須変数チェック
    if os.getenv("DEFAULT_EMBEDDING_PROVIDER") == "azure_openai":
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            raise ValueError("AZURE_OPENAI_API_KEY is required for azure_openai provider")
```

### 手動検証

```bash
# .env ファイルの読み込み確認
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('DEFAULT_LLM_PROVIDER'))"

# 設定オブジェクトの確認
python -c "from config import SearchConfig; config = SearchConfig(); print(config)"
```

---

## トラブルシューティング

### 環境変数が読み込まれない

**原因:** `.env` ファイルが正しい場所にない

**解決策:**
```bash
# プロジェクトルートに配置
ls -l .env

# パーミッション確認
chmod 600 .env
```

### API キーエラー

**原因:** 環境変数が設定されていない

**解決策:**
```bash
# 環境変数の確認
echo $AZURE_OPENAI_API_KEY

# .env ファイルを再読み込み
source .env  # Linux/Mac
```

詳細は [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) を参照してください。

---

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要
- [docs/GOOGLE_CLOUD_AUTH.md](./GOOGLE_CLOUD_AUTH.md) - Google Cloud 認証
- [docs/SECURITY.md](./SECURITY.md) - セキュリティガイド
- [docs/API_REFERENCE.md](./API_REFERENCE.md) - API仕様
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
