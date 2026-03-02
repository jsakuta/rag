# 設定ガイド

> 初回セットアップは [README.md](../README.md) を参照してください。このドキュメントは設定の詳細リファレンスです。

このドキュメントでは、RAG-Localシステムの全環境変数と設定オプションを詳しく説明します。

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

> **Note:** `DEFAULT_EMBEDDING_MODEL` は廃止されました。埋め込みモデルはプロバイダーから自動解決されます（azure_openai → `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`、vertex_ai → `VERTEX_AI_EMBEDDING_MODEL`）。

### Google Cloud / VertexAI 設定

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
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

### GCP認証方式

`CREDENTIAL_SOURCE` でサービスアカウント認証情報の取得元を選択します。

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `CREDENTIAL_SOURCE` | 認証方式（`local` / `key_vault`） | `local` | `key_vault` |
| `GEMINI_CREDENTIALS_PATH` | ローカル認証ファイルパス | `gemini_credentials.json` | `creds/sa.json` |
| `AZURE_KEY_VAULT_URL` | Key Vault URL（`key_vault` 時必須） | - | `https://your-vault.vault.azure.net/` |
| `AZURE_KEY_VAULT_SECRET_NAME` | シークレット名（`key_vault` 時必須） | - | `gcp-sa-credentials` |
| `AZURE_KEY_VAULT_SCOPES` | GCPスコープ | `https://www.googleapis.com/auth/cloud-platform` | - |

**`local`（デフォルト）**: サービスアカウント JSON ファイルをローカルに配置して使用。

**`key_vault`**: Azure Key Vault にサービスアカウント JSON をシークレットとして格納し、`DefaultAzureCredential` で取得。ローカルにファイルを配置できない環境向け。

### その他

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `LOG_LEVEL` | ログレベル | `INFO` | `DEBUG` |

---

## LLM設定

### プロバイダーの選択

#### Gemini（推奨）

```env
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

# 必須: Google Cloud認証
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

---

## 埋め込みモデル設定

### プロバイダーの選択

#### Azure OpenAI（推奨）

```env
DEFAULT_EMBEDDING_PROVIDER=azure_openai
# モデルは AZURE_OPENAI_EMBEDDING_DEPLOYMENT から自動解決

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
# モデルは VERTEX_AI_EMBEDDING_MODEL から自動解決

# 必須: Google Cloud認証
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
    vector_weight=0.9,          # ベクトル検索の重み（keyword_weight は 1.0 - vector_weight で自動計算）

    # 検索モード: original | llm_enhanced | multi_stage（multi_stage は改定影響調査専用）
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
- 改定影響調査に適する

### 重み調整

`keyword_weight` は `1.0 - vector_weight` で自動計算されます。`vector_weight` のみ設定してください。
設定場所は `config/settings.yaml` の各セクション（`common`, `ui`, `batch`, `evaluation.revision_areas`）。

| vector_weight | keyword_weight（自動） | 用途 |
|---|---|---|
| 0.9 | 0.1 | 意味的類似性を重視（推奨） |
| 0.7 | 0.3 | 意味とキーワード両方を考慮 |
| 0.5 | 0.5 | 専門用語の完全一致を重視 |
| 0.3 | 0.7 | キーワード重視（用語置換検出等） |

---

## データベース設定

### ディレクトリ構造

以下のパスは `SearchConfig.base_dir`（デフォルト: プロジェクトルート）を基点としたハードコードパスで、環境変数では変更できません。

```
data/vector_db/          # ベクトルDB
data/source/scenarios/   # シナリオExcel
data/source/faq/         # FAQデータ
data/input/              # 入力ファイル
data/output/             # 出力ファイル
```

### コレクション命名規則

| 日本語 | コレクション名 | 備考 |
|--------|----------------|------|
| 内部事務 | naibujimu | 預金+総則を統合 |
| スマイル | smile | |
| rev01_smile | rev01_smile | 改定別（運用保守効率化AI（改定影響調査）用） |
| rev02_souzoku | rev02_souzoku | 改定別 |
| ... | rev{XX}_{bot} | 改定番号_ボット名 |

### タイムスタンプ管理

**ファイル:** `data/vector_db/update_timestamps.json`

```json
{
  "naibujimu_azure_openai_faq": 1735567200.0,
  "naibujimu_azure_openai_scenario": 1735567200.0,
  "smile_azure_openai_faq": 1735567200.0,
  "smile_azure_openai_scenario": 1735567200.0
}
```

**動作:**
- ファイル更新時刻とタイムスタンプを比較
- 変更があればDB再構築
- 変更なければ既存DBを使用

---

## 設定ファイル（YAML）

### config/settings.yaml

検索パラメータとアプリケーション動作を制御する設定ファイル。

| セクション | 用途 | 主な設定 |
|-----------|------|---------|
| `common` | 全プログラム共通 | search_type, vector_weight, search_mode, keyword設定 |
| `ui` | Streamlit UI専用 | top_k, search_type, vector_weight（スライダー初期値） |
| `batch` | バッチ処理専用 | top_k, vector_weight |
| `evaluation` | 改定影響調査専用 | max_results, filter_mode, thresholds, revision_areas |

> **重要:** settings.yaml は起動時に必須です。ファイルが存在しないか common セクションが欠落している場合、`RuntimeError` が発生します。全キーはフォールバックなしの直接アクセスのため、キー欠落時は `KeyError` になります。

**設定の読み込み方法**:
- `load_settings("ui")` → common + ui をマージして返す
- `load_settings("batch")` → common + batch をマージして返す
- セクション固有の値が common を上書き

### config/business_areas.yaml

業務分野の日本語名から ChromaDB コレクション名への変換マッピング。

| セクション | 用途 |
|-----------|------|
| `mappings` | 通常業務分野（スマイル→smile, 内部事務→naibujimu 等） |
| `revision_mappings` | 改定別DB名（rev01_smile, rev02_souzoku 等） |
| `collection_constraints` | ChromaDB命名制約（3-512文字、英数字+._-） |

---

## UI設定

### Streamlit 設定

```bash
# デフォルトポート
streamlit run apps/answer-support/ui/chat.py

# カスタムポート
streamlit run apps/answer-support/ui/chat.py --server.port 8502

# 外部アクセス許可
streamlit run apps/answer-support/ui/chat.py --server.address 0.0.0.0
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
from src.utils.logger import setup_logger

logger = setup_logger(__name__)  # ログレベルは LOG_LEVEL 環境変数で制御
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

### ローカル認証

```env
# ===== GCP認証 =====
CREDENTIAL_SOURCE=local
# GEMINI_CREDENTIALS_PATH=gemini_credentials.json  # デフォルト

# ===== Vertex AI =====
GEMINI_PROJECT_ID=your-project-id
GEMINI_LOCATION=us-central1

# ===== Azure OpenAI（埋め込み） =====
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# ===== デフォルト選択 =====
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite
DEFAULT_EMBEDDING_PROVIDER=azure_openai
```

### Key Vault 認証

```env
# ===== GCP認証（Key Vault経由） =====
CREDENTIAL_SOURCE=key_vault
AZURE_KEY_VAULT_URL=https://prod-vault.vault.azure.net/
AZURE_KEY_VAULT_SECRET_NAME=gcp-sa-credentials
AZURE_KEY_VAULT_SCOPES=https://www.googleapis.com/auth/cloud-platform

# ===== Vertex AI =====
GEMINI_PROJECT_ID=your-project-id
GEMINI_LOCATION=us-central1

# ===== Azure OpenAI（埋め込み） =====
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# ===== デフォルト選択 =====
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite
DEFAULT_EMBEDDING_PROVIDER=azure_openai
```

---

## 設定検証

### 起動時チェック

```python
# SearchConfig.__post_init__() で自動検証（config.py）
# SearchConfig インスタンス生成時に以下を検証:
# - vector_weight が 0〜1 の範囲
# - top_k が 1 以上の整数
# - search_mode が有効な値（original / llm_enhanced / multi_stage）
# - EMBEDDING_BATCH_SIZE が 1〜250 の範囲
# - embedding_provider が必須（未設定時 ValueError）
# - embedding_model をプロバイダーから自動解決:
#     azure_openai → AZURE_OPENAI_EMBEDDING_DEPLOYMENT 環境変数（デフォルト: text-embedding-3-large）
#     vertex_ai    → VERTEX_AI_EMBEDDING_MODEL 環境変数（デフォルト: gemini-embedding-001）
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

- [README.md](../README.md) - プロジェクト概要・セットアップ
- [docs/ANSWER_SUPPORT.md](./ANSWER_SUPPORT.md) - 回答支援AI詳細
- [docs/REVISION_OPS.md](./REVISION_OPS.md) - 改定影響調査詳細
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) - アーキテクチャ・API仕様
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
