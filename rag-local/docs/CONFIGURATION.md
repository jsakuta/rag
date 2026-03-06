# 設定ガイド

> 初回セットアップは [README.md](../README.md) を参照してください。このドキュメントは設定の詳細リファレンスです。

このドキュメントでは、RAG-Localシステムの全環境変数と設定オプションを詳しく説明します。

## 目次

- [環境変数一覧](#環境変数一覧)
- [検索設定](#検索設定)
- [データベース設定](#データベース設定)
- [UI設定](#ui設定)
- [ログ設定](#ログ設定)

---

## 環境変数一覧

### 必須環境変数

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `DEFAULT_LLM_PROVIDER` | LLM（大規模言語モデル）のプロバイダー（`gemini` のみサポート）。利用可能なモデル: gemini-2.5-flash-lite（推奨）/ gemini-2.5-flash / gemini-2.5-pro | **必須** | `gemini` |
| `DEFAULT_LLM_MODEL` | LLMのモデル名 | **必須** | `gemini-2.5-flash-lite` |
| `DEFAULT_EMBEDDING_PROVIDER` | 埋め込みプロバイダー（文章を数値に変換するサービス）。埋め込みモデルはプロバイダーに応じて自動決定される（azure_openai → `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`、vertex_ai → `VERTEX_AI_EMBEDDING_MODEL`）。azure_openai: 3072次元、バッチ250。vertex_ai: 3072次元、バッチ100（SDK上限キャップ） | **必須** | `azure_openai` |

### Google Cloud / VertexAI 設定

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `GEMINI_PROJECT_ID` | Google Cloud プロジェクトID | **必須** | `pj-cbk001` |
| `GEMINI_LOCATION` | リージョン | `us-central1` | `asia-northeast1` |
| `VERTEX_AI_EMBEDDING_MODEL` | VertexAI埋め込みモデル | `gemini-embedding-001` | `text-embedding-004` |

### Azure OpenAI 設定

> **Note:** 回答支援AI（類似回答検索）のみ使用し `DEFAULT_EMBEDDING_PROVIDER=vertex_ai` に設定する場合、以下の設定は不要です。運用保守効率化AI（改定影響調査）の `run_eval.py` をデフォルト（`--provider both`）で実行する場合に必須となります。

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

### 運用保守効率化AI オプション

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `ENABLE_LLM_ANALYSIS` | LLM関連性判定の有効化（`run_eval.py` 専用） | `false` | `true` |

### その他

| 変数名 | 説明 | デフォルト値 | 例 |
|--------|------|------------|-----|
| `LOG_LEVEL` | ログレベル | `INFO` | `DEBUG` |

> **Note:** `GOOGLE_APPLICATION_CREDENTIALS` は Google Cloud SDK の標準環境変数で、認証情報JSONファイルの絶対パスを指定します。`.env` ではなく OS 環境変数として設定してください（`config.py` では直接参照しません）。

### 環境変数の必須度分類

| 分類 | 変数 | 条件 |
|------|------|------|
| **常に必須** | `CREDENTIAL_SOURCE`, `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL`, `DEFAULT_EMBEDDING_PROVIDER`, `GEMINI_PROJECT_ID`, `GEMINI_LOCATION` | 全機能で必要 |
| **プロバイダー別必須** | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` | `azure_openai` 使用時 |
| **プロバイダー別必須** | `GOOGLE_APPLICATION_CREDENTIALS` | `vertex_ai` 使用時（OS環境変数として設定） |
| **オプション** | `VERTEX_AI_EMBEDDING_MODEL`, `GEMINI_CREDENTIALS_PATH`, `GEMINI_LOCATION`, `ENABLE_LLM_ANALYSIS`, `LOG_LEVEL` | デフォルト値あり、または特定機能のみ |
| **条件付き必須** | `AZURE_KEY_VAULT_URL`, `AZURE_KEY_VAULT_SECRET_NAME` | `CREDENTIAL_SOURCE=key_vault` 時のみ |

---

## 検索設定

### SearchConfig パラメータ

**ファイル:** `config.py`

> **Note:** デフォルト値は `config/settings.yaml` の各セクション（`common`, `batch`, `ui`, `evaluation`）から動的にロードされます。以下のコード例の値はリテラルではなく、settings.yaml を変更することでデフォルト値を変更できます。

```python
from config import SearchConfig

config = SearchConfig(
    # 検索パラメータ
    top_k=4,                    # 返却する結果数（batch=4, ui=3, eval=改定ごと）
    vector_weight=0.9,          # ベクトル検索の重み

    # 検索モード: original | llm_enhanced
    search_mode="original",

    # 検索タイプ: hybrid | keyword_filter
    search_type="hybrid",       # hybrid=ベクトル+キーワード, keyword_filter=キーワードのみ

    # 参照データ形式
    reference_type="multi_folder",   # excel | hierarchical_excel | multi_folder
)
```

### SearchConfig の主要フィールド一覧

| フィールド | 型 | デフォルト値 | 説明 |
|-----------|-----|-----------|------|
| `top_k` | int | batch=4, ui=3 | 返却する結果数 |
| `vector_weight` | float | 0.9 | ベクトル検索の重み（0.0〜1.0） |
| `search_type` | str | `hybrid` | 検索アルゴリズム（`hybrid` / `keyword_filter`） |
| `search_mode` | str | `original` | クエリ処理方法。回答支援AI（類似回答検索）: `original` / `llm_enhanced`。運用保守効率化AI（改定影響調査）: 上記に加え `multi_stage`（多段階検索） |
| `search_source` | str | `history_data` | 検索対象（`scenario` / `history_data`） |
| `reference_type` | str | `multi_folder` | 参照データ形式 |
| `multi_stage_threshold` | float | 0.45 | 多段階検索の統合スコア閾値 |
| `multi_stage_max_results` | int | 100 | 多段階検索の各検索結果最大数 |
| `multi_stage_enable_judgment_support` | bool | True | LLM判断支援の有効化 |
| `include_hierarchy_in_vector` | bool | True | 階層情報をベクトル化に含める |
| `force_db_update` | bool | False | 強制DB更新フラグ |
| `embedding_provider` | str | 環境変数 | 埋め込みプロバイダー |
| `llm_provider` | str | 環境変数 | LLMプロバイダー（`gemini` のみ） |
| `credential_source` | str | `local` | GCP認証方式（`local` / `key_vault`） |

### 検索設定の2軸

検索は `search_mode` と `search_type` の2つの独立した設定で制御されます:

| 設定 | 値 | 説明 |
|------|-----|------|
| `search_mode` | `original` / `llm_enhanced` / `multi_stage`* | クエリの処理方法（原文そのまま / LLMで拡張 / 多段階で網羅的に検索） |
| `search_type` | `hybrid` / `keyword_filter` | 検索アルゴリズム（ベクトル+キーワード / キーワードのみ） |

> \* `multi_stage` は運用保守効率化AI（改定影響調査）専用です。詳細は [REVISION_OPS.md](./REVISION_OPS.md) を参照。

- `search_type=hybrid`: 意味の近さで検索する方式（ベクトル検索）とキーワードの一致で検索する方式を、`vector_weight` で重み付けして合算
- `search_type=keyword_filter`: 事前構築されたキーワードキャッシュでキーワードマッチのみ実行（ベクトル検索なし、`vector_weight` は不使用）

回答支援AI（UI）では `search_type` は `hybrid` に固定されています。運用保守効率化AI（改定影響調査）では改定ごとに `settings.yaml` の `evaluation.revision_areas` で設定します。

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

**前提条件:**
- `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL` は全モードで起動時に必須（`SearchConfig` バリデーション）
- `GEMINI_PROJECT_ID` + GCP認証が `llm_enhanced` / `multi_stage` モードで追加で必要
- LLM API 未到達時エラー: `ValueError: DEFAULT_LLM_PROVIDER環境変数が設定されていません`
- LLM API 呼び出しが不要な場合は `search_mode: original` を使用（環境変数は必要）

#### 多段階検索モード（運用保守効率化AI専用）

> **Note:** このモードは運用保守効率化AI（改定影響調査）専用です。回答支援AI（類似回答検索）には組み込まれていません。

```python
search_mode="multi_stage"
```

**特徴:**
- 原文での検索結果とLLMで拡張した検索結果を合わせて、漏れを減らす（OR結合）
- 検索の網羅性が最も高い
- 改定影響調査で、変更の影響範囲を広く把握するために使用する

### 重み調整

ベクトル検索とキーワード検索の重みバランスを調整します。`vector_weight` を設定すると、`keyword_weight` は `1.0 - vector_weight` で自動計算されます（`config.py` のプロパティ）。

設定場所は `config/settings.yaml` の各セクション（`common`, `ui`, `batch`, `evaluation.revision_areas`）。

| vector_weight | keyword_weight（自動） | 用途 |
|---|---|---|
| 0.9 | 0.1 | 意味的類似性を重視（推奨） |
| 0.7 | 0.3 | 意味とキーワード両方を考慮 |
| 0.5 | 0.5 | 専門用語の完全一致を重視 |
| 0.3 | 0.7 | キーワード重視（用語置換検出等） |

### top_k 設定一覧

| 用途 | 設定場所 | 値 | 説明 |
|------|---------|-----|------|
| 回答支援AI（UI） | `ui.top_k` | 3 | 画面表示用 |
| 回答支援AI（バッチ） | `batch.top_k` | 4 | Excel出力用 |
| 運用保守効率化AI（改定影響調査） | `evaluation.top_k` | 130 | 網羅性重視（`filter_mode: top_k` 時） |

> 評価用の top_k が大きいのは、改定影響調査が再現率（漏れの少なさ）を重視するため。この制限はエリア単位で適用される（複数エリアの改定では 130 × エリア数）。設計意図の詳細は [REVISION_OPS.md の設定値セクション](./REVISION_OPS.md#設定値) を参照。

---

## データベース設定

### ディレクトリ構造

以下のパスは `SearchConfig.base_dir`（デフォルト: プロジェクトルート）を基点としたハードコードパスで、環境変数では変更できません。

```
data/vector_db/          # 検索用データベース（ベクトルDB）
data/source/scenarios/   # シナリオExcel
data/source/faq/         # 問い合わせ履歴データ（FAQ）
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
| `common` | 全プログラム共通 | search_type, vector_weight, search_mode, search_source, keyword設定, columns設定 |
| `ui` | Streamlit UI専用 | top_k, search_type, vector_weight（スライダー初期値） |
| `batch` | バッチ処理専用 | top_k, vector_weight |
| `evaluation` | 運用保守効率化AI（改定影響調査）専用 | max_results, filter_mode, thresholds, revision_areas（[詳細](./REVISION_OPS.md#新しい改定の追加手順)） |

### 運用保守効率化AI — evaluation の詳細パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `max_results` | int | 100 | 最大検索結果数 |
| `filter_mode` | str | `top_k` | フィルタリング（`threshold` / `top_k`） |
| `top_k` | int | 130 | `filter_mode: top_k` 時の上位K件数 |
| `thresholds.azure_openai` | float | 0.40 | Azure OpenAI 類似度閾値 |
| `thresholds.vertex_ai` | float | 0.50 | VertexAI 類似度閾値 |
| `enable_judgment_support` | bool | true | LLM判断支援の有効化 |

**revision_areas:**
改定番号ごとに `areas`（検索対象DBエリア）、`search_type`、`vector_weight` を指定。

**area_to_bot / area_to_category:**
エリア名からボット名・日本語カテゴリ名への変換マッピング。

> **重要:** settings.yaml は起動時に必須です。`import config` 時に以下のチェックが実行されます:
> - `common` セクションが空（ファイル未存在 / 空ファイル / common 欠落）→ `RuntimeError`
> - `batch` セクションの個別キー（`top_k`, `vector_weight` 等）が欠落 → `KeyError`
> - 全キーはフォールバックなしの直接アクセスのため、キー追加忘れは即座にエラーとなります

**設定の読み込み方法**:
- `load_settings("ui")` → common + ui をマージして返す
- `load_settings("batch")` → common + batch をマージして返す
- セクション固有の値が common を上書き

### search_source（検索対象）

検索対象データの種別を指定する。

| 値 | 説明 |
|----|------|
| `history_data` | 問い合わせ履歴データ（FAQ）を検索対象とする（デフォルト） |
| `scenario` | シナリオデータを検索対象とする |

**設定場所:** `config/settings.yaml` の `common.search_source`

**UI vs バッチでの挙動:**
- **UI**: サイドバーで動的に切替可能（`settings.yaml` の値は初期値として使用）
- **バッチ**: `settings.yaml` の値を使用（CLI 引数での変更不可）

> **Note:** 運用保守効率化AI UI（`ops_ui.py`）の影響調査モードでも、`source_filter` としてシナリオ / 問い合わせ履歴データ（FAQ）のデータソース切替が可能です。

### keyword設定（キーワード検索パラメータ）

`config/settings.yaml` の `common.keyword` でキーワード検索の動作を制御する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `position_weight` | float | 1.2 | テキスト前半に出現するキーワードの重み係数 |
| `stop_words` | list | 13語 | 除外する一般的な単語のリスト |

**動作概要:** 入力テキスト → 日本語形態素解析ツール（Sudachi）で単語に分割 → 名詞を抽出 → stop_words を除外 → 出現頻度の高い上位5語を選出 → 共通キーワードの割合（Jaccard類似度）で検索結果をスコアリング

### columns設定（Excel列名候補）

`config/settings.yaml` の `common.columns` でバッチ入力 Excel の列名自動検出候補を定義する。各キーに対して候補リストの先頭から順にマッチを試みる。

| キー | 説明 | 必須 | 未検出時の動作 |
|------|------|------|---------------|
| `query` | 質問列 | 必須 | `ValueError` で停止 |
| `answer` | 回答列 | 必須 | `ValueError` で停止 |
| `tag` | タグ・分類列 | 任意 | 警告を出力して続行 |
| `correct_id` | 正解ID列 | 任意 | スキップ（精度評価なし） |

**カスタマイズ例:** 独自の列名を使用する場合、候補リストの先頭に追加する。

```yaml
common:
  columns:
    query:
      - 問い合わせテキスト    # 独自列名を先頭に追加
      - 分割後質問
      - 問合せ内容
      # ...
```

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
├── app.log          # メインログ（全レベル統合出力）
└── archive/         # アーカイブ
```

---

## 設定検証

### 起動時チェック

**モジュールロード時（`import config`）:**
- `config/settings.yaml` を読み込み、`common` セクションが空なら `RuntimeError`
- `_batch_settings` も読み込むが、個別キー（`top_k` 等）が欠落している場合は `KeyError`

**`SearchConfig.__post_init__()` で自動検証:**

| パラメータ | 範囲 | エラー |
|-----------|------|--------|
| `vector_weight` | 0.0 〜 1.0 | `ValueError` |
| `top_k` | 1 以上の整数（>100 で警告） | `ValueError` |
| `multi_stage_threshold` | 0.0 〜 1.0 | `ValueError` |
| `multi_stage_max_results` | 1 以上の整数（>1000 で警告） | `ValueError` |
| `EMBEDDING_BATCH_SIZE` | 1 〜 250 | `ValueError` |
| `VECTOR_DB_BATCH_SIZE` | 1 〜 1000 | `ValueError` |
| `VECTOR_SEARCH_MULTIPLIER` | 1 以上 | `ValueError` |
| `search_type` | `hybrid` / `keyword_filter` | `ValueError` |
| `search_mode` | `original` / `llm_enhanced` / `multi_stage` | `ValueError` |
| `search_source` | `scenario` / `history_data` | `ValueError` |
| `embedding_provider` | `vertex_ai` / `azure_openai`（必須） | `ValueError` |
| `llm_provider` | `gemini`（必須・唯一） | `ValueError` |
| `credential_source` | `local` / `key_vault` | `ValueError` |

埋め込みモデルはプロバイダーに応じて自動決定:
- `azure_openai` → `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`（デフォルト: `text-embedding-3-large`）
- `vertex_ai` → `VERTEX_AI_EMBEDDING_MODEL`（デフォルト: `gemini-embedding-001`）

### 手動検証

```bash
# .env ファイルの読み込み確認
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('DEFAULT_LLM_PROVIDER'))"

# 設定オブジェクトの確認
python -c "from config import SearchConfig; config = SearchConfig(); print(config)"
```

---

## トラブルシューティング

設定に関する問題は [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) を参照。
