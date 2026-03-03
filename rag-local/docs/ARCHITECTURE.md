# システムアーキテクチャ

このドキュメントでは、RAG-Localシステムのアーキテクチャとモジュール間の依存関係を説明します。

## 目次

- [全体構成](#全体構成)
- [レイヤー構造](#レイヤー構造)
- [コアモジュール](#コアモジュール)
- [データフロー](#データフロー)
- [依存関係](#依存関係)
- [拡張性](#拡張性)

---

## 全体構成

```
┌────────────────────────────────────────────────────────────┐
│                      Entry Points                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   main.py    │  │  ui/chat.py  │  │   scripts/   │    │
│  │  (Batch)     │  │ (Streamlit)  │  │ (Evaluation) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                      Core Layer                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ processor.py - データ処理エンジン                    │ │
│  │ judgment_support.py - LLM判断支援                    │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ search/ - 多段階検索サブシステム                     │ │
│  │   ├─ search_strategy.py - 検索戦略切替（4戦略）    │ │
│  │   ├─ multi_stage_orchestrator.py - オーケストレータ │ │
│  │   ├─ query_enhancer.py - クエリ拡張                 │ │
│  │   ├─ vector_search_engine.py - ベクトル検索         │ │
│  │   ├─ keyword_search_engine.py - キーワード検索      │ │
│  │   ├─ chromadb_keyword_search.py - ChromaDBキーワード│ │
│  │   └─ text_combiner.py - テキスト結合                │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    Handler Layer                           │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ input_handler.py │  │ output_handler.py│               │
│  │ (Excel読み込み)  │  │ (Excel出力)      │               │
│  └──────────────────┘  └──────────────────┘               │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     Utils Layer                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │ データベース管理                                    │   │
│  │   ├─ dynamic_db_manager.py - 動的DB管理           │   │
│  │   ├─ vector_db.py - ChromaDB ラッパー             │   │
│  │   └─ business_area_translator.py - 業務領域変換   │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 埋め込みモデル                                      │   │
│  │   ├─ base_embedding.py - 抽象基底クラス           │   │
│  │   ├─ gemini_embedding.py - VertexAI埋め込み       │   │
│  │   └─ azure_embedding.py - Azure OpenAI埋め込み    │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ その他                                              │   │
│  │   ├─ auth.py - Google Cloud認証                   │   │
│  │   └─ logger.py - ログ設定                         │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                  External Services                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Vertex   │  │  Azure   │  │  Azure   │               │
│  │   AI     │  │  OpenAI  │  │Key Vault │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    Data Storage                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ ChromaDB   │  │  Excel     │  │   Logs     │          │
│  │(data/      │  │(data/      │  │  (logs/)   │          │
│  │ vector_db/)│  │ input/out) │  │            │          │
│  └────────────┘  └────────────┘  └────────────┘          │
└────────────────────────────────────────────────────────────┘
```

---

## レイヤー構造

### 1. Entry Points Layer

| モジュール | 説明 | 用途 |
|-----------|------|------|
| `main.py` | バッチ処理エントリーポイント | 一括検索実行 |
| `ui/chat.py` | Streamlit UI | インタラクティブ検索 |
| `scripts/*` | ユーティリティスクリプト | DB再構築、評価実行 |

### 2. Core Layer

#### processor.py - データ処理エンジン

```python
class Processor:
    """データ処理の統合管理"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.searcher = Searcher(config)

    def process_data(self, mode: str = "batch", limit: int = None):
        """データ処理のメイン関数"""
        # 入力データ読み込み → 参照データ読み込み → 検索準備 →
        # 各質問の検索実行 → 結果出力
```

**責務:**
- 入力データの読み込みと検証
- Searcher + SearchStrategy による検索実行の管理
- 結果の集約と整形（通常モード / 多段階検索モード）

#### searcher.py - 検索統合管理

```python
class Searcher:
    """Processor と SearchStrategy を仲介する統合検索クラス"""

    def __init__(self, config, db_manager=None, embedding_model=None):
        # 依存性注入対応（テスト時にモック可能）

    def prepare_search(self, reference_data):
        """参照データの読み込みとキーワードキャッシュ構築"""

    def search(self, input_number, query_text, original_answer, input_file=None):
        """SearchStrategy に委譲して検索実行"""
```

**責務:**
- 動的ベクトルDB選択（業務分野・入力ファイルに基づく）
- 埋め込みモデル・LLMの初期化管理
- キーワード抽出・キャッシング（N+1問題解消）
- ベクトル検索実行とスコア計算
- SearchStrategy への検索委譲

#### search/multi_stage_orchestrator.py - 多段階検索

```python
class MultiStageOrchestrator:
    """多段階ハイブリッド検索のオーケストレーター"""

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        keyword_engine: KeywordSearchEngine,
        query_enhancer: QueryEnhancer,
        text_combiner: TextCombiner,
        vector_weight: float = 0.9,
        threshold: float = 0.45,
        max_results: int = 100,
        filter_mode: str = "threshold",
        top_k: int = 50
    ):
        ...

    def execute(
        self,
        input_number: str,
        query_text: str,
        original_answer: str,
        filter_metadata: Dict[str, str] = None,
        pre_enhanced_query: Optional[str] = None,
    ) -> List[MultiStageSearchResultDict]:
        """多段階OR検索を実行"""
        # Stage 1: 原文でハイブリッド検索
        # Stage 2: LLMクエリでハイブリッド検索
        # Stage 3: OR結合と3分類（Both / Original_Only / LLM_Enhanced_Only）
```

**責務:**
- 検索ステージの管理
- ベクトル・キーワード検索の統合
- スコア計算とランキング

#### judgment_support.py - LLM判断支援

```python
class JudgmentSupport:
    """検索結果の関連性判定"""

    def evaluate(self, revision_content: str, search_result_q: str, search_result_a: str) -> Dict[str, str]:
        """関連性を判定"""
        # LLMで判定し {"relevance_judgment": "...", "judgment_reason": "..."} を返す
```

**責務:**
- 検索結果の関連性判定
- 判定根拠の生成

### 3. Handler Layer

#### input_handler.py - 入力処理

```python
class InputHandler:
    """入力処理の基底クラス"""

    def load_data(self) -> list:
        """入力データを読み込み、共通の形式に変換"""

    def load_reference_data(self) -> dict:
        """参照データを読み込み、共通の形式に変換"""
```

**責務:**
- Excelファイル読み込み
- データ形式検証
- 階層構造解析

#### output_handler.py - 出力処理

```python
class OutputHandler:
    """出力処理の基底クラス"""

    def save_data(self, data: list):
        """データを保存"""
```

**責務:**
- Excel出力
- フォーマット調整
- ファイル命名

### 4. Utils Layer

#### データベース管理

**dynamic_db_manager.py** - 動的DB管理

```python
class DynamicDBManager:
    """業務領域別のベクトルDB管理"""

    def __init__(self, config: SearchConfig):
        self.translator = BusinessAreaTranslator()

    def update_business_db(self, business_area: str, files: Dict):
        """業務分野のDBを更新（タイムスタンプ検証、差分ベクトル化）"""

    def get_db_path_for_business(self, business_area: str) -> str:
        """業務分野に対応するDBパスを返す"""
```

**vector_db.py** - ChromaDB ラッパー

```python
class MetadataVectorDB:
    """メタデータ対応のベクトルデータベースクラス"""

    def add_documents(self, texts: List[str], embeddings, metadatas: List[Dict],
                      ids: Optional[List[str]] = None):
        """ドキュメントとメタデータをベクトルDBに追加"""

    def search(self, query_embedding: List[float], n_results: int = 10,
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """メタデータフィルタリング付きベクトル検索"""
```

#### 埋め込みモデル

**base_embedding.py** - 抽象基底クラス

```python
class BaseEmbeddingModel(ABC):
    """埋め込みモデルの共通インターフェース"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]],
               normalize_embeddings: bool = True) -> np.ndarray:
        """テキストをベクトル化（2次元配列を返す）"""

    def encode_single(self, text: str,
                      normalize_embeddings: bool = True) -> np.ndarray:
        """単一テキストをベクトル化（1次元配列を返す）"""
```

**gemini_embedding.py** / **azure_embedding.py**

各プロバイダー固有の実装:
- API呼び出し
- バッチ処理（`EMBEDDING_BATCH_SIZE = 250`）
- エラーハンドリング
- リトライロジック

---

## データフロー

### 通常バッチ処理フロー（回答支援AI）

```
1. 入力ファイル読み込み
   ↓ input_handler.py (load_data)
2. 参照データ読み込み + ベクトル化
   ↓ dynamic_db_manager.py → embedding → vector_db.py
3. 各行を検索実行
   ↓ processor.py → searcher.py → search_strategy.py
   ├─ OriginalSearchStrategy: 原文ハイブリッド検索（search_mode=original）
   │   ├─ vector_search_engine.py（ベクトル検索）
   │   └─ keyword_search_engine.py（キーワード類似度計算）
   └─ LLMEnhancedSearchStrategy: LLM拡張検索（search_mode=llm_enhanced）
       ├─ searcher.summarize_text()（LLMクエリ生成）
       ├─ vector_search_engine.py
       └─ keyword_search_engine.py
4. 結果出力
   ↓ output_handler.py (save_data)
5. Excelファイル保存
```

### 多段階検索バッチフロー（改定影響調査）

```
1. 入力ファイル読み込み
   ↓ input_handler.py (load_data)
2. 参照データ読み込み + ベクトル化
   ↓ dynamic_db_manager.py → embedding → vector_db.py
3. 各行を検索実行（search_mode=multi_stage）
   ↓ multi_stage_orchestrator.py (execute)
   ├─ Stage 1: 原文でハイブリッド検索
   │   ├─ vector_search_engine.py
   │   └─ keyword_search_engine.py
   ├─ Stage 2: LLM強化クエリで検索
   │   ├─ query_enhancer.py
   │   ├─ vector_search_engine.py
   │   └─ keyword_search_engine.py
   └─ Stage 3: OR結合と3分類（Both / Original_Only / LLM_Enhanced_Only）
4. LLM判断支援（オプション、ThreadPoolExecutor で並列実行）
   ↓ judgment_support.py
5. 結果出力（3シート形式）
   ↓ output_handler.py (save_data_multi_stage)
6. Excelファイル保存
```

### 改定影響調査フロー

```
1. 変更前シナリオExcel配置
   └─ data/source/scenarios/revXXボット_シナリオデータ_YYYYMMDD.xlsx
2. DB再構築
   ↓ scripts/build_db.py --revisions-only
   ├─ Azure OpenAI でベクトル化
   │   └─ data/vector_db/revXX_{bot}/azure_openai/
   └─ VertexAI でベクトル化
       └─ data/vector_db/revXX_{bot}/vertex_ai/
3. 評価実行
   ↓ apps/revision-ops/run_eval.py
   ├─ 各改定内容をクエリとして検索
   ├─ 正解IDとの照合
   └─ Azure / VertexAI 横並び比較
4. Excel出力
   └─ data/output/latest/rev/rev_eval_batch_YYYYMMDD_HHMMSS.xlsx
```

---

## 依存関係

### モジュール間依存

```
main.py
  ├─ config.py (SearchConfig)
  ├─ src/core/processor.py
  │   ├─ src/handlers/input_handler.py
  │   ├─ src/handlers/output_handler.py
  │   ├─ src/core/search/search_strategy.py (4戦略切替)
  │   ├─ src/core/search/multi_stage_orchestrator.py
  │   │   ├─ src/core/search/query_enhancer.py
  │   │   ├─ src/core/search/vector_search_engine.py
  │   │   │   └─ src/utils/vector_db.py
  │   │   ├─ src/core/search/keyword_search_engine.py
  │   │   ├─ src/core/search/chromadb_keyword_search.py
  │   │   └─ src/core/search/text_combiner.py
  │   ├─ src/core/judgment_support.py
  │   └─ src/utils/dynamic_db_manager.py
  │       ├─ src/utils/vector_db.py
  │       ├─ src/utils/base_embedding.py
  │       │   ├─ src/utils/gemini_embedding.py
  │       │   │   └─ src/utils/auth.py
  │       │   └─ src/utils/azure_embedding.py
  │       └─ src/utils/business_area_translator.py
  └─ src/utils/logger.py
```

### 外部ライブラリ依存

| カテゴリ | ライブラリ | 用途 |
|---------|-----------|------|
| **Google Cloud** | google-genai | Vertex AI Embedding API |
| | google-auth | Google Cloud 認証 |
| **Azure** | openai | Azure OpenAI Embedding API |
| | azure-identity | Azure 認証 |
| | azure-keyvault-secrets | Key Vault 認証（オプション） |
| **ベクトルDB** | chromadb | ベクトルストレージ |
| **LangChain** | langchain-google-genai | Gemini LLM API |
| **NLP** | sudachipy | 日本語形態素解析 |
| **データ処理** | pandas | データフレーム操作 |
| | openpyxl | Excel読み書き |
| **UI** | streamlit | Web UI |

---

## 拡張性

### 新しい埋め込みモデルの追加

1. `src/utils/base_embedding.py` を継承
2. `encode()`, `embedding_dimension`, `provider_name` を実装
3. `src/utils/auth.py` の `create_embedding_model()` にプロバイダー分岐を追加
4. `config.py` の `VALID_EMBEDDING_PROVIDERS` タプルにプロバイダー名を追加

```python
# src/utils/custom_embedding.py
from src.utils.base_embedding import BaseEmbeddingModel

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, config: SearchConfig):
        super().__init__(config)

    def encode(self, texts: Union[str, List[str]],
               normalize_embeddings: bool = True) -> np.ndarray:
        # 実装

    @property
    def embedding_dimension(self) -> int:
        return 1024  # 例

    @property
    def provider_name(self) -> str:
        return "custom"
```

```python
# config.py
VALID_EMBEDDING_PROVIDERS: Tuple[str, ...] = ("vertex_ai", "azure_openai", "custom")  # 追加
```

```python
# src/utils/auth.py の create_embedding_model() に分岐追加
def create_embedding_model(config, use_singleton=True):
    ...
    elif provider == "custom":
        from src.utils.custom_embedding import CustomEmbeddingModel
        return CustomEmbeddingModel(config)
```

### 新しい業務分野の追加

例: 「為替」（kawase）を追加する場合

#### Step 1: マッピング登録

`config/business_areas.yaml` の `mappings` に追加:

```yaml
mappings:
  スマイル: smile
  内部事務: naibujimu
  為替: kawase          # 追加
```

ChromaDB コレクション名の制約: 英数字・`.`・`_`・`-` のみ、3-512文字。

#### Step 2: 参照データの配置

ファイル命名規則: `{日本語業務名}_{データ種別}_{YYYYMMDD}.xlsx`

```
data/source/
├── faq/latest/
│   └── 為替_履歴データ_20260601.xlsx        # FAQ（任意）
└── scenarios/latest/
    └── 為替_シナリオデータ_20260601.xlsx    # シナリオ（任意）
```

- FAQ・シナリオの片方だけでも可。両方なくてもDB構築自体は通るがデータが空になる
- 同一業務名で複数ファイルがある場合、最新日付が優先される
- ファイル名の日本語部分が `business_areas.yaml` のキーと一致する必要がある

#### Step 3: DB構築

```bash
# Streamlit UI を停止してから実行（ChromaDB ファイルロック競合防止）
python scripts/build_db.py --business kawase
```

確認:
```bash
python scripts/check_db_content.py  # ドキュメント数が 0 でないことを確認
```

#### Step 4: 入力ファイルの作成

バッチ処理用の入力ファイル命名規則: `{英語DB名}_{YYYYMMDD}.xlsx`

```
data/input/
└── kawase_20260601.xlsx    # 列: 番号, 質問内容, 既存回答
```

**注意**: 入力ファイル名は**英語DB名**（`kawase`）を使用する。参照データは日本語名（`為替`）、入力ファイルは英語名という非対称に注意。

#### Step 5: 動作確認

```bash
python apps/answer-support/main.py --business kawase --limit 3
```

#### コード変更が不要な理由

- `analyze_reference_files()` が `data/source/` を走査してファイル名から業務分野を自動検出する
- `BusinessAreaTranslator` が YAML マッピングで日本語→英語を変換する
- `extract_business_area_from_input()` が入力ファイル名から業務分野を抽出する
- DB パスは `data/vector_db/{英語名}/{provider}/` に自動生成される

つまり、**設定ファイル + データファイルの配置だけで完結し、コード変更は不要**。

### 新しい検索エンジンの追加

```python
# src/core/search/custom_search_engine.py
class CustomSearchEngine:
    def search(self, query: str, documents: List[Dict]) -> List[SearchResult]:
        # カスタム検索ロジック
        pass
```

MultiStageOrchestrator は個別引数でエンジンを受け取るため、呼び出し元で注入する:

```python
# 呼び出し元で MultiStageOrchestrator を構築する際にカスタムエンジンを渡す
orchestrator = MultiStageOrchestrator(
    vector_engine=vector_engine,
    keyword_engine=keyword_engine,
    query_enhancer=query_enhancer,
    text_combiner=text_combiner,
    vector_weight=0.9
)
```

### 認証方式

LLM は Gemini（Vertex AI）のみをサポート。認証は `CREDENTIAL_SOURCE` 環境変数で切替:

| 値 | 認証方式 | 用途 |
|----|---------|------|
| `local`（デフォルト） | ローカルのサービスアカウント JSON ファイル | 開発環境 |
| `key_vault` | Azure Key Vault からサービスアカウント JSON を取得 | 本番環境 |

```python
# src/utils/auth.py - get_google_credentials() の認証ディスパッチ
_CREDENTIAL_HANDLERS = {
    "local": _get_credentials_local,        # from_service_account_file()
    "key_vault": _get_credentials_key_vault, # SecretClient → from_service_account_info()
}
```

---

## パフォーマンス最適化

### キャッシング戦略

1. **ベクトル化結果のキャッシュ**
   - タイムスタンプベースの検証
   - ChromaDB永続化

2. **LLMレスポンスのキャッシュ**（将来実装予定）
   - クエリハッシュベースのキャッシュ
   - TTL: 24時間

### 並列処理

1. **バッチベクトル化**
   - バッチサイズ: `EMBEDDING_BATCH_SIZE = 250`（Vertex AI / Azure 共通、`config.py`）
   - 並列API呼び出し

2. **LLM判断支援の並列化**（実装済み）
   - `processor.py`: `ThreadPoolExecutor(max_workers=10)` でLLM評価を並列実行

3. **検索の並列化**（実装済み）
   - `ops_ui.py`: `ThreadPoolExecutor(max_workers=2)` でAzure / VertexAI検索を並列実行
   - `run_eval.py`: `ThreadPoolExecutor(max_workers=5)` で複数エリアの検索を並列実行

---

## セキュリティ考慮事項

### 認証フロー

```
1. 環境変数読み込み (.env)
   ↓
2. 認証情報取得
   ├─ Google Cloud: サービスアカウントキー
   └─ Azure: API キー
   ↓
3. API クライアント初期化
   ├─ auth.py (Google Cloud)
   └─ azure_embedding.py (Azure)
   ↓
4. API呼び出し
```

---

## テスト

`pytest` で実行（`tests/`, `pytest.ini`, `requirements-dev.txt`）。

### ユニットテスト

```
tests/unit/
├── test_business_area_mapping.py    # 業務分野マッピング・DB選択
├── test_chromadb_keyword_search.py  # ChromaDBキーワード検索
├── test_keyword_search_engine.py    # キーワード抽出・類似度計算
├── test_keyword_similarity_sync.py  # Searcher/Orchestrator間のキーワード同期
├── test_logger_dashboard.py         # ダッシュボードログ出力
├── test_logger_noise.py             # サードパーティログ抑制
├── test_run_eval_cache.py           # 評価キャッシュ動作
├── test_search_strategy.py          # 検索戦略パターン（4戦略）
├── test_text_combiner.py            # テキスト結合・パース
├── test_timestamp_migration.py      # タイムスタンプ移行
└── test_ui_shared.py               # UI共通ユーティリティ
```

---

## API リファレンス

このドキュメントでは、RAG-Localシステムの主要なPythonモジュールのAPI仕様を説明します。

### 目次

- [設定](#設定)
- [コアモジュール](#コアモジュール)
- [検索エンジン](#検索エンジン)
- [データベース管理](#データベース管理)
- [埋め込みモデル](#埋め込みモデル)
- [ハンドラー](#ハンドラー)
- [ユーティリティ](#ユーティリティ)

---

### 設定

#### load_settings()

**モジュール:** `config.py`

`config/settings.yaml` を読み込み、セクション別の設定辞書を返すユーティリティ関数。

```python
def load_settings(section: Optional[str] = None) -> Dict[str, Any]:
    """settings.yamlを読み込み、指定セクションの設定を返す

    Args:
        section: 読み込むセクション名 ("ui", "batch", "evaluation")
                 Noneの場合は全設定を返す

    Returns:
        commonセクションと指定セクションをマージした辞書
        sectionがNoneの場合は全設定辞書
    """
```

#### SearchConfig

**モジュール:** `config.py`

検索システムの設定を管理するデータクラス。デフォルト値は `config/settings.yaml` と環境変数から読み込まれる。

```python
@dataclass
class SearchConfig:
    """検索設定を管理するデータクラス"""

    # LLM設定（環境変数から読み込み）
    llm_provider: str   # DEFAULT_LLM_PROVIDER 環境変数（必須）
    llm_model: str      # DEFAULT_LLM_MODEL 環境変数（必須）

    # 埋め込みモデル設定
    embedding_provider: str  # DEFAULT_EMBEDDING_PROVIDER（必須、vertex_ai / azure_openai）
    embedding_model: str     # __post_init__ でプロバイダーから自動解決

    # 検索設定（settings.yaml batch セクションから読み込み、フォールバックなし）
    top_k: int = 4                    # 返却する結果数
    vector_weight: float = 0.9        # ベクトル検索の重み（0〜1）※ batch セクション固有値
    keyword_weight: float             # 自動計算（1.0 - vector_weight）

    # 検索モード: original | llm_enhanced | multi_stage
    # 回答支援AIでは original / llm_enhanced のみ使用。
    # multi_stage は改定影響調査（apps/revision-ops/）専用。
    search_mode: str = "original"

    # 入出力設定
    input_type: str = "excel"
    output_type: str = "excel"
    reference_type: str = "multi_folder"
```

##### 主要パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `credential_source` | str | `local` | GCP認証方式（local/key_vault） |
| `llm_provider` | str | 環境変数 | LLMプロバイダー（gemini のみ） |
| `llm_model` | str | 環境変数 | LLMモデル名 |
| `embedding_provider` | str | 環境変数 | 埋め込みプロバイダー（vertex_ai/azure_openai） |
| `embedding_model` | str | 環境変数 | 埋め込みモデル名 |
| `top_k` | int | settings.yaml | 返却する結果数 |
| `vector_weight` | float | settings.yaml | ベクトル検索の重み（0〜1） |
| `keyword_weight` | float | 自動計算 | キーワード検索の重み（1.0 - vector_weight） |
| `search_mode` | str | settings.yaml | 検索モード（original/llm_enhanced/multi_stage）。multi_stage は改定影響調査（`apps/revision-ops/`）専用 |
| `search_type` | str | settings.yaml | 検索タイプ（hybrid/keyword_filter） |
| `search_source` | str | settings.yaml | 検索対象（scenario/history_data） |
| `reference_type` | str | "multi_folder" | 参照データ形式（excel/hierarchical_excel/multi_folder） |
| `force_db_update` | bool | False | 強制DB更新フラグ |
| `dual_provider_mode` | bool | False | 両プロバイダー比較モード |

##### 使用例

```python
from config import SearchConfig, load_settings

# settings.yaml のセクション読み込み
ui_settings = load_settings("ui")

# SearchConfig は環境変数が必須（.env ファイルで設定）
config = SearchConfig(
    top_k=10,
    vector_weight=0.7,
    search_mode="llm_enhanced"
)
```

---

### コアモジュール

#### Processor

**モジュール:** `src/core/processor.py`

データ処理の統合管理クラス。InputHandlerFactory / OutputHandlerFactory で入出力を切り替える。

```python
class Processor:
    """データ処理エンジン"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """
```

##### メソッド

###### process_data

```python
def process_data(self, mode: str = "batch", limit: int = None):
    """
    データ処理のメイン関数

    Args:
        mode: 処理モード（"batch" 等）
        limit: 処理する入力データの件数上限（Noneで全件）

    Note:
        内部で入力データ読み込み → 参照データ読み込み → 検索準備 →
        各質問の検索実行 → 結果出力 を一括で行う。
        search_mode が "multi_stage" の場合は LLM 判断支援付き3シート出力。

    Raises:
        Exception: 処理中のエラー
    """
```

##### 使用例

```python
from config import SearchConfig
from src.core.processor import Processor

config = SearchConfig()
processor = Processor(config)

# バッチ処理実行
processor.process_data(mode="batch")

# 件数制限付き
processor.process_data(mode="batch", limit=10)
```

#### Searcher

**モジュール:** `src/core/searcher.py`

Processor と SearchStrategy を仲介する統合検索クラス。動的DB選択、キーワードキャッシング、検索委譲を管理する。

```python
class Searcher:
    """メタデータ対応ハイブリッド検索クラス"""

    def __init__(
        self,
        config: SearchConfig,
        db_manager: Optional[DynamicDBManager] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None
    ):
        """
        Args:
            config: 検索設定
            db_manager: 動的DB管理（省略時は自動生成）
            embedding_model: 埋め込みモデル（省略時は自動生成）
        """
```

##### メソッド

###### prepare_search

```python
def prepare_search(self, reference_data: dict):
    """
    参照データの読み込みとベクトル化、キーワードキャッシュ構築

    Args:
        reference_data: 業務分野別の参照データ辞書
    """
```

###### search

```python
def search(
    self,
    input_number: str,
    query_text: str,
    original_answer: str,
    input_file: str = None
) -> list:
    """
    SearchStrategy に委譲して検索実行

    Args:
        input_number: 入力番号
        query_text: 検索クエリテキスト
        original_answer: 元の回答
        input_file: 入力ファイル名（動的DB選択用）

    Returns:
        検索結果のリスト
    """
```

#### JudgmentSupport

**モジュール:** `src/core/judgment_support.py`

LLMによる改定内容と検索結果の関連性判定（人間の意思決定を支援）。

```python
class JudgmentSupport:
    """LLMを使用して人間の判断を支援するクラス"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """
```

##### メソッド

###### evaluate

```python
def evaluate(
    self,
    revision_content: str,
    search_result_q: str,
    search_result_a: str
) -> Dict[str, str]:
    """
    単一の検索結果に対する関連性評価を実行

    Args:
        revision_content: 改定内容テキスト
        search_result_q: 検索結果の質問
        search_result_a: 検索結果の回答

    Returns:
        {
            "relevance_judgment": "関連あり" | "要確認" | "明らかに無関係" | "エラー",
            "judgment_reason": "判定根拠"
        }
    """
```

##### 使用例

```python
from src.core.judgment_support import JudgmentSupport

support = JudgmentSupport(config)

result = support.evaluate(
    revision_content="相続人確認方法を追加する",
    search_result_q="相続預金の手続きについて",
    search_result_a="本人確認書類をご持参の上..."
)

print(result["relevance_judgment"])  # "関連あり"
print(result["judgment_reason"])     # "改定内容と既存QAが同一業務を対象としている"
```

---

### 検索エンジン

#### MultiStageOrchestrator

**モジュール:** `src/core/search/multi_stage_orchestrator.py`

多段階ハイブリッド検索のオーケストレーター。原文検索 + LLMクエリ検索のOR結合を管理。

```python
class MultiStageOrchestrator:
    """多段階検索オーケストレーター"""

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        keyword_engine: KeywordSearchEngine,
        query_enhancer: QueryEnhancer,
        text_combiner: TextCombiner,
        vector_weight: float = 0.9,
        threshold: float = 0.45,
        max_results: int = 100,
        filter_mode: str = "threshold",
        top_k: int = 50
    ):
        """
        Args:
            vector_engine: ベクトル検索エンジン
            keyword_engine: キーワード検索エンジン
            query_enhancer: クエリ拡張エンジン
            text_combiner: テキスト結合ユーティリティ
            vector_weight: ベクトルスコアの重み
            threshold: 結果に含めるスコアしきい値
            max_results: 各検索の最大結果数
            filter_mode: フィルタリングモード ('threshold' or 'top_k')
            top_k: TOP-K件数（filter_mode='top_k'の場合に使用）
        """
```

#### QueryEnhancer

**モジュール:** `src/core/search/query_enhancer.py`

LLMによるクエリ拡張。

```python
class QueryEnhancer:
    """クエリ拡張エンジン"""

    def __init__(self, llm, base_dir: str = "."):
        """
        Args:
            llm: LangChain LLMインスタンス
            base_dir: プロンプトファイルの基準ディレクトリ
        """
```

##### メソッド

###### enhance

```python
def enhance(self, query: str) -> str:
    """
    クエリを拡張

    Args:
        query: 元のクエリ

    Returns:
        拡張されたクエリ（キーワード列挙形式）

    Example:
        >>> enhancer.enhance("口座開設について")
        "検索クエリ: 銀行口座 新規開設 必要書類 手続き 流れ"
    """
```

---

### データベース管理

#### DynamicDBManager

**モジュール:** `src/utils/dynamic_db_manager.py`

業務領域別のベクトルDB管理。タイムスタンプ検証による差分更新を行う。

```python
class DynamicDBManager:
    """動的DB管理システム"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
                    （config.base_dir から data/vector_db パスを生成）
        """
```

#### MetadataVectorDB

**モジュール:** `src/utils/vector_db.py`

ChromaDBの操作ラッパー。メタデータ対応のベクトルデータベースクラス。

```python
class MetadataVectorDB:
    """メタデータ対応のベクトルデータベースクラス"""

    def __init__(
        self,
        base_dir: str = ".",
        collection_name: str = None,
        batch_size: int = 100,
        db_path: str = None
    ):
        """
        Args:
            base_dir: ベースディレクトリ
            collection_name: コレクション名
            batch_size: バッチサイズ
            db_path: DB パス（直接指定時、base_dir より優先）
        """
```

---

### 埋め込みモデル

#### BaseEmbeddingModel

**モジュール:** `src/utils/base_embedding.py`

埋め込みモデルの抽象基底クラス。

```python
class BaseEmbeddingModel(ABC):
    """埋め込みモデル基底クラス"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: SearchConfig インスタンス
        """

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        テキストをベクトル化

        Args:
            texts: 単一テキストまたはテキストのリスト
            normalize_embeddings: ベクトルを正規化するかどうか

        Returns:
            numpy.ndarray: 埋め込みベクトル（2次元配列）
        """
        pass

    def encode_single(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """
        単一テキストをベクトル化

        Returns:
            numpy.ndarray: 埋め込みベクトル（1次元配列）
        """

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す"""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """プロバイダー名を返す（例: "vertex_ai", "azure_openai"）"""
```

#### GeminiEmbeddingModel

**モジュール:** `src/utils/gemini_embedding.py`

VertexAI Gemini埋め込みモデル。

```python
class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Gemini埋め込みモデル"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定

        Raises:
            ValueError: 認証エラー
        """
```

#### AzureEmbeddingModel

**モジュール:** `src/utils/azure_embedding.py`

Azure OpenAI埋め込みモデル。

```python
class AzureEmbeddingModel(BaseEmbeddingModel):
    """Azure OpenAI埋め込みモデル"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定

        Raises:
            ValueError: API キーまたはエンドポイントが未設定
        """
```

---

### ハンドラー

#### InputHandler / InputHandlerFactory

**モジュール:** `src/handlers/input_handler.py`

入力ファイルの処理。Factoryパターンで入力形式に応じたハンドラーを生成する。

```python
class InputHandler:
    """入力処理の基底クラス"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """

    def load_data(self) -> list:
        """入力データを読み込み、共通の形式に変換"""
        raise NotImplementedError

    def load_reference_data(self) -> dict:
        """参照データを読み込み、共通の形式に変換"""
        raise NotImplementedError


class InputHandlerFactory:
    """入力ハンドラーのファクトリ"""

    @staticmethod
    def create(input_type: str, config: SearchConfig) -> InputHandler:
        """入力タイプに応じたハンドラーを生成"""
```

#### OutputHandler / OutputHandlerFactory

**モジュール:** `src/handlers/output_handler.py`

出力ファイルの生成。Factoryパターンで出力形式に応じたハンドラーを生成する。

```python
class OutputHandler:
    """出力処理の基底クラス"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """

    def save_data(self, data: list):
        """データを保存"""
        raise NotImplementedError


class OutputHandlerFactory:
    """出力ハンドラーのファクトリ"""

    @staticmethod
    def create(output_type: str, config: SearchConfig) -> OutputHandler:
        """出力タイプに応じたハンドラーを生成"""
```

---

### ユーティリティ

#### BusinessAreaTranslator

**モジュール:** `src/utils/business_area_translator.py`

業務領域の日本語→英語変換。YAMLファイル（`config/business_areas.yaml`）からマッピングを読み込む。

```python
class BusinessAreaTranslator:
    """業務分野名変換クラス"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルのパス（省略時はデフォルトパス）
        """
        self.mappings: Dict[str, str] = {}         # YAMLから読み込み
        self.revision_mappings: Dict[str, str] = {} # 事務改定用マッピング

    def translate(self, business_area: str) -> str:
        """
        業務分野名を英語に変換

        Args:
            business_area: 日本語の業務分野名（例: 内部事務）

        Returns:
            ChromaDB互換の英語名（例: naibujimu）

        Note:
            完全一致 → 部分一致 → 英数字サニタイズのフォールバック順で変換。
            revXXで始まる場合は revision_mappings を優先チェック。
        """
```

現在の主要マッピング（`config/business_areas.yaml`から読み込み）:

| 日本語名 | 英語名 |
|----------|--------|
| 内部事務 | naibujimu |
| スマイル | smile |
| 相続 | souzoku |
| 取引時確認 | torikaku |

#### Logger

**モジュール:** `src/utils/logger.py`

ログ設定。rich ライブラリによる色付き出力に対応。

```python
def setup_logger(name: str) -> logging.Logger:
    """
    ロガーを設定

    Args:
        name: ロガー名（通常は __name__）

    Returns:
        ロガーインスタンス

    Note:
        - ログレベルは LOG_LEVEL 環境変数で制御（デフォルト: INFO）
        - ファイル出力: logs/app.log（詳細フォーマット）
        - コンソール出力: rich 使用時は色付き、未使用時は短縮フォーマット
    """
```

---

### エラーハンドリング

#### DynamicDBError

```python
# src/utils/dynamic_db_manager.py

class DynamicDBError(Exception):
    """動的DB管理のエラー"""
    pass
```

> **Note:** 汎用の `src/exceptions.py` は未実装。各モジュールが個別に例外を定義している。

---

### 使用例：完全なワークフロー

```python
from config import SearchConfig
from src.core.processor import Processor

# 1. 設定（環境変数 .env で LLM/Embedding プロバイダーを設定済み）
config = SearchConfig(
    top_k=4,
    vector_weight=0.9,
    search_mode="llm_enhanced"
)

# 2. Processor が入力読み込み〜検索〜出力を一括実行
processor = Processor(config)
processor.process_data(mode="batch")
```

---

## プロンプト

このディレクトリには、LLMに送信するプロンプトテンプレートが格納されています。

### AI使用箇所マップ

```
入力
  │
  ├─ ベクトル化 ─────────────────────────────────── [Embedding Model]
  │   └─ azure_openai: text-embedding-3-large
  │   └─ vertex_ai: gemini-embedding-001
  │
  ├─ クエリ拡張 ─────────────────────────────────── [LLM]
  │   └─ prompt/summarize_v1.0.txt
  │   └─ 呼び出し元: src/core/search/query_enhancer.py
  │
  ├─ 検索 ───────────────────────────────────────── [ベクトルDB]
  │   └─ ChromaDB
  │
  └─ 関連性判定 ─────────────────────────────────── [LLM]
      └─ prompt/judgment_support.txt
      └─ 呼び出し元: src/core/judgment_support.py
```

### プロンプトファイル詳細

| ファイル | 用途 | 呼び出し元 | AIモデル |
|---------|------|-----------|---------|
| `summarize_v1.0.txt` | 検索クエリ拡張 | `query_enhancer.py` | LLM (gemini-2.5-flash-lite) |
| `judgment_support.txt` | 関連性判定（関連あり/要確認/無関係） | `judgment_support.py` | LLM (gemini-2.5-flash-lite) |

### 各プロンプトの説明

#### summarize_v1.0.txt（クエリ拡張）

ユーザーの質問文を、ベクトル検索に最適化された検索クエリに変換します。

**入力**: ユーザーの質問文
**出力**: 検索クエリ（キーワード列挙形式）

```
検索クエリ: 現金処理 誤操作 WAVE 700 PRO 有高調整 手続き
```

#### judgment_support.txt（関連性判定）

改定内容と検索結果（既存FAQ）の関連性を3段階で判定します。

**入力**: 改定内容、検索結果（質問・回答）
**出力**:
```
関連性: 関連あり / 要確認 / 明らかに無関係
根拠: 判定理由（1-2文）
```

### 設定

使用するLLMモデルは環境変数で設定します：

```
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite
```

詳細は `.env.example` を参照してください。

---

## 規約と注意事項

- **検索モード**: `search_mode` パラメータで制御（`enable_query_enhancement`は廃止済み）
- **検索実行**: `SearchStrategy`パターン（`src/core/search/search_strategy.py`）で4戦略クラスを切替
- **タイムスタンプ**: フラット形式（旧3階層から自動移行対応済み）
- **テスト**: `pytest` で実行（`tests/`, `pytest.ini`, `requirements-dev.txt`）
- **コレクション命名**: `rev{XX}_{bot}` 形式（例: `rev01_smile`, `rev02_souzoku`）

---

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要・セットアップ
- [docs/ANSWER_SUPPORT.md](./ANSWER_SUPPORT.md) - 回答支援AI詳細
- [docs/REVISION_OPS.md](./REVISION_OPS.md) - 改定影響調査詳細
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 設定リファレンス
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
