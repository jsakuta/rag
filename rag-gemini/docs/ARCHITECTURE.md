# システムアーキテクチャ

このドキュメントでは、RAG-Geminiシステムのアーキテクチャとモジュール間の依存関係を説明します。

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
│  │ searcher.py - 統合検索エンジン（レガシー）          │ │
│  │ judgment_support.py - LLM判断支援                    │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ search/ - 多段階検索サブシステム                     │ │
│  │   ├─ multi_stage_orchestrator.py - オーケストレータ │ │
│  │   ├─ query_enhancer.py - クエリ拡張                 │ │
│  │   ├─ vector_search_engine.py - ベクトル検索         │ │
│  │   ├─ keyword_search_engine.py - キーワード検索      │ │
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
│  │   ├─ db_version_manager.py - バージョン管理       │   │
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
│  │   ├─ logger.py - ログ設定                         │   │
│  │   └─ utils.py - ユーティリティ関数                │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                  External Services                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Vertex   │  │  Azure   │  │Anthropic │  │  OpenAI  │ │
│  │   AI     │  │  OpenAI  │  │  Claude  │  │   GPT    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    Data Storage                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ ChromaDB   │  │  Excel     │  │   Logs     │          │
│  │(vector_db/)│  │(input/out) │  │  (logs/)   │          │
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
class DataProcessor:
    """データ処理の統合管理"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.orchestrator = MultiStageOrchestrator(config)

    def process_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """バッチ処理実行"""
        # 入力データ → 検索 → 結果集約
```

**責務:**
- 入力データの検証
- 検索実行の管理
- 結果の集約と整形

#### search/multi_stage_orchestrator.py - 多段階検索

```python
class MultiStageOrchestrator:
    """多段階ハイブリッド検索のオーケストレーター"""

    def __init__(self, config: SearchConfig):
        self.vector_engine = VectorSearchEngine(config)
        self.keyword_engine = KeywordSearchEngine()
        self.query_enhancer = QueryEnhancer(config)

    def search(self, query: str) -> List[SearchResult]:
        """多段階検索実行"""
        # Stage 1: 原文検索
        # Stage 2: LLM強化検索
        # Stage 3: 結果マージ
```

**責務:**
- 検索ステージの管理
- ベクトル・キーワード検索の統合
- スコア計算とランキング

#### judgment_support.py - LLM判断支援

```python
class JudgmentSupport:
    """検索結果の関連性判定"""

    def analyze_relevance(self, query: str, result: SearchResult) -> JudgmentResult:
        """関連性を分析"""
        # LLMで判定: 関連あり / 要確認 / 関連なし
```

**責務:**
- 検索結果の関連性判定
- 判定根拠の生成
- 修正案の提示

### 3. Handler Layer

#### input_handler.py - 入力処理

```python
class InputHandler:
    """Excel入力ファイルの処理"""

    def read_input_file(self, file_path: str) -> pd.DataFrame:
        """入力ファイル読み込み"""

    def read_reference_data(self, folder_path: str) -> List[Dict]:
        """参照データ読み込み"""
```

**責務:**
- Excelファイル読み込み
- データ形式検証
- 階層構造解析

#### output_handler.py - 出力処理

```python
class OutputHandler:
    """Excel出力ファイルの生成"""

    def save_results(self, results: pd.DataFrame, output_path: str):
        """結果を保存"""
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
        self.db_map = {}  # 業務領域 → VectorDB

    def get_or_create_db(self, business_area: str) -> VectorDB:
        """DBの取得または作成"""
        # タイムスタンプ検証
        # 必要に応じて再ベクトル化
```

**vector_db.py** - ChromaDB ラッパー

```python
class VectorDB:
    """ChromaDBの操作ラッパー"""

    def add_documents(self, documents: List[Dict]):
        """ドキュメント追加"""

    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """ベクトル検索"""
```

#### 埋め込みモデル

**base_embedding.py** - 抽象基底クラス

```python
class BaseEmbeddingModel(ABC):
    """埋め込みモデルの共通インターフェース"""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """テキストをベクトル化"""

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """クエリをベクトル化"""
```

**gemini_embedding.py** / **azure_embedding.py**

各プロバイダー固有の実装:
- API呼び出し
- バッチ処理
- エラーハンドリング
- リトライロジック

---

## データフロー

### バッチ処理フロー

```
1. 入力ファイル読み込み
   ↓ input_handler.py
2. 参照データ読み込み + ベクトル化
   ↓ dynamic_db_manager.py → embedding → vector_db.py
3. 各行を検索実行
   ↓ multi_stage_orchestrator.py
   ├─ Stage 1: 原文でハイブリッド検索
   │   ├─ vector_search_engine.py
   │   └─ keyword_search_engine.py
   ├─ Stage 2: LLM強化クエリで検索
   │   ├─ query_enhancer.py
   │   ├─ vector_search_engine.py
   │   └─ keyword_search_engine.py
   └─ Stage 3: 結果マージ + カテゴリ分類
4. LLM判断支援（オプション）
   ↓ judgment_support.py
5. 結果出力
   ↓ output_handler.py
6. Excelファイル保存
```

### 事務改定評価フロー

```
1. 変更前シナリオExcel配置
   └─ reference/scenario/revXXボット_シナリオデータ_YYYYMMDD.xlsx
2. DB再構築
   ↓ scripts/rebuild_before_scenario_db.py
   ├─ Azure OpenAI でベクトル化
   │   └─ reference/vector_db/revXX/azure_openai/
   └─ VertexAI でベクトル化
       └─ reference/vector_db/revXX/vertex_ai/
3. 評価実行
   ↓ scripts/evaluate_revisions.py
   ├─ 各改定内容をクエリとして検索
   ├─ 正解IDとの照合
   └─ Azure / VertexAI 横並び比較
4. Excel出力
   └─ output/revision_evaluation_YYYYMMDD_HHMMSS.xlsx
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
  │   ├─ src/core/search/multi_stage_orchestrator.py
  │   │   ├─ src/core/search/query_enhancer.py
  │   │   ├─ src/core/search/vector_search_engine.py
  │   │   │   └─ src/utils/vector_db.py
  │   │   ├─ src/core/search/keyword_search_engine.py
  │   │   └─ src/core/search/text_combiner.py
  │   ├─ src/core/judgment_support.py
  │   └─ src/utils/dynamic_db_manager.py
  │       ├─ src/utils/vector_db.py
  │       ├─ src/utils/base_embedding.py
  │       │   ├─ src/utils/gemini_embedding.py
  │       │   │   └─ src/utils/auth.py
  │       │   └─ src/utils/azure_embedding.py
  │       ├─ src/utils/db_version_manager.py
  │       └─ src/utils/business_area_translator.py
  └─ src/utils/logger.py
```

### 外部ライブラリ依存

| カテゴリ | ライブラリ | 用途 |
|---------|-----------|------|
| **Google Cloud** | google-cloud-aiplatform | Vertex AI 統合 |
| | google-auth | Google Cloud 認証 |
| | google-generativeai | Gemini API |
| **Azure** | openai | Azure OpenAI API |
| **ベクトルDB** | chromadb | ベクトルストレージ |
| **LangChain** | langchain-anthropic | Claude API |
| | langchain-openai | OpenAI API |
| | langchain-google-genai | Gemini API |
| **NLP** | sudachipy | 日本語形態素解析 |
| **データ処理** | pandas | データフレーム操作 |
| | openpyxl | Excel読み書き |
| **UI** | streamlit | Web UI |

---

## 拡張性

### 新しい埋め込みモデルの追加

1. `src/utils/base_embedding.py` を継承
2. 必要なメソッドを実装
3. `config.py` に設定を追加

```python
# src/utils/custom_embedding.py
from src.utils.base_embedding import BaseEmbeddingModel

class CustomEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, config: SearchConfig):
        # 初期化処理

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # 実装

    def embed_query(self, query: str) -> List[float]:
        # 実装
```

```python
# config.py
EMBEDDING_PROVIDERS = {
    "azure_openai": AzureEmbeddingModel,
    "vertex_ai": GeminiEmbeddingModel,
    "custom": CustomEmbeddingModel,  # 追加
}
```

### 新しい検索エンジンの追加

```python
# src/core/search/custom_search_engine.py
class CustomSearchEngine:
    def search(self, query: str, documents: List[Dict]) -> List[SearchResult]:
        # カスタム検索ロジック
        pass
```

```python
# src/core/search/multi_stage_orchestrator.py
def __init__(self, config: SearchConfig):
    self.vector_engine = VectorSearchEngine(config)
    self.keyword_engine = KeywordSearchEngine()
    self.custom_engine = CustomSearchEngine()  # 追加
```

### 新しいLLMプロバイダーの追加

1. `config.py` の `LLM_PROVIDERS` に追加
2. 必要に応じて `src/core/search/query_enhancer.py` を拡張

```python
# config.py
LLM_PROVIDERS = {
    "gemini": "VertexAI",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "custom": "CustomLLM",  # 追加
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
   - バッチサイズ: 5（Gemini）/ 16（Azure）
   - 並列API呼び出し

2. **検索の並列化**（将来実装予定）
   - ベクトル検索とキーワード検索を並列実行
   - ThreadPoolExecutor 使用

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

詳細は [docs/SECURITY.md](./SECURITY.md) を参照してください。

---

## テスト戦略（将来実装予定）

### ユニットテスト

```
tests/
├── unit/
│   ├── test_embedding.py
│   ├── test_vector_db.py
│   └── test_search_engine.py
├── integration/
│   ├── test_processor.py
│   └── test_orchestrator.py
└── e2e/
    └── test_batch_processing.py
```

### カバレッジ目標

- ユニットテスト: 80%
- 統合テスト: 60%
- E2Eテスト: 主要フロー

---

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要
- [docs/API_REFERENCE.md](./API_REFERENCE.md) - API仕様
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 設定詳細
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
