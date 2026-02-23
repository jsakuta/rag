# API リファレンス

このドキュメントでは、RAG-Localシステムの主要なPythonモジュールのAPI仕様を説明します。

## 目次

- [設定](#設定)
- [コアモジュール](#コアモジュール)
- [検索エンジン](#検索エンジン)
- [データベース管理](#データベース管理)
- [埋め込みモデル](#埋め込みモデル)
- [ハンドラー](#ハンドラー)
- [ユーティリティ](#ユーティリティ)

---

## 設定

### load_settings()

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

### SearchConfig

**モジュール:** `config.py`

検索システムの設定を管理するデータクラス。デフォルト値は `config/settings.yaml` と環境変数から読み込まれる。

```python
@dataclass
class SearchConfig:
    """検索設定を管理するデータクラス"""

    # LLM設定（環境変数から読み込み）
    llm_provider: str   # DEFAULT_LLM_PROVIDER 環境変数（必須）
    llm_model: str      # DEFAULT_LLM_MODEL 環境変数（必須）

    # 埋め込みモデル設定（環境変数から読み込み）
    embedding_provider: str  # DEFAULT_EMBEDDING_PROVIDER（必須、vertex_ai / azure_openai）
    embedding_model: str     # DEFAULT_EMBEDDING_MODEL（必須）

    # 検索設定（settings.yaml batch セクションから読み込み）
    top_k: int = 4                    # 返却する結果数
    vector_weight: float = 0.9        # ベクトル検索の重み（0〜1）
    keyword_weight: float             # 自動計算（1.0 - vector_weight）

    # 検索モード: original | llm_enhanced | multi_stage
    search_mode: str = "original"

    # 入出力設定
    input_type: str = "excel"
    output_type: str = "excel"
    reference_type: str = "multi_folder"
```

#### 主要パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `llm_provider` | str | 環境変数 | LLMプロバイダー（gemini/anthropic/openai） |
| `llm_model` | str | 環境変数 | LLMモデル名 |
| `embedding_provider` | str | 環境変数 | 埋め込みプロバイダー（vertex_ai/azure_openai） |
| `embedding_model` | str | 環境変数 | 埋め込みモデル名 |
| `top_k` | int | settings.yaml | 返却する結果数 |
| `vector_weight` | float | settings.yaml | ベクトル検索の重み（0〜1） |
| `keyword_weight` | float | 自動計算 | キーワード検索の重み（1.0 - vector_weight） |
| `search_mode` | str | settings.yaml | 検索モード（original/llm_enhanced/multi_stage） |
| `search_type` | str | settings.yaml | 検索タイプ（hybrid/keyword_filter） |
| `search_source` | str | settings.yaml | 検索対象（scenario/history_data） |
| `reference_type` | str | "multi_folder" | 参照データ形式（excel/hierarchical_excel/multi_folder） |
| `force_db_update` | bool | False | 強制DB更新フラグ |
| `dual_provider_mode` | bool | False | 両プロバイダー比較モード |

#### 使用例

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

## コアモジュール

### Processor

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

#### メソッド

##### process_data

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

#### 使用例

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

### JudgmentSupport

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

#### メソッド

##### evaluate

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

#### 使用例

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

## 検索エンジン

### MultiStageOrchestrator

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

### QueryEnhancer

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

#### メソッド

##### enhance

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

## データベース管理

### DynamicDBManager

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

### MetadataVectorDB

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

## 埋め込みモデル

### BaseEmbeddingModel

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

### GeminiEmbeddingModel

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

### AzureEmbeddingModel

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

## ハンドラー

### InputHandler / InputHandlerFactory

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

### OutputHandler / OutputHandlerFactory

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

## ユーティリティ

### BusinessAreaTranslator

**モジュール:** `src/utils/business_area_translator.py`

業務領域の日本語→英語変換。

```python
class BusinessAreaTranslator:
    """業務領域変換"""

    TRANSLATION_MAP = {
        "預金": "deposit",
        "融資": "loan",
        "外貨": "foreign_currency",
        ...
    }

    @classmethod
    def translate(cls, japanese_name: str) -> str:
        """
        日本語名を英語に変換

        Args:
            japanese_name: 日本語名（例: 預金）

        Returns:
            英語名（例: deposit）

        Raises:
            ValueError: 未定義の業務領域
        """

    @classmethod
    def get_collection_name(cls, japanese_name: str) -> str:
        """
        コレクション名を取得

        Args:
            japanese_name: 日本語名

        Returns:
            コレクション名（例: deposit_DB）
        """
```

### Logger

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

## エラーハンドリング

### DynamicDBError

```python
# src/utils/dynamic_db_manager.py

class DynamicDBError(Exception):
    """動的DB管理のエラー"""
    pass
```

> **Note:** 汎用の `src/exceptions.py` は未実装。各モジュールが個別に例外を定義している。

---

## 使用例：完全なワークフロー

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

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) - アーキテクチャ
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 設定詳細
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
