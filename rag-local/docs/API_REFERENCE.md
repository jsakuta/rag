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

### SearchConfig

**モジュール:** `config.py`

検索システムの設定を管理するデータクラス。

```python
@dataclass
class SearchConfig:
    """検索設定の統合管理"""

    # LLM設定
    llm_provider: str = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"

    # 埋め込みモデル設定
    embedding_provider: str = "azure_openai"
    embedding_model: str = "text-embedding-3-large"

    # 検索設定
    top_k: int = 4
    vector_weight: float = 0.9
    keyword_weight: float = 0.1

    # 検索モード: original | llm_enhanced | multi_stage
    search_mode: str = "original"

    # 参照データ形式
    reference_type: str = "multi_folder"
```

#### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `llm_provider` | str | "gemini" | LLMプロバイダー（gemini/anthropic/openai） |
| `llm_model` | str | "gemini-2.5-flash-lite" | LLMモデル名 |
| `embedding_provider` | str | "azure_openai" | 埋め込みプロバイダー |
| `embedding_model` | str | "text-embedding-3-large" | 埋め込みモデル名 |
| `top_k` | int | 4 | 返却する結果数 |
| `vector_weight` | float | 0.9 | ベクトル検索の重み |
| `keyword_weight` | float | 0.1 | キーワード検索の重み |
| `search_mode` | str | "original" | 検索モード（original/llm_enhanced/multi_stage） |
| `reference_type` | str | "multi_folder" | 参照データ形式 |

#### 使用例

```python
from config import SearchConfig

# デフォルト設定
config = SearchConfig()

# カスタム設定
config = SearchConfig(
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022",
    embedding_provider="vertex_ai",
    top_k=10,
    vector_weight=0.7,
    search_mode="llm_enhanced"
)
```

---

## コアモジュール

### DataProcessor

**モジュール:** `src/core/processor.py`

データ処理の統合管理クラス。

```python
class DataProcessor:
    """データ処理エンジン"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """
```

#### メソッド

##### process_batch

```python
def process_batch(self, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    バッチ処理実行

    Args:
        input_data: 入力データフレーム（列: 番号、質問内容）

    Returns:
        結果データフレーム（列: 番号、質問、検索クエリ、類似質問、類似回答、類似度）

    Raises:
        ValueError: 入力データの形式が不正
    """
```

#### 使用例

```python
from config import SearchConfig
from src.core.processor import DataProcessor
import pandas as pd

config = SearchConfig()
processor = DataProcessor(config)

# 入力データ
input_df = pd.DataFrame({
    "番号": [1, 2],
    "質問内容": ["口座開設について", "残高照会の方法"]
})

# 処理実行
result_df = processor.process_batch(input_df)
```

### JudgmentSupport

**モジュール:** `src/core/judgment_support.py`

LLMによる検索結果の関連性判定。

```python
class JudgmentSupport:
    """関連性判定エンジン"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """
```

#### メソッド

##### analyze_relevance

```python
def analyze_relevance(
    self,
    query: str,
    search_result_q: str,
    search_result_a: str
) -> Dict[str, str]:
    """
    関連性を分析

    Args:
        query: ユーザークエリ
        search_result_q: 検索結果の質問
        search_result_a: 検索結果の回答

    Returns:
        {
            "relevance": "関連あり" | "要確認" | "関連なし",
            "reason": "判定根拠",
            "suggestion": "修正案（オプション）"
        }

    Raises:
        Exception: LLM API呼び出しエラー
    """
```

#### 使用例

```python
from src.core.judgment_support import JudgmentSupport

support = JudgmentSupport(config)

result = support.analyze_relevance(
    query="口座開設について",
    search_result_q="普通預金口座の開設方法",
    search_result_a="本人確認書類をご持参の上..."
)

print(result["relevance"])  # "関連あり"
print(result["reason"])     # "質問の主題が一致しています"
```

---

## 検索エンジン

### MultiStageOrchestrator

**モジュール:** `src/core/search/multi_stage_orchestrator.py`

多段階ハイブリッド検索のオーケストレーター。

```python
class MultiStageOrchestrator:
    """多段階検索エンジン"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
        """
```

#### メソッド

##### search

```python
def search(
    self,
    query: str,
    db_manager: DynamicDBManager,
    business_area: str = "general"
) -> List[SearchResult]:
    """
    多段階検索実行

    Args:
        query: 検索クエリ
        db_manager: DB管理インスタンス
        business_area: 業務領域

    Returns:
        検索結果リスト（SearchResultオブジェクト）

    Raises:
        ValueError: DBが存在しない
    """
```

#### SearchResult

```python
@dataclass
class SearchResult:
    """検索結果"""
    scenario_id: str          # シナリオID
    question: str             # 質問
    answer: str               # 回答
    similarity: float         # 類似度スコア
    category: str             # カテゴリ（Both/Original_Only/LLM_Enhanced_Only）
    source: str               # ソース（scenario/faq_data）
    metadata: Dict            # メタデータ
```

#### 使用例

```python
from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
from src.utils.dynamic_db_manager import DynamicDBManager

orchestrator = MultiStageOrchestrator(config)
db_manager = DynamicDBManager(config)

results = orchestrator.search(
    query="口座開設について",
    db_manager=db_manager,
    business_area="deposit"
)

for result in results:
    print(f"{result.scenario_id}: {result.similarity:.3f} ({result.category})")
```

### QueryEnhancer

**モジュール:** `src/core/search/query_enhancer.py`

LLMによるクエリ拡張。

```python
class QueryEnhancer:
    """クエリ拡張エンジン"""

    def __init__(self, config: SearchConfig):
        """
        Args:
            config: 検索設定
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

業務領域別のベクトルDB管理。

```python
class DynamicDBManager:
    """動的DB管理"""

    def __init__(
        self,
        config: SearchConfig,
        db_base_dir: str = "data/vector_db"
    ):
        """
        Args:
            config: 検索設定
            db_base_dir: DBベースディレクトリ
        """
```

#### メソッド

##### get_or_create_db

```python
def get_or_create_db(self, business_area: str) -> VectorDB:
    """
    DBの取得または作成

    Args:
        business_area: 業務領域（例: deposit, loan）

    Returns:
        VectorDBインスタンス

    Raises:
        ValueError: 業務領域が不正
    """
```

##### reset_db

```python
def reset_db(self, business_area: str):
    """
    DBをリセット

    Args:
        business_area: 業務領域

    Note:
        既存のコレクションを削除し、再ベクトル化
    """
```

#### 使用例

```python
from src.utils.dynamic_db_manager import DynamicDBManager

db_manager = DynamicDBManager(config)

# DBの取得（自動作成）
deposit_db = db_manager.get_or_create_db("deposit")

# DBのリセット
db_manager.reset_db("deposit")
```

### VectorDB

**モジュール:** `src/utils/vector_db.py`

ChromaDBの操作ラッパー。

```python
class VectorDB:
    """ベクトルDB管理"""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: BaseEmbeddingModel
    ):
        """
        Args:
            collection_name: コレクション名
            persist_directory: 永続化ディレクトリ
            embedding_model: 埋め込みモデル
        """
```

#### メソッド

##### add_documents

```python
def add_documents(
    self,
    documents: List[Dict],
    batch_size: int = 100
):
    """
    ドキュメント追加

    Args:
        documents: ドキュメントリスト
            [
                {
                    "text": "質問と回答を結合したテキスト",
                    "metadata": {...}
                },
                ...
            ]
        batch_size: バッチサイズ

    Raises:
        Exception: 追加処理エラー
    """
```

##### search

```python
def search(
    self,
    query: str,
    top_k: int = 10,
    filters: Dict = None
) -> List[Dict]:
    """
    ベクトル検索

    Args:
        query: 検索クエリ
        top_k: 返却数
        filters: メタデータフィルタ（例: {"source": "scenario"}）

    Returns:
        検索結果リスト
        [
            {
                "text": "...",
                "distance": 0.85,
                "metadata": {...}
            },
            ...
        ]
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

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        テキストをベクトル化

        Args:
            texts: テキストリスト

        Returns:
            ベクトルリスト（各ベクトルは3072次元）
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        クエリをベクトル化

        Args:
            query: クエリテキスト

        Returns:
            ベクトル（3072次元）
        """
        pass
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

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 5
    ) -> List[List[float]]:
        """
        テキストをベクトル化

        Args:
            texts: テキストリスト
            batch_size: バッチサイズ（最大250）

        Returns:
            ベクトルリスト
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

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[List[float]]:
        """
        テキストをベクトル化

        Args:
            texts: テキストリスト
            batch_size: バッチサイズ（最大2048）

        Returns:
            ベクトルリスト
        """
```

---

## ハンドラー

### InputHandler

**モジュール:** `src/handlers/input_handler.py`

入力ファイルの処理。

```python
class InputHandler:
    """入力処理"""

    @staticmethod
    def read_input_file(file_path: str) -> pd.DataFrame:
        """
        入力ファイル読み込み

        Args:
            file_path: Excelファイルパス

        Returns:
            データフレーム（列: 番号、質問内容）

        Raises:
            FileNotFoundError: ファイルが存在しない
            ValueError: 必須列が存在しない
        """

    @staticmethod
    def read_reference_data(
        folder_path: str,
        reference_type: str = "multi_folder"
    ) -> List[Dict]:
        """
        参照データ読み込み

        Args:
            folder_path: フォルダパス
            reference_type: 参照データ形式

        Returns:
            ドキュメントリスト
            [
                {
                    "text": "...",
                    "metadata": {
                        "source": "scenario",
                        "hierarchy": "Lv0 > Lv1 > Lv2",
                        ...
                    }
                },
                ...
            ]
        """
```

### OutputHandler

**モジュール:** `src/handlers/output_handler.py`

出力ファイルの生成。

```python
class OutputHandler:
    """出力処理"""

    @staticmethod
    def save_results(
        results: pd.DataFrame,
        output_path: str,
        config: SearchConfig
    ):
        """
        結果を保存

        Args:
            results: 結果データフレーム
            output_path: 出力ファイルパス
            config: 検索設定

        Raises:
            Exception: 保存エラー
        """
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

ログ設定。

```python
def setup_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    ロガーを設定

    Args:
        name: ロガー名
        log_file: ログファイルパス
        level: ログレベル

    Returns:
        ロガーインスタンス
    """
```

---

## エラーハンドリング

### カスタム例外

```python
# src/exceptions.py（将来実装予定）

class RAGException(Exception):
    """RAGシステムの基底例外"""
    pass

class EmbeddingError(RAGException):
    """埋め込みエラー"""
    pass

class SearchError(RAGException):
    """検索エラー"""
    pass

class DBError(RAGException):
    """DB操作エラー"""
    pass
```

---

## 使用例：完全なワークフロー

```python
from config import SearchConfig
from src.core.processor import DataProcessor
from src.handlers.input_handler import InputHandler
from src.handlers.output_handler import OutputHandler
import pandas as pd

# 1. 設定
config = SearchConfig(
    llm_provider="gemini",
    llm_model="gemini-2.5-flash-lite",
    embedding_provider="azure_openai",
    top_k=4,
    vector_weight=0.9,
    search_mode="llm_enhanced"
)

# 2. 入力データ読み込み
input_df = InputHandler.read_input_file("data/input/data.xlsx")

# 3. 処理実行
processor = DataProcessor(config)
result_df = processor.process_batch(input_df)

# 4. 結果保存
OutputHandler.save_results(
    results=result_df,
    output_path="data/output/results.xlsx",
    config=config
)
```

---

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト概要
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) - アーキテクチャ
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 設定詳細
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
