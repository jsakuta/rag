# タスク: rag-gemini 非機能改善（Phase 1-3）

## 目的
rag-geminiプロジェクトのコード品質、保守性、可読性を向上させる非機能改善。機能は一切変更せず、リファクタリングのみ実施する。

---

## Phase 1: 即時対応（クリティカル）

### 1.1 デッドコードの削除

#### 変更対象: `src/core/searcher.py`

**削除項目1: 未使用インポート（行14）**
```python
# 削除対象
from langchain_google_vertexai import ChatVertexAI
```

**削除項目2: 未使用インスタンス変数（行47-49）**
```python
# 削除対象
self.reference_vectors = None
self.reference_texts = None
self.reference_df = None # processor.pyから移動
```

**削除項目3: 未使用メソッド群（行232-336）**
以下のメソッドは動的DB管理システムで代替されており未使用のため削除：
- `_is_data_unchanged()`（行232-261）
- `_check_file_timestamps()`（行263-296）
- `_get_file_timestamps()`（行298-320）
- `_save_cache_info()`（行322-336）

### 1.2 ハードコードの修正

#### 変更対象: `src/core/searcher.py` 行75-78

**現在のコード:**
```python
# Vertex AI初期化
vertexai.init(
    project="pj-cbk001",
    location="us-central1",
    credentials=credentials
)
```

**修正後のコード:**
```python
# Vertex AI初期化
vertexai.init(
    project=self.config.gemini_project_id,
    location=self.config.gemini_location,
    credentials=credentials
)
```

### 1.3 エラーハンドリング改善

#### 変更対象: `src/utils/dynamic_db_manager.py` 行306

**現在のコード:**
```python
except:
    logger.info(f"ChromaDBコレクションは存在しません: {collection_name}")
```

**修正後のコード:**
```python
except chromadb.errors.InvalidCollectionException:
    logger.info(f"ChromaDBコレクションは存在しません: {collection_name}")
```

---

## Phase 2: 短期対応

### 2.1 Google認証処理の共通化

#### 新規作成: `src/utils/auth.py`

```python
"""Google Cloud認証処理の共通モジュール"""
import os
import vertexai
from google.oauth2 import service_account
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_google_credentials(config):
    """Google Cloud認証情報を取得

    Args:
        config: SearchConfig インスタンス

    Returns:
        service_account.Credentials: 認証情報
    """
    credentials_path = os.path.join(config.base_dir, config.gemini_credentials_path)
    return service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )


def initialize_vertex_ai(config, credentials=None):
    """Vertex AIを初期化

    Args:
        config: SearchConfig インスタンス
        credentials: 認証情報（省略時は自動取得）
    """
    if credentials is None:
        credentials = get_google_credentials(config)

    vertexai.init(
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials
    )
    logger.info("Vertex AI initialized successfully")
```

#### 修正対象: `src/core/searcher.py` 行66-79

**現在のコード:**
```python
elif self.config.llm_provider == "gemini":
    # Vertex AI Gemini用の認証設定
    try:
        # 認証情報を読み込み
        credentials = service_account.Credentials.from_service_account_file(
            os.path.join(self.config.base_dir, "gemini_credentials.json"),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        # Vertex AI初期化
        vertexai.init(
            project=self.config.gemini_project_id,
            location=self.config.gemini_location,
            credentials=credentials
        )
```

**修正後のコード:**
```python
elif self.config.llm_provider == "gemini":
    # Vertex AI Gemini用の認証設定
    try:
        from src.utils.auth import get_google_credentials, initialize_vertex_ai
        credentials = get_google_credentials(self.config)
        initialize_vertex_ai(self.config, credentials)
```

#### 修正対象: `src/utils/gemini_embedding.py` 行18-32

**現在のコード:**
```python
def _setup_model(self):
    """Gemini Embedding APIの初期化"""
    try:
        # 認証情報を読み込み
        credentials = service_account.Credentials.from_service_account_file(
            os.path.join(self.config.base_dir, "gemini_credentials.json"),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        # Vertex AI初期化
        vertexai.init(
            project=self.config.gemini_project_id,
            location=self.config.gemini_location,
            credentials=credentials
        )
```

**修正後のコード:**
```python
def _setup_model(self):
    """Gemini Embedding APIの初期化"""
    try:
        from src.utils.auth import initialize_vertex_ai
        initialize_vertex_ai(self.config)
```

### 2.2 コード重複解消（input_handler.py）

#### 修正対象: 基底クラス `InputHandler` に共通メソッドを追加

**追加するメソッド（行23付近に追加）:**
```python
def _get_column_names(self, df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    """Excelファイルの列名を取得・検証"""
    if len(df.columns) < 2:
        raise ValueError("Input file must have at least 2 columns (Number and Query)")

    number_col = df.columns[0]
    query_col = df.columns[1]
    answer_col = df.columns[2] if len(df.columns) > 2 else None
    logger.info(f"Using columns: Number='{number_col}', Query='{query_col}', Answer='{answer_col}'")
    return number_col, query_col, answer_col

def _build_combined_text(self, hierarchy: str, query: str, answer: str) -> str:
    """結合テキストを生成"""
    text_parts = []
    if hierarchy and hierarchy.strip():
        text_parts.append(f"分類: {hierarchy}")
    if query and query.strip():
        text_parts.append(f"質問: {query}")
    if answer and answer.strip():
        text_parts.append(f"回答: {answer}")
    return " | ".join(text_parts) if text_parts else ""
```

**削除対象:**
- `ExcelInputHandler._get_column_names()` (行138-147) - 基底クラスを使用
- `MultiFolderInputHandler._get_column_names()` (行404-413) - 基底クラスを使用

### 2.3 マジックナンバー定数化

#### 修正対象: `config.py` に定数追加（行30付近）

```python
# バッチサイズ設定
EMBEDDING_BATCH_SIZE: int = 5
VECTOR_DB_BATCH_SIZE: int = 100

# 検索設定
VECTOR_SEARCH_MULTIPLIER: int = 2
POSITION_WEIGHT: float = 1.2

# 列名候補
QUERY_COLUMN_CANDIDATES: tuple = ('分割後質問', '問合せ内容', '質問内容', '問い合わせ', '質問', 'query', 'Query')
ANSWER_COLUMN_CANDIDATES: tuple = ('分割後回答', '回答', '既存回答', 'answer', 'Answer')
TAG_COLUMN_CANDIDATES: tuple = ('タグ付け', 'タグ', '分類', 'category', 'Category', 'tag', 'Tag')

# 原則文判定マーカー
PRINCIPLE_MARKER: str = "以下の選択肢から選んでください"
```

#### 各ファイルでの定数参照

**gemini_embedding.py 行61:**
```python
# Before
batch_size = 5

# After
batch_size = self.config.EMBEDDING_BATCH_SIZE
```

**vector_db.py 行69:**
```python
# Before
batch_size = 100

# After（コンストラクタにconfigを渡す方式に変更）
# __init__でconfigを受け取り、self.config.VECTOR_DB_BATCH_SIZEを使用
```

**searcher.py 行117:**
```python
# Before
position_weight = 1.2

# After
position_weight = self.config.POSITION_WEIGHT
```

**searcher.py 行409:**
```python
# Before
n_results=self.config.top_k * 2

# After
n_results=self.config.top_k * self.config.VECTOR_SEARCH_MULTIPLIER
```

**input_handler.py 行213:**
```python
# Before
is_principle = "以下の選択肢から選んでください" in answer

# After
is_principle = self.config.PRINCIPLE_MARKER in answer
```

### 2.4 ロギング最適化

#### 修正対象: `src/utils/logger.py`

**修正後のコード（全体置換）:**
```python
# --- utils/logger.py ---
import logging
import os


def setup_logger(name):
    """ロガーの設定"""
    logger = logging.getLogger(name)

    # 環境変数でログレベルを制御
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # ハンドラの重複追加を防止
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
```

#### 修正対象: `src/core/searcher.py` - 詳細ログをdebugレベルに変更

**行506-514 の詳細ログをdebugに変更:**
```python
# Before
logger.info(f"  【結果{i+1}】(i={i})")
logger.info(f"    Input_Number: {result_data['Input_Number']}")
...

# After
logger.debug(f"  【結果{i+1}】(i={i})")
logger.debug(f"    Input_Number: {result_data['Input_Number']}")
...
```

### 2.5 型ヒント統一

#### 修正対象: `src/core/searcher.py`

**行94:**
```python
# Before
def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:

# After
def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
```

**行109:**
```python
# Before
def _calculate_keyword_similarity(self, query_keywords: list[str], reference_text: str) -> float:

# After
def _calculate_keyword_similarity(self, query_keywords: List[str], reference_text: str) -> float:
```

#### 修正対象: `src/handlers/input_handler.py`

**行7:**
```python
# Before
from typing import List

# After
from typing import List, Tuple, Optional, Dict, Any
```

**行138, 404:**
```python
# Before
def _get_column_names(self, df: pd.DataFrame) -> tuple[str, str, str]:

# After (基底クラスに移動後)
def _get_column_names(self, df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
```

---

## Phase 3: 中期対応

### 3.1 search()メソッド分割

#### 変更対象: `src/core/searcher.py`

現状の`search()`メソッド（約186行）を以下のプライベートメソッドに分割：

```python
def search(self, input_number: str, query_text: str, original_answer: str, input_file: str = None) -> List[Dict[str, Any]]:
    """メタデータ対応ハイブリッド検索を実行"""
    self._select_db_if_needed(input_file)
    search_query, query_for_vector = self._prepare_search_query(query_text)
    keywords = self._extract_keywords(query_text)
    search_results = self._execute_vector_search(query_for_vector)
    results = self._calculate_and_merge_scores(search_results, keywords)
    return self._format_final_results(results, input_number, query_text, original_answer, search_query)


def _select_db_if_needed(self, input_file: Optional[str]) -> None:
    """動的DB選択（入力ファイルが指定されている場合）"""
    # 行368-376 のロジックを移動


def _prepare_search_query(self, query_text: str) -> Tuple[str, str]:
    """検索クエリの生成（検索方式による分岐）"""
    # 行386-396 のロジックを移動


def _execute_vector_search(self, query_for_vector: str) -> List[Dict[str, Any]]:
    """ベクトル検索を実行"""
    # 行405-419 のロジックを移動


def _calculate_and_merge_scores(self, search_results: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    """スコア計算と結果統合"""
    # 行435-516 のロジックを移動


def _format_final_results(self, results: List[Dict[str, Any]], input_number: str, query_text: str, original_answer: str, search_query: str) -> List[Dict[str, Any]]:
    """結果の整形とソート"""
    # 行529-551 のロジックを移動
```

### 3.2 依存性注入導入

#### 変更対象: `src/core/searcher.py` の `__init__` メソッド

**修正後のコード:**
```python
def __init__(self, config: SearchConfig,
             db_manager: Optional[DynamicDBManager] = None,
             embedding_model: Optional['GeminiEmbeddingModel'] = None):
    self.config = config
    self.tokenizer = Dictionary().create()
    self.mode = tokenizer.Tokenizer.SplitMode.C

    # 依存性注入: 外部から注入されない場合はデフォルトを生成
    if embedding_model is None:
        from src.utils.gemini_embedding import GeminiEmbeddingModel
        embedding_model = GeminiEmbeddingModel(config)
    self.model = embedding_model

    self.db_manager = db_manager or DynamicDBManager(config)
    self.current_db_path = None
    self.current_business_area = None
    logger.info("動的DB管理システムを初期化しました")

    # LLM初期化（条件付き）
    if self.config.search_mode == "llm_enhanced" and self.config.enable_query_enhancement:
        self.llm = self._setup_llm()
        logger.info("LLM initialized for enhanced search mode")
    else:
        self.llm = None
        logger.info("LLM not initialized - using original search mode")
```

### 3.3 バックアップファイル整理

1. `backup/` フォルダをZIPアーカイブ
2. `old/` フォルダをZIPアーカイブ
3. `.gitignore` に追加:
```
backup/
old/
*.bak
```

---

## 完了条件

- [ ] Phase 1: デッドコード削除、ハードコード修正、エラーハンドリング改善
- [ ] Phase 2: 認証処理共通化、コード重複解消、定数化、ログ最適化、型ヒント統一
- [ ] Phase 3: search()分割、依存性注入、バックアップ整理
- [ ] Pythonエラーがない（`python -m py_compile`）
- [ ] 既存機能が正常動作
