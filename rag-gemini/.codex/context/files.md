# 関連ファイル一覧

## 主要修正対象ファイル

### 1. src/core/searcher.py (577行)

**役割**: ハイブリッド検索エンジン（ベクトル検索 + キーワード検索）

**主要な問題点**:
- 未使用インポート（行14: `ChatVertexAI`）
- 未使用変数（行47-49: `reference_vectors`, `reference_texts`, `reference_df`）
- 未使用メソッド（行232-336: キャッシュ関連4メソッド）
- ハードコード（行76-77: Vertex AI設定）
- 複雑なsearch()メソッド（約186行）
- 型ヒント互換性（`list[str]` → `List[str]`）

**修正箇所**:
```
行14: 削除 - from langchain_google_vertexai import ChatVertexAI
行47-49: 削除 - 未使用インスタンス変数
行66-79: 修正 - 認証処理をauth.pyに委譲
行94, 109: 修正 - 型ヒント
行117: 修正 - position_weight定数化
行232-336: 削除 - 未使用メソッド群
行365-551: 分割 - search()メソッド
行409: 修正 - VECTOR_SEARCH_MULTIPLIER定数化
行506-514: 修正 - logger.info → logger.debug
```

---

### 2. src/handlers/input_handler.py (428行)

**役割**: 入力ファイル処理（Excel, 階層構造Excel, 複数フォルダ対応）

**主要な問題点**:
- `_get_column_names()` がExcelInputHandlerとMultiFolderInputHandlerで重複
- `_build_combined_text()` 相当のロジックが複数箇所で重複
- 型ヒント互換性（`tuple[str, str, str]` → `Tuple[str, str, Optional[str]]`）
- マジックストリング（行213: 原則文判定）

**修正箇所**:
```
行7: 修正 - from typing import List, Tuple, Optional, Dict, Any
行23付近: 追加 - 基底クラスに_get_column_names(), _build_combined_text()
行138-147: 削除 - ExcelInputHandler._get_column_names()（基底クラス使用）
行213: 修正 - PRINCIPLE_MARKER定数化
行404-413: 削除 - MultiFolderInputHandler._get_column_names()（基底クラス使用）
```

---

### 3. src/utils/dynamic_db_manager.py (469行)

**役割**: 業務分野別の動的DB管理

**主要な問題点**:
- 裸のexcept:（行306）
- バッチサイズハードコード（行335）

**修正箇所**:
```
行306: 修正 - except: → except chromadb.errors.InvalidCollectionException:
行335: 修正 - batch_size = 100 → self.config.VECTOR_DB_BATCH_SIZE
```

---

### 4. config.py (98行)

**役割**: 検索設定管理（SearchConfigデータクラス）

**修正内容**: マジックナンバー・マジックストリングの定数追加

**追加箇所** (行30付近に追加):
```python
# バッチサイズ設定
EMBEDDING_BATCH_SIZE: int = 5
VECTOR_DB_BATCH_SIZE: int = 100

# 検索設定
VECTOR_SEARCH_MULTIPLIER: int = 2
POSITION_WEIGHT: float = 1.2

# 列名候補
QUERY_COLUMN_CANDIDATES: tuple = (...)
ANSWER_COLUMN_CANDIDATES: tuple = (...)
TAG_COLUMN_CANDIDATES: tuple = (...)

# 原則文判定マーカー
PRINCIPLE_MARKER: str = "以下の選択肢から選んでください"
```

---

### 5. src/utils/gemini_embedding.py (104行)

**役割**: Gemini Embedding APIによるテキストベクトル化

**主要な問題点**:
- 認証処理が重複（行22-32: searcher.pyと同じ処理）
- バッチサイズハードコード（行61）

**修正箇所**:
```
行18-32: 修正 - 認証処理をauth.pyに委譲
行61: 修正 - batch_size = 5 → self.config.EMBEDDING_BATCH_SIZE
```

---

### 6. src/utils/vector_db.py (149行)

**役割**: ChromaDBベクトルデータベース操作

**主要な問題点**:
- バッチサイズハードコード（行69）
- 裸のexcept:（行36）

**修正箇所**:
```
行14: 修正 - コンストラクタにconfig引数追加
行36: 修正 - except: → 具体的な例外型
行69: 修正 - batch_size = 100 → self.config.VECTOR_DB_BATCH_SIZE
```

---

### 7. src/utils/logger.py (23行)

**役割**: ロギング設定

**修正内容**: 環境変数によるログレベル制御、ハンドラ重複防止

**修正後**:
```python
def setup_logger(name):
    logger = logging.getLogger(name)
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    if logger.handlers:
        return logger
    # ... ハンドラ設定
```

---

## 新規作成ファイル

### src/utils/auth.py (新規)

**役割**: Google Cloud認証処理の共通モジュール

**内容**:
```python
def get_google_credentials(config): ...
def initialize_vertex_ai(config, credentials=None): ...
```

---

## ディレクトリ構造

```
rag-gemini/
├── src/
│   ├── core/
│   │   ├── processor.py      # 変更なし
│   │   └── searcher.py       # 主要修正
│   ├── handlers/
│   │   ├── input_handler.py  # 主要修正
│   │   └── output_handler.py # 変更なし
│   └── utils/
│       ├── auth.py           # 新規作成
│       ├── dynamic_db_manager.py # 修正
│       ├── gemini_embedding.py   # 修正
│       ├── logger.py         # 修正
│       ├── utils.py          # 変更なし
│       └── vector_db.py      # 修正
├── config.py                 # 修正（定数追加）
├── main.py                   # 変更なし
├── backup/                   # ZIPアーカイブ後削除
└── old/                      # ZIPアーカイブ後削除
```

---

## 依存関係

```
main.py
├── config.py (SearchConfig)
├── DynamicDBManager
│   └── config.py
└── Processor
    ├── InputHandlerFactory
    │   ├── ExcelInputHandler
    │   ├── HierarchicalExcelInputHandler
    │   └── MultiFolderInputHandler
    ├── OutputHandlerFactory
    └── Searcher
        ├── auth.py (新規) ← GeminiEmbeddingModel, _setup_llm()
        ├── GeminiEmbeddingModel
        │   └── auth.py (新規)
        ├── DynamicDBManager
        └── MetadataVectorDB
```
