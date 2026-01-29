# アーキテクチャリファクタリング実装記録 (2026-01-29)

## 概要
RAG-Geminiプロジェクトのアーキテクチャ改善として、5つの優先度項目を実装。

---

## Priority 1: 検索結果の型安全化

### 新規ファイル
- `src/types/search_types.py`
- `src/types/__init__.py`

### 主要な型定義
```python
# TypedDict（辞書型の型安全化）
SearchResultDict          # 検索結果の基本型
MultiStageSearchResultDict # 多段階検索結果（Search_Category付き）
VectorSearchResultDict    # ベクトルDB検索結果
MetadataDict             # メタデータ
ReferenceDataDict        # 参照データ

# dataclass（イミュータブルオブジェクト）
SearchResult             # 検索結果クラス
MultiStageSearchResult   # 多段階検索結果クラス
VectorSearchResult       # ベクトル検索結果クラス
ParsedCombinedText       # 結合テキスト解析結果

# 定数クラス（キー名の一元管理）
SearchResultKeys         # 'Input_Number', 'Similarity' など
MetadataKeys            # 'source', 'hierarchy' など
SourceValues            # 'scenario', 'history_data'
SearchCategoryValues    # 'Both', 'Original_Only', 'LLM_Enhanced_Only'
```

---

## Priority 2: Searcherクラスの責務分離

### 新規ディレクトリ
`src/core/search/`

### 新規ファイル
| ファイル | クラス | 責務 |
|---------|--------|------|
| `vector_search_engine.py` | `VectorSearchEngine` | ベクトル検索 |
| `keyword_search_engine.py` | `KeywordSearchEngine` | キーワード抽出・Jaccard類似度 |
| `query_enhancer.py` | `QueryEnhancer` | LLMクエリ拡張 |
| `multi_stage_orchestrator.py` | `MultiStageOrchestrator` | 多段階検索のオーケストレーション |
| `text_combiner.py` | `TextCombiner` | テキスト結合・解析 |

### 使用例
```python
from src.core.search import (
    VectorSearchEngine,
    KeywordSearchEngine,
    QueryEnhancer,
    MultiStageOrchestrator,
    TextCombiner,
)
```

---

## Priority 3: タイムスタンプ管理の簡素化

### 新規ファイル
`src/utils/db_version_manager.py`

### 主要クラス
```python
# バージョン情報
@dataclass
class DBVersionInfo:
    faq_mtime: Optional[float]
    scenario_mtime: Optional[float]
    last_updated: Optional[str]

# バージョン管理（3階層構造を抽象化）
class DBVersionManager:
    def get_faq_mtime(business_area: str) -> float
    def get_scenario_mtime(business_area: str) -> float
    def update_after_success(business_area, faq_path, scenario_path)
    def needs_update(business_area, faq_path, scenario_path) -> bool
    def save()
```

---

## Priority 4: 結合テキスト生成の統一

### ファイル
`src/core/search/text_combiner.py`

### TextCombinerクラス
```python
combiner = TextCombiner()

# 結合
combined = combiner.combine(query, answer, hierarchy)
# 出力: "分類: Lv1 > Lv2 | 質問: Q | 回答: A"

# 解析
parsed = combiner.parse(combined_text)
# parsed.hierarchy, parsed.query, parsed.answer

# 表示用クエリ構築
display_q = combiner.build_display_query(parsed, source='scenario')

# シングルトン取得
from src.core.search.text_combiner import get_text_combiner
combiner = get_text_combiner()
```

---

## Priority 5: 設定の外部化

### 新規ファイル
- `config/business_areas.yaml` - 業務分野マッピング設定
- `src/utils/business_area_translator.py` - 変換ユーティリティ

### YAML設定構造
```yaml
mappings:
  総則: general
  預金: deposit
  # ...

revision_mappings:
  rev01smile: rev01smile
  rev02souzoku: rev02souzoku
  # ...
```

### 使用例
```python
from src.utils.business_area_translator import get_translator

translator = get_translator()
english_name = translator.translate("預金")  # -> "deposit"
english_name = translator.translate("rev01smile")  # -> "rev01smile"
```

### requirements.txt追加
```
PyYAML>=6.0
```

---

## 既存コードとの統合

これらの新モジュールは既存の`Searcher`クラスと並行して使用可能。
段階的な統合を推奨：

1. 新コードで型定義をインポートして使用
2. `TextCombiner`で結合テキスト処理を統一
3. 必要に応じて`DBVersionManager`を`DynamicDBManager`に統合
4. 大規模リファクタリング時に検索エンジン分離を適用

---

## ファイル一覧

```
src/
├── types/
│   ├── __init__.py
│   └── search_types.py
├── core/
│   ├── __init__.py
│   └── search/
│       ├── __init__.py
│       ├── vector_search_engine.py
│       ├── keyword_search_engine.py
│       ├── query_enhancer.py
│       ├── multi_stage_orchestrator.py
│       └── text_combiner.py
└── utils/
    ├── __init__.py
    ├── db_version_manager.py
    └── business_area_translator.py

config/
└── business_areas.yaml
```
