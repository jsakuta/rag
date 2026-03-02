# rag-local コード簡素化 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** レビューで特定された14件の正しい指摘（デッドコード、重複ロジック、バグ、効率問題）を5 Phase で修正する。

**Architecture:** Phase 1でデッドコードを安全に除去 → Phase 2で共通ユーティリティを構築 → Phase 3でPhase 2を使って重複を集約（バグ修正含む） → Phase 4でパフォーマンス最適化 → Phase 5で小改善。各Phase完了後にpytestで回帰テスト。

**Tech Stack:** Python 3.11, pytest, Streamlit, ChromaDB, pandas, dataclasses

**Design doc:** `docs/plans/2026-03-02-code-simplification-design.md`

---

## Task 1: Phase 1 — デッドコード・孤立参照の削除

**Files:**
- Modify: `src/utils/__init__.py`
- Modify: `src/utils/business_area_translator.py`
- Modify: `apps/revision-ops/run_eval.py:1169,1177`
- Modify: `CLAUDE.md`
- Modify: `docs/ARCHITECTURE.md`

### Step 1: `__init__.py` から db_version_manager と get_translator の参照を削除

`src/utils/__init__.py` を以下に置換:

```python
# --- src/utils/__init__.py ---
"""ユーティリティモジュール"""

from src.utils.logger import setup_logger
from src.utils.business_area_translator import BusinessAreaTranslator

__all__ = [
    'setup_logger',
    'BusinessAreaTranslator',
]
```

削除内容:
- `from src.utils.db_version_manager import DBVersionManager, DBVersionInfo`（ファイルが存在しないのにインポートしている残骸）
- `get_translator` のインポートとエクスポート（呼び出し元ゼロ）

### Step 2: `business_area_translator.py` から `get_translator()` と `_default_translator` を削除

`src/utils/business_area_translator.py` の L155末尾〜L174 を削除:

```python
# 削除: L158-174
# グローバルインスタンス（シングルトンパターン）
_default_translator: Optional[BusinessAreaTranslator] = None


def get_translator(config_path: Optional[str] = None) -> BusinessAreaTranslator:
    ...
```

残す: `_AREA_DISPLAY_NAMES` dict（L177-183）と `get_display_name()` 関数（使用中）。

### Step 3: `run_eval.py` のノーオプ三項演算子を修正

`apps/revision-ops/run_eval.py` L1169, L1177:

```python
# Before:
worksheet.write(row_num, start_col + i, value if value != "" else "", fmt)
# After:
worksheet.write(row_num, start_col + i, value, fmt)
```

L1177 も同様:
```python
# Before:
worksheet.write(row_num, start_col + i, value if value != "" else "", formats["cell"])
# After:
worksheet.write(row_num, start_col + i, value, formats["cell"])
```

### Step 4: ドキュメントから `db_version_manager.py` 参照を削除

**CLAUDE.md**: ディレクトリツリーの `├─ db_version_manager.py # DBバージョン管理` 行を削除。
「データベース管理」セクションの `- **db_version_manager.py**: DBバージョン管理` 行を削除。

**docs/ARCHITECTURE.md**: Utils Layer の `│   ├─ db_version_manager.py - バージョン管理` 行を削除。
依存ツリーの `│       ├─ src/utils/db_version_manager.py` 行を削除。

### Step 5: テスト実行

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v
```

Expected: 全テスト PASS

### Step 6: コミット

```bash
git add src/utils/__init__.py src/utils/business_area_translator.py apps/revision-ops/run_eval.py CLAUDE.md docs/ARCHITECTURE.md
git commit -m "refactor: remove dead code (db_version_manager refs, get_translator, no-op conditionals)"
```

---

## Task 2: Phase 2 — 共通ユーティリティ構築

**Files:**
- Modify: `src/core/search/text_combiner.py`
- Create: `tests/unit/test_text_combiner.py`
- Modify: `src/utils/business_area_translator.py`

### Step 1: `TextCombiner.build()` のテストを書く

`tests/unit/test_text_combiner.py` を新規作成:

```python
"""TextCombiner のユニットテスト"""
import pytest
from src.core.search.text_combiner import TextCombiner, get_text_combiner


class TestTextCombinerBuild:
    """TextCombiner.build() のテスト"""

    def setup_method(self):
        self.combiner = TextCombiner()

    def test_build_full(self):
        """階層・質問・回答すべてあり"""
        result = self.combiner.build("口座 > 開設", "口座開設方法は？", "窓口へお越しください")
        assert result == "分類: 口座 > 開設 | 質問: 口座開設方法は？ | 回答: 窓口へお越しください"

    def test_build_query_and_answer_only(self):
        """FAQ形式（階層なし）"""
        result = self.combiner.build(query="質問文", answer="回答文")
        assert result == "質問: 質問文 | 回答: 回答文"

    def test_build_empty_strings(self):
        """空文字列は除外"""
        result = self.combiner.build("", "質問のみ", "")
        assert result == "質問: 質問のみ"

    def test_build_whitespace_only(self):
        """空白のみも除外"""
        result = self.combiner.build("  ", "質問", "  ")
        assert result == "質問: 質問"

    def test_build_all_empty(self):
        """全て空"""
        result = self.combiner.build("", "", "")
        assert result == ""

    def test_build_roundtrip_with_parse(self):
        """build → parse のラウンドトリップ"""
        original = self.combiner.build("Lv1 > Lv2", "質問テキスト", "回答テキスト")
        parsed = self.combiner.parse(original)
        assert parsed.hierarchy == "Lv1 > Lv2"
        assert parsed.query == "質問テキスト"
        assert parsed.answer == "回答テキスト"


class TestTextCombinerParse:
    """既存 parse() の動作確認テスト"""

    def setup_method(self):
        self.combiner = TextCombiner()

    def test_parse_full(self):
        result = self.combiner.parse("分類: A > B | 質問: Q | 回答: A")
        assert result.hierarchy == "A > B"
        assert result.query == "Q"
        assert result.answer == "A"

    def test_parse_partial(self):
        result = self.combiner.parse("質問: Q | 回答: A")
        assert result.hierarchy == ""
        assert result.query == "Q"
        assert result.answer == "A"


class TestGetTextCombiner:
    """シングルトンインスタンスのテスト"""

    def test_returns_same_instance(self):
        a = get_text_combiner()
        b = get_text_combiner()
        assert a is b
```

### Step 2: テストが失敗することを確認

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_text_combiner.py -v
```

Expected: `test_build_*` が AttributeError で FAIL（`build` メソッド未定義）

### Step 3: `TextCombiner.build()` を実装

`src/core/search/text_combiner.py` の `parse()` メソッドの直後（`build_display_query()` の前）に追加:

```python
    def build(self, hierarchy: str = "", query: str = "", answer: str = "") -> str:
        """構造化データから結合テキストを生成（parse の逆操作）

        Args:
            hierarchy: 階層構造テキスト
            query: 質問テキスト
            answer: 回答テキスト

        Returns:
            str: 結合テキスト（例: "分類: A | 質問: Q | 回答: A"）
        """
        parts = []
        if hierarchy and hierarchy.strip():
            parts.append(f"{self.LABEL_HIERARCHY}: {hierarchy}")
        if query and query.strip():
            parts.append(f"{self.LABEL_QUERY}: {query}")
        if answer and answer.strip():
            parts.append(f"{self.LABEL_ANSWER}: {answer}")
        return self.separator.join(parts)
```

### Step 4: テスト通過を確認

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/unit/test_text_combiner.py -v
```

Expected: 全テスト PASS

### Step 5: `resolve_bot_name()` を `business_area_translator.py` に追加

`src/utils/business_area_translator.py` の `get_display_name()` 関数の後（ファイル末尾付近）に追加:

```python
def resolve_bot_name(area: str, area_to_bot: dict) -> str:
    """area名からbot名をsubstring-matchで解決

    Args:
        area: エリア名（例: "rev02_souzoku", "smile"）
        area_to_bot: エリア→ボット名マッピング dict

    Returns:
        str: ボット名（マッチなしの場合 "unknown-bot"）
    """
    area_lower = area.lower()
    for keyword, bot_name in area_to_bot.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"
```

`src/utils/__init__.py` にエクスポート追加:

```python
from src.utils.business_area_translator import (
    BusinessAreaTranslator,
    resolve_bot_name,
)

__all__ = [
    'setup_logger',
    'BusinessAreaTranslator',
    'resolve_bot_name',
]
```

### Step 6: テスト実行

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v
```

Expected: 全テスト PASS（新規テスト含む）

### Step 7: コミット

```bash
git add src/core/search/text_combiner.py tests/unit/test_text_combiner.py src/utils/business_area_translator.py src/utils/__init__.py
git commit -m "refactor: add TextCombiner.build() and resolve_bot_name() shared utilities"
```

---

## Task 3: Phase 3 — 重複ロジック統合 + バグ修正

**Files:**
- Modify: `src/handlers/input_handler.py:182-191,367-384,398-416,477-494`
- Modify: `src/utils/dynamic_db_manager.py:1004-1016,1070-1076`
- Modify: `apps/revision-ops/ui/ops_ui.py:92-98`
- Modify: `apps/revision-ops/run_eval.py:184-189`
- Modify: `src/core/search/chromadb_keyword_search.py:197-203`
- Modify: `src/core/searcher.py:261-284,459`
- Modify: `src/core/search/search_strategy.py:275`

### Step 1: `combined_text` 構築を `TextCombiner.build()` に置換

**input_handler.py — ExcelInputHandler.load_reference_data() (L182-191)**

Before:
```python
text_parts = []
if query_text.strip():
    text_parts.append(f"質問: {query_text}")
if answer_text.strip():
    text_parts.append(f"回答: {answer_text}")
combined_text = " | ".join(text_parts) if text_parts else ""
```

After:
```python
combined_text = self._text_combiner.build(query=query_text, answer=answer_text)
```

`__init__` にインスタンス追加が必要:
```python
from src.core.search.text_combiner import get_text_combiner
# __init__ 内:
self._text_combiner = get_text_combiner()
```

**input_handler.py — HierarchicalExcelInputHandler.load_reference_data() (L367-384)**

Before（18行）:
```python
text_parts = []
if hierarchy.strip():
    text_parts.append(f"分類: {hierarchy}")
if query.strip():
    text_parts.append(f"質問: {query}")
if answer.strip():
    text_parts.append(f"回答: {answer}")
combined_text = " | ".join(text_parts) if text_parts else ""
```

After（1行）:
```python
combined_text = self._text_combiner.build(hierarchy, query, answer)
```

**input_handler.py — MultiFolderInputHandler.load_reference_data() (L477-494)**

同パターン → 同置換。

**dynamic_db_manager.py — _prepare_reference_data() (L1004-1016)**

Before:
```python
all_combined_texts = []
for query, answer, metadata in zip(all_queries, all_answers, all_metadatas):
    hierarchy = metadata.get('hierarchy', '') if metadata else ''
    text_parts = []
    if hierarchy.strip():
        text_parts.append(f"分類: {hierarchy}")
    if query.strip():
        text_parts.append(f"質問: {query}")
    if answer.strip():
        text_parts.append(f"回答: {answer}")
    combined_text = " | ".join(text_parts) if text_parts else ""
    all_combined_texts.append(combined_text)
```

After:
```python
text_combiner = get_text_combiner()
all_combined_texts = [
    text_combiner.build(
        hierarchy=(metadata.get('hierarchy', '') if metadata else ''),
        query=query,
        answer=answer,
    )
    for query, answer, metadata in zip(all_queries, all_answers, all_metadatas)
]
```

**dynamic_db_manager.py — _load_faq_file() (L1070-1076)**

Before:
```python
text_parts = []
if query_text.strip():
    text_parts.append(f"質問: {query_text}")
if answer_text.strip():
    text_parts.append(f"回答: {answer_text}")
combined_texts.append(" | ".join(text_parts) if text_parts else "")
```

After:
```python
combined_texts.append(text_combiner.build(query=query_text, answer=answer_text))
```

（`_load_faq_file` の先頭で `text_combiner = get_text_combiner()` を追加）

### Step 2: `bot_name` 抽出を `resolve_bot_name()` に置換

**ops_ui.py — `extract_bot_name_from_area()` (L92-98)**

Before:
```python
def extract_bot_name_from_area(area: str) -> str:
    area_lower = area.lower()
    for keyword, bot_name in AREA_TO_BOT.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"
```

After:
```python
from src.utils.business_area_translator import resolve_bot_name

def extract_bot_name_from_area(area: str) -> str:
    return resolve_bot_name(area, AREA_TO_BOT)
```

**run_eval.py — `_extract_bot_name_from_area()` (L184-189)**

Before:
```python
def _extract_bot_name_from_area(self, area: str) -> str:
    area_lower = area.lower()
    for keyword, bot_name in AREA_TO_BOT.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"
```

After:
```python
def _extract_bot_name_from_area(self, area: str) -> str:
    return resolve_bot_name(area, AREA_TO_BOT)
```

（ファイル先頭に `from src.utils.business_area_translator import resolve_bot_name` を追加）

**chromadb_keyword_search.py — `_resolve_bot_name()` (L197-203)**

Before:
```python
def _resolve_bot_name(self, area: str) -> str:
    area_lower = area.lower()
    for keyword, bot_name in self.area_to_bot.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"
```

After:
```python
def _resolve_bot_name(self, area: str) -> str:
    return resolve_bot_name(area, self.area_to_bot)
```

（ファイル先頭に `from src.utils.business_area_translator import resolve_bot_name` を追加）

### Step 3: `parse_enhanced_combined_text` 削除 + バグ修正

**searcher.py — メソッド削除 (L261-284)**

`parse_enhanced_combined_text()` メソッドを完全に削除。

**searcher.py — 呼び出し元更新 (L459)**

Before:
```python
parsed_text = self.parse_enhanced_combined_text(combined_text)
```

After:
```python
from src.core.search.text_combiner import get_text_combiner
# ...
parsed_text = get_text_combiner().parse(combined_text)
```

返り値が dict → `ParsedCombinedText` dataclass に変わるため、アクセス方法を確認:
- `parsed_text['query']` → `parsed_text.query`（属性アクセス）
- `parsed_text['answer']` → `parsed_text.answer`
- `parsed_text['hierarchy']` → `parsed_text.hierarchy`

L459-475 周辺のアクセスを全て属性アクセスに変更。

**search_strategy.py — 呼び出し元更新 (L275)**

Before:
```python
parsed = self.searcher.parse_enhanced_combined_text(combined_text)
```

After:
```python
parsed = get_text_combiner().parse(combined_text)
```

ファイル先頭に `from src.core.search.text_combiner import get_text_combiner` を追加。

L279-282 のアクセスも dict → 属性に変更:
```python
# Before:
q = parsed['query']
sa = parsed['answer']
# After:
q = parsed.query
sa = parsed.answer
```

**バグ修正の効果**: `part[3:]` の off-by-one（「: テスト」とコロンが残る）が `TextCombiner.parse()` の `part[len("分類: "):]` で正しく解消。

### Step 4: `MultiFolderInputHandler.load_data()` 削除

**input_handler.py — L398-416 を削除**

`MultiFolderInputHandler.load_data()` は `ExcelInputHandler.load_data()` (L103-120) と完全に同一。
`MultiFolderInputHandler` は `ExcelInputHandler` を継承しているため、削除すれば親クラスのメソッドが使われる。

削除対象:
```python
    def load_data(self) -> list:
        # 入力データの読み込み（従来通り）
        input_file = self._get_latest_file(self.input_dir, "*.xlsx", name_regex=self.config.INPUT_FILE_PATTERN)
        ... # 全18行
        return data
```

### Step 5: テスト実行

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v
```

Expected: 全テスト PASS

### Step 6: コミット

```bash
git add src/handlers/input_handler.py src/utils/dynamic_db_manager.py apps/revision-ops/ui/ops_ui.py apps/revision-ops/run_eval.py src/core/search/chromadb_keyword_search.py src/core/searcher.py src/core/search/search_strategy.py
git commit -m "refactor: consolidate duplicated combined_text/bot_name logic, fix parse off-by-one bug"
```

---

## Task 4: Phase 4 — Searcher クリーンアップ + パフォーマンス最適化

**Files:**
- Modify: `src/core/searcher.py:22-54,80-112,258,424,434`
- Modify: `src/core/search/search_strategy.py:56,82,102,243`
- Modify: `src/core/search/multi_stage_orchestrator.py:292-294`
- Modify: `src/core/search/search_strategy.py:191-192`
- Modify: `src/core/search/keyword_search_engine.py:91-96`
- Modify: `tests/unit/test_search_strategy.py`
- Modify: `tests/unit/test_keyword_similarity_sync.py`

### Step 1: `searcher.py` — `_keyword_engine` を `__init__` で明示初期化

モジュールレベルに import 追加:
```python
from src.core.search.keyword_search_engine import KeywordSearchEngine
```

`__init__` メソッド内（L62付近、`logger.debug("Searcherを初期化しました")` の前）に追加:
```python
        # キーワード検索エンジン（明示初期化）
        self._keyword_engine = KeywordSearchEngine(
            stop_words=self.config.STOP_WORDS,
            position_weight=self.config.POSITION_WEIGHT
        )
```

### Step 2: `searcher.py` — deprecated ラッパーメソッド削除

L80-112 の `_extract_keywords()` と `_calculate_keyword_similarity()` を完全に削除。

### Step 3: `searcher.py` — orphaned tokenizer 関連コード削除

L22-36 付近の以下を全て削除:
```python
import threading as _threading
_shared_tokenizer = None
_tokenizer_lock = _threading.Lock()

@classmethod
def _get_shared_tokenizer(cls):
    ...
```

L53-54 を削除:
```python
self.tokenizer = self._get_shared_tokenizer()
self.mode = tokenizer.Tokenizer.SplitMode.C
```

モジュールレベルの不要になったimport（`sudachipy` の `Dictionary`, `tokenizer`）も削除
（ただし `KeywordSearchEngine` が内部で使っているため、searcher.py のimportのみ削除）。

### Step 4: `searcher.py` — 内部呼び出しを `_keyword_engine` に書き換え

L258:
```python
# Before:
self._reference_keywords_cache[i] = set(self._extract_keywords(query))
# After:
self._reference_keywords_cache[i] = set(self._keyword_engine.extract_keywords(query))
```

L424:
```python
# Before:
keyword_sim = self._calculate_keyword_similarity(keywords, ref_query)
# After:
keyword_sim = self._keyword_engine.calculate_similarity(keywords, ref_query)
```

L434:
```python
# Before:
doc_keywords = set(self._extract_keywords(doc_text))
# After:
doc_keywords = set(self._keyword_engine.extract_keywords(doc_text))
```

### Step 5: `search_strategy.py` — 呼び出しを `_keyword_engine` に書き換え

4箇所を置換:

L56:
```python
# Before:
keywords = self.searcher._extract_keywords(query_text)
# After:
keywords = self.searcher._keyword_engine.extract_keywords(query_text)
```

L82, L102, L243: 同パターン。

### Step 6: テストの mock を更新

**test_search_strategy.py**

L67:
```python
# Before:
searcher._extract_keywords.return_value = ["口座", "開設"]
# After:
searcher._keyword_engine.extract_keywords.return_value = ["口座", "開設"]
```

L75:
```python
# Before:
searcher._extract_keywords.assert_called_once_with("口座開設の方法")
# After:
searcher._keyword_engine.extract_keywords.assert_called_once_with("口座開設の方法")
```

L92, L112, L124: 同パターン。

**test_keyword_similarity_sync.py**

`_make_searcher_with_cache()` 関数:
```python
# Before:
searcher._extract_keywords = MagicMock(return_value=["個人事業主", "カード"])
searcher._calculate_keyword_similarity = MagicMock(return_value=0.5)
# After:
searcher._keyword_engine = MagicMock()
searcher._keyword_engine.extract_keywords = MagicMock(return_value=["個人事業主", "カード"])
searcher._keyword_engine.calculate_similarity = MagicMock(return_value=0.5)
```

L105:
```python
# Before:
searcher._extract_keywords = MagicMock(return_value=["個人事業主", "カード"])
# After:
searcher._keyword_engine.extract_keywords = MagicMock(return_value=["個人事業主", "カード"])
```

L120:
```python
# Before:
searcher._extract_keywords.assert_called()
# After:
searcher._keyword_engine.extract_keywords.assert_called()
```

L147:
```python
# Before:
searcher._calculate_keyword_similarity.assert_called_once_with(["振込"], "質問5")
# After:
searcher._keyword_engine.calculate_similarity.assert_called_once_with(["振込"], "質問5")
```

### Step 7: `_merge_results` O(N×M) → O(N+M) 最適化

**multi_stage_orchestrator.py (L279-294)**

Before:
```python
original_ids = {r['_doc_id'] for r in original_results}
llm_ids = {r['_doc_id'] for r in llm_results}
# ...
for doc_id in both_ids:
    orig_result = next((r for r in original_results if r.get('_doc_id') == doc_id), None)
    llm_result = next((r for r in llm_results if r.get('_doc_id') == doc_id), None)
```

After:
```python
orig_dict = {r['_doc_id']: r for r in original_results}
llm_dict = {r['_doc_id']: r for r in llm_results}

both_ids = set(orig_dict) & set(llm_dict)
original_only_ids = set(orig_dict) - set(llm_dict)
llm_only_ids = set(llm_dict) - set(orig_dict)
# ...
for doc_id in both_ids:
    orig_result = orig_dict.get(doc_id)
    llm_result = llm_dict.get(doc_id)
```

**search_strategy.py (L177-192)**

同パターン:
```python
orig_dict = {r['_doc_id']: r for r in original_results}
llm_dict = {r['_doc_id']: r for r in llm_results}

both_ids = set(orig_dict) & set(llm_dict)
original_only = set(orig_dict) - set(llm_dict)
llm_only = set(llm_dict) - set(orig_dict)
# ...
for doc_id in both_ids:
    orig = orig_dict.get(doc_id)
    llm = llm_dict.get(doc_id)
```

### Step 8: `Counter` 二重構築修正

**keyword_search_engine.py (L91-96)**

Before:
```python
filtered_words = {
    word: count
    for word, count in Counter(keywords).items()
    if word not in self.stop_words
}
return [word for word, _ in Counter(filtered_words).most_common(top_k)]
```

After:
```python
counter = Counter(keywords)
for stop_word in self.stop_words:
    del counter[stop_word]  # Counter は存在しないキーの del を許容（KeyError にならない）
return [word for word, _ in counter.most_common(top_k)]
```

### Step 9: テスト実行

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v
```

Expected: 全テスト PASS

### Step 10: コミット

```bash
git add src/core/searcher.py src/core/search/search_strategy.py src/core/search/multi_stage_orchestrator.py src/core/search/keyword_search_engine.py tests/unit/test_search_strategy.py tests/unit/test_keyword_similarity_sync.py
git commit -m "refactor: clean up Searcher keyword engine init, optimize _merge_results O(N+M)"
```

---

## Task 5: Phase 5 — 小改善

**Files:**
- Modify: `apps/revision-ops/ui/ops_ui.py:101-128`
- Modify: `config.py:150,202`
- Modify: `src/handlers/output_handler.py:30-34,120-124,248-252`

### Step 1: `build_scenario_id` 2関数を統合

**ops_ui.py (L101-128)**

Before（2関数）:
```python
def build_scenario_id(result: Dict) -> str:
    sheet_name = result.get("Sheet_Name", "")
    row_index = result.get("Row_Index", "")
    if not sheet_name or row_index == "":
        return ""
    try:
        excel_row = int(row_index) + 2
        bot_name = extract_bot_name_from_category(sheet_name)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""

def build_scenario_id_from_area(result: Dict, area: str) -> str:
    row_index = result.get("Row_Index", "")
    if row_index == "":
        return ""
    try:
        excel_row = int(row_index) + 2
        bot_name = extract_bot_name_from_area(area)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""
```

After（1関数）:
```python
def build_scenario_id(result: Dict, area: str = "") -> str:
    """検索結果からシナリオIDを構築

    Args:
        result: 検索結果 dict
        area: エリア名（指定時はsubstring-match、未指定時はSheet_Nameから辞書参照）
    """
    row_index = result.get("Row_Index", "")
    if row_index == "":
        return ""

    try:
        excel_row = int(row_index) + 2
        if area:
            bot_name = extract_bot_name_from_area(area)
        else:
            sheet_name = result.get("Sheet_Name", "")
            if not sheet_name:
                return ""
            bot_name = extract_bot_name_from_category(sheet_name)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""
```

旧 `build_scenario_id_from_area(result, area)` の呼び出し元を `build_scenario_id(result, area)` に変更。
ファイル内を grep して全呼び出しを更新。

### Step 2: `keyword_weight` を `@property` 化

**事前確認**: `keyword_weight` に値を代入するコードがないか確認:
```bash
cd /c/VSCode/rag/rag-local && grep -rn "keyword_weight\s*=" --include="*.py" | grep -v "\.venv" | grep -v "__pycache__"
```

`__post_init__` 内の `self.keyword_weight = 1.0 - self.vector_weight` 以外に代入がないことを確認。

**config.py**

L150 削除:
```python
# 削除:
keyword_weight: float = field(init=False)  # keyword_weight は vector_weight から自動計算
```

L202 削除:
```python
# 削除:
self.keyword_weight = 1.0 - self.vector_weight
```

`__post_init__` の後（クラス末尾付近）に @property を追加:
```python
    @property
    def keyword_weight(self) -> float:
        """vector_weight から自動計算（常に 1.0 - vector_weight）"""
        return 1.0 - self.vector_weight
```

**注意**: `dataclasses.asdict()` や `dataclasses.fields()` に `keyword_weight` が含まれなくなる。
これらを使用している箇所がないか事前に確認:
```bash
cd /c/VSCode/rag/rag-local && grep -rn "asdict\|dataclasses.fields" --include="*.py" | grep -v ".venv"
```

### Step 3: `_make_output_path()` ヘルパー抽出

**output_handler.py**

クラスにヘルパーメソッド追加:
```python
    def _make_output_path(self, mode: str) -> str:
        """タイムスタンプ付き出力ファイルパスを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"answer_{mode}_{timestamp}.xlsx")
```

3箇所を置換:

L30-34 → `output_file = self._make_output_path(mode)`
L120-124 → `output_file = self._make_output_path(mode)`
L248-252 → `output_file = self._make_output_path(mode)`

### Step 4: inline import 整理

**ops_ui.py** — 重複している `KeywordSearchEngine` の import を確認:

L145 と L183 で同じモジュールをインポート。L145 の `@st.cache_resource` 関数内は Streamlit のキャッシュ分離のため維持。
L183 も `@st.cache_resource` 内のため維持。

→ `@st.cache_resource` 内の import は Streamlit の仕様上そのまま維持する方が安全。
ただし `@st.cache_resource` でない通常関数内の inline import（あれば）はモジュールレベルに移動。

### Step 5: テスト実行

```bash
cd /c/VSCode/rag/rag-local && python -m pytest tests/ -v
```

Expected: 全テスト PASS

### Step 6: Streamlit UI 起動確認

```bash
cd /c/VSCode/rag/rag-local && python -c "from apps.revision_ops.ui.ops_ui import *; print('Import OK')" 2>&1 | head -5
```

import エラーがないことを確認。

### Step 7: コミット

```bash
git add apps/revision-ops/ui/ops_ui.py config.py src/handlers/output_handler.py
git commit -m "refactor: unify build_scenario_id, keyword_weight property, extract output path helper"
```

---

## 全Phase完了後のチェックリスト

- [ ] `pytest tests/ -v` 全テスト PASS
- [ ] Streamlit UI import エラーなし
- [ ] `git log --oneline -5` で5コミット確認
- [ ] 各コミットメッセージが Phase に対応
