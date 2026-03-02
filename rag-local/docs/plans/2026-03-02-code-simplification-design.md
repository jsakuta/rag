# rag-local コード簡素化設計書

**日付**: 2026-03-02
**対象**: rag-local（~12,600行/53ファイル）
**目的**: デッドコード削除、重複ロジック統合、効率改善。動作を壊さずにコードを簡素化する。

## レビュー結果サマリ

全体レビューで17件の指摘を検出。検証の結果14件が正しい指摘、3件が誤り/誇張と判定。

### 除外した3件
| 指摘 | 除外理由 |
|------|----------|
| `dual_provider_mode` 削除 | ops_ui.py で活発に使用中（False Positive） |
| サイドバーウィジェット重複 | 実質的な重複なし（誇張） |
| keyword検索フォーマッター統合 | 意図的な設計差異あり |

## 5 Phase 構成

### Phase 1: デッドコード・孤立参照の削除

**コミット**: `refactor: remove dead code and orphaned references`

| # | 修正 | ファイル | 変更内容 |
|---|------|----------|----------|
| 1 | db_version_manager import残骸 | `src/utils/__init__.py` | import文 + `__all__` から削除 |
| 2 | get_translator() 未使用 | `src/utils/business_area_translator.py` | `_default_translator` + `get_translator()` 削除 |
| 2 | export削除 | `src/utils/__init__.py` | `get_translator` のimport/export削除 |
| 9 | deprecated ラッパー | `src/core/searcher.py` | `_extract_keywords()`, `_calculate_keyword_similarity()` 削除 |
| 9 | orphaned tokenizer | `src/core/searcher.py` | `_shared_tokenizer`, `_tokenizer_lock`, `_get_shared_tokenizer()`, `self.tokenizer`, `self.mode` 削除 |
| 16 | ノーオプ三項演算子 | `apps/revision-ops/run_eval.py` | `value if value != "" else ""` → `value` |
| — | ドキュメント | `CLAUDE.md`, `docs/ARCHITECTURE.md` | `db_version_manager.py` 参照削除 |

**事前確認**: `_extract_keywords()` / `_calculate_keyword_similarity()` の呼び出し元を確認し、`KeywordSearchEngine` 直接アクセスに書き換え。

### Phase 2: 共通ユーティリティ構築

**コミット**: `refactor: add TextCombiner.build() and resolve_bot_name() utilities`

| # | 修正 | ファイル | 変更内容 |
|---|------|----------|----------|
| 3 | TextCombiner.build() | `src/core/search/text_combiner.py` | `build(hierarchy, query, answer) -> str` メソッド追加 |
| 4 | resolve_bot_name() | `src/utils/business_area_translator.py` | `resolve_bot_name(area, area_to_bot) -> str` 関数追加 |

```python
# TextCombiner.build()
def build(self, hierarchy: str = "", query: str = "", answer: str = "") -> str:
    parts = []
    if hierarchy and hierarchy.strip():
        parts.append(f"{self.LABEL_HIERARCHY}: {hierarchy}")
    if query and query.strip():
        parts.append(f"{self.LABEL_QUERY}: {query}")
    if answer and answer.strip():
        parts.append(f"{self.LABEL_ANSWER}: {answer}")
    return self.separator.join(parts)

# resolve_bot_name()
def resolve_bot_name(area: str, area_to_bot: dict) -> str:
    area_lower = area.lower()
    for keyword, bot_name in area_to_bot.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"
```

### Phase 3: 重複ロジック統合 + バグ修正

**コミット**: `refactor: consolidate duplicated logic and fix parse off-by-one bug`

| # | 修正 | ファイル | 変更内容 |
|---|------|----------|----------|
| 5 | combined_text 5箇所 | `input_handler.py`, `dynamic_db_manager.py` | 手書き構築 → `get_text_combiner().build()` |
| 6 | bot_name 3箇所 | `ops_ui.py`, `run_eval.py`, `chromadb_keyword_search.py` | ローカル関数 → `resolve_bot_name()` |
| 7 | parse バグ修正 | `searcher.py` | `parse_enhanced_combined_text()` 削除 |
| 7 | 呼び出し元更新 | `search_strategy.py` | `get_text_combiner().parse()` に変更 |
| 8 | load_data 基底クラス化 | `input_handler.py` | `MultiFolderInputHandler.load_data()` 削除 |

**バグ修正**: `part[3:]` → `TextCombiner.parse()` の `part[len("分類: "):]` で正しくパース。

### Phase 4: パフォーマンス最適化

**コミット**: `refactor: optimize _merge_results to O(N+M) and fix double Counter`

| # | 修正 | ファイル | 変更内容 |
|---|------|----------|----------|
| 10 | _merge_results | `multi_stage_orchestrator.py` | `next()` 線形探索 → dict lookup |
| 10 | _merge_results | `search_strategy.py` | 同パターン → dict lookup |
| 15 | Counter 二重構築 | `keyword_search_engine.py` | 1回目の Counter でフィルタ+most_common |

### Phase 5: 小改善

**コミット**: `refactor: unify build_scenario_id, clean imports, extract output path helper`

| # | 修正 | ファイル | 変更内容 |
|---|------|----------|----------|
| 11 | build_scenario_id | `ops_ui.py` | 2関数 → `build_scenario_id(result, area=None)` |
| 12 | inline import整理 | `ops_ui.py` | 重複importをモジュールレベルに移動 |
| 13 | keyword_weight @property | `config.py` | `field(init=False)` → `@property` |
| 14 | _make_output_path | `output_handler.py` | 3重複 → ヘルパーメソッド |

## 検証戦略

| タイミング | 方法 |
|------------|------|
| Phase 1-4 各完了後 | `pytest tests/` 全テスト通過 |
| Phase 5 完了後 | `pytest tests/` + `streamlit run` でUI起動確認 |

## スコープ外

| 項目 | 理由 |
|------|------|
| `dual_provider_mode` 削除 | 実際に使用中 |
| サイドバーウィジェット統合 | 実質的重複なし |
| keyword検索フォーマッター統合 | 意図的差異 |
| `MultiStageSearchStrategy` → Orchestrator委譲 | 構造差異大 |
| プロンプト読み込み + LLMリトライ共通化 | 統合の価値 < リスク |
