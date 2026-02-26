# eval_ui.py 2モード再構成 + レビュー指摘修正 設計書

## 概要

`eval_ui.py` を「評価モード」と「影響調査モード」の2モード構成に再設計する。
併せて、直近3コミット（`e8b766a`, `87ff21e`, `3f5c842`）のコードレビュー指摘7件を修正する。

## 背景

- 現在の `eval_ui.py` は評価モードのみで、影響調査が検索タイプの1つとして混在
- ユーザーは「評価モード」と「影響調査モード」を明確に切り替えたい
- 影響調査モードでは シナリオ/FAQ のデータソース選択が必要
- 両モードとも keyword_filter（キーワード検索）と hybrid（意味検索）の両方が使える

## 2モード定義

| 項目 | 評価モード | 影響調査モード |
|------|-----------|---------------|
| 目的 | 改定1〜6の正解ID精度検証 | 通常業務データから影響候補を検索 |
| データソース | `rev*` コレクション（改定別） | `naibujimu`, `smile` コレクション |
| 検索タイプ | hybrid / keyword_filter（改定ごとに設定可） | hybrid / keyword_filter |
| データフィルタ | なし（revコレクションは全てシナリオ） | シナリオ / FAQ / 両方（`metadata.source`） |
| カテゴリ選択 | 改定番号で自動決定 | 内部事務 / スマイル / 全て |
| 結果表示 | Azure/VertexAI 2タブ + 正解IDバッジ | 単一リスト（正解判定なし） |
| 正解ID | 必要（multi_stage_input.xlsx） | 不要 |
| 改定番号選択 | 必須（①〜⑥） | 不要 |
| Teams Bot対応 | — | FR-003（意味検索）/ FR-004（キーワード）のローカル版 |

## サイドバー構成

```
[モード選択] ← 最上部 st.radio
  (*) 評価モード
  ( ) 影響調査モード

=== 評価モード時 ===
[改定番号] ①〜⑥  (st.selectbox)
  正解ID: XX件
  対象エリア: rev02_souzoku
---
[検索タイプ] hybrid / keyword_filter  (st.radio)
[設定] vector_weight, top_k (hybrid時)
       "キーワード検索: 全件返却" (keyword_filter時)

=== 影響調査モード時 ===
[カテゴリ] 内部事務 / スマイル / 全て  (st.radio)
[データソース] シナリオ / FAQ / 両方  (st.radio)
---
[検索タイプ] hybrid / keyword_filter  (st.radio)
[設定] vector_weight, top_k (hybrid時)
       "キーワード検索: 全件返却" (keyword_filter時)
```

## 検索パイプライン

### 評価モード — hybrid
既存のまま: `_search_with_provider()` → `MultiStageOrchestrator` → rev* コレクション
Azure/VertexAI 2プロバイダー並列

### 評価モード — keyword_filter
既存のまま: `ChromaDBKeywordSearcher.search()` → rev* コレクション全件取得 → キーワードマッチ

### 影響調査モード — hybrid（新規）
`_search_with_provider()` に naibujimu/smile を渡す。
`MultiStageOrchestrator` はコレクション名を引数に取るので、rev* → naibujimu/smile に差し替えるだけ。
source_filter は `_search_with_provider` の結果を Python 側でフィルタ。

### 影響調査モード — keyword_filter
`ChromaDBKeywordSearcher.search()` に naibujimu/smile を渡す。
source_filter パラメータ追加で `metadata.source` フィルタ。

## ChromaDBKeywordSearcher 変更

```python
def search(
    self,
    collection_names: List[str],
    query: str,
    provider: str = "azure_openai",
    max_results: int = 50,
    source_filter: Optional[str] = None,  # 追加: "scenario" | "history_data" | None
) -> List[MatchResult]:
```

`_search_collection` 内で全件取得後、`source_filter` が指定されていれば
`meta.get("source") == source_filter` でフィルタしてからキーワードマッチング。

## レビュー指摘修正（7件）

### BUG-1 (High): キーワード2重抽出
- `evaluate_revisions.py:278-279` の `keyword_engine.extract_keywords(query)` を削除
- searcher 内部で1回だけ呼ぶ
- ログ出力も searcher に任せる

### BUG-2 (High): 例外の握りつぶしが広すぎる
- `chromadb_keyword_search.py:114` の `except (ChromaNotFoundError, Exception)` を限定
- `ChromaNotFoundError` → warning + return []
- `ValueError`, `FileNotFoundError` → warning + return []
- その他 → `logger.error` + raise

### BUG-3 (Medium): 全件取得のパフォーマンス
- `_search_collection` の結果をインスタンスレベルでキャッシュ
- `Dict[Tuple[collection_name, provider], Tuple[docs, metadatas]]`
- 同一インスタンスで複数回 search() を呼ぶケースに有効

### BUG-4 (Medium): session_state 初期化漏れ
- `initialize_session_state()` に以下を追加:
  - `app_mode`: "evaluation" | "impact_analysis"
  - `impact_categories`: ["naibujimu", "smile"]
  - `impact_source_filter`: None (全て)

### BUG-5 (Medium): O(N×M) の area フィルタリング
- `evaluate_revisions.py:295-297` を `defaultdict(list)` でグルーピング
- 1回のループで area 別に分類してから処理

### BUG-6 (Low): private static method の外部参照
- `_extract_area` → `extract_area`（public化）
- `MatchResult` に `area` プロパティを追加（`extract_area(self.collection_name)`）

### BUG-7 (Low): 無意味なテスト
- `test_match_result_has_required_fields` を `test_source_field_values` に変更
- `source` が "scenario" | "history_data" | "unknown" のいずれかであることを検証

## 影響範囲

| ファイル | 変更内容 |
|---------|---------|
| `src/core/search/chromadb_keyword_search.py` | source_filter追加、例外処理修正、キャッシュ、extract_area public化 |
| `apps/revision-eval/ui/eval_ui.py` | 2モード構成、サイドバー再設計、影響調査hybrid対応 |
| `apps/revision-eval/evaluate_revisions.py` | BUG-1(2重抽出), BUG-5(O(N×M)) 修正 |
| `tests/unit/test_chromadb_keyword_search.py` | source_filterテスト追加、BUG-7テスト修正 |

## 非変更

- `apps/answer-support/ui/chat.py` — 回答支援AIには影響なし
- `evaluate_revisions.py` のバッチ版 — 影響調査モードは追加しない（UI専用）
- `settings.yaml` — 既存の revision_areas 設定はそのまま維持
- DB再構築 — 不要（既存の metadata.source でフィルタ可能）
