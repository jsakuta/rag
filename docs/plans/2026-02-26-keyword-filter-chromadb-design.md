# keyword_filter ChromaDB化 + 影響調査モード追加 設計書

## 概要

事務改定評価AIの keyword_filter 検索を Excel 直読みから ChromaDB に移行し、
影響調査モード（通常業務データの横断キーワード検索）を新規追加する。

## 背景・動機

- `evaluate_revisions.py`（バッチ版）と `eval_ui.py`（UI版）の keyword_filter が
  同じ Excel 直読みロジックを重複実装している
- ChromaDB に `rev*` コレクションとしてデータが構築済みであるにもかかわらず未活用
- 通常業務データ（naibujimu, smile）の横断キーワード検索ニーズがある

## 要件（確定済み）

| 項目 | 決定 |
|------|------|
| 対象ファイル | `evaluate_revisions.py:285-380` + `eval_ui.py:175-259` |
| 評価モード | `rev*` コレクションからキーワード検索 |
| 影響調査モード（新規） | `naibujimu`, `smile` コレクションから FAQ+シナリオ横断検索 |
| マッチング方式 | `collection.get()` 全件取得 → Python側キーワードマッチ |
| シナリオID | `row_index + 2`（Excel行番号ベース）維持 |
| タグなし行 | ChromaDB未格納のため除外（正しい挙動） |
| 回答支援AIとの関係 | 干渉なし（回答支援AIはベクトル検索 + リランキング、本モジュールは全件キーワードマッチ） |

## アーキテクチャ

### アプローチ: 共通モジュール抽出（アプローチA）

`src/core/search/chromadb_keyword_search.py` を新設し、バッチ版・UI版の両方から呼び出す。

```
src/core/search/chromadb_keyword_search.py  (新規)
├── ChromaDBKeywordSearcher
│   ├── __init__(db_path, keyword_engine)
│   ├── search(collection_names, query, max_results) → List[Dict]
│   └── _format_result(doc, metadata, match_count, ...) → Dict
│
evaluate_revisions.py  (変更)
├── _execute_keyword_filter_search() → ChromaDBKeywordSearcher.search() 呼び出し
└── _load_scenario_excel() → 削除
│
eval_ui.py  (変更)
├── _execute_keyword_filter_search() → ChromaDBKeywordSearcher.search() 呼び出し
└── 影響調査モードUI追加
```

### 依存関係

```
chromadb_keyword_search.py
  ├── chromadb (PersistentClient)
  └── keyword_search_engine.py (extract_keywords のみ)
```

- `MetadataVectorDB` の LRU キャッシュ付きクライアントは使わない
  （検索頻度が低く、回答支援AI用キャッシュと混在させたくないため）
- `KeywordSearchEngine.extract_keywords()` はキーワード抽出に再利用

## コンポーネント設計

### ChromaDBKeywordSearcher

```python
class ChromaDBKeywordSearcher:
    """ChromaDB全件取得 + キーワードマッチング"""

    def __init__(self, db_path: str, keyword_engine: KeywordSearchEngine):
        self.client = chromadb.PersistentClient(path=db_path)
        self.keyword_engine = keyword_engine

    def search(
        self,
        collection_names: List[str],
        query: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        1. keyword_engine.extract_keywords(query) でキーワード抽出
        2. 各コレクションで collection.get() 全件取得（documents + metadatas）
        3. 各ドキュメントに対してキーワードマッチ数カウント（大文字小文字無視）
        4. マッチ数 > 0 を降順ソート → max_results 件に制限
        5. 統一フォーマットで返却
        """

    def _format_result(
        self,
        doc: str,
        metadata: dict,
        match_count: int,
        total_keywords: int,
        collection_name: str
    ) -> Dict:
        """
        返却フォーマット（既存の結果辞書と互換）:
        {
            "Similarity": match_count / total_keywords,
            "Search_Result_Q": document から質問部分をパース,
            "Search_Result_A": document から回答部分をパース,
            "Search_Category": "Keyword",
            "Sheet_Name": metadata["sheet_name"],
            "Row_Index": metadata["row_index"],
            "Scenario_ID": f"{bot_name}_{row_index + 2}",
            "_area": collection_name から area を逆引き,
        }
        """
```

### ChromaDB データ構造（既存）

- **document**: `分類: {hierarchy} | 質問: {query} | 回答: {answer}`
- **metadata**: `source`, `sheet_name`, `row_index`, `hierarchy`
  （`input_handler.py:350-358` で定義）

### document パース

`|` で分割し、`質問:` / `回答:` プレフィックスで質問・回答を抽出。
`分類:` 部分は hierarchy として利用可能。

## 呼び出し側の変更

### evaluate_revisions.py（バッチ版）

- `_execute_keyword_filter_search()`: Excel 読み込みを `ChromaDBKeywordSearcher.search()` に置換
- `_load_scenario_excel()`: 削除
- 戻り値の形式は変更なし（下流の正解ID照合・Excel出力に影響なし）

### eval_ui.py（UI版）

- `_execute_keyword_filter_search()`: 同様に置換
- `SCENARIO_DIR` 定数削除
- 検索タイプに `"impact_analysis"` を追加

## 影響調査モード（新機能）

### UI変更

```python
# 検索タイプ選択
search_type = ["multi_stage", "keyword_filter", "impact_analysis"]
```

**影響調査モード選択時**:
- 改定（revision）選択は不要
- カテゴリ選択:「全て」「内部事務」「スマイル」
- コレクション名: `["naibujimu"]`, `["smile"]`, `["naibujimu", "smile"]`

### バッチ版

影響調査モードは **UI専用**。バッチ版には追加しない。
（バッチ版は正解ID付きデータの評価用途、影響調査は対話的探索用途）

### コレクション名の解決

| モード | コレクション名 |
|--------|---------------|
| 評価（keyword_filter） | `rev02_souzoku`, `rev03_smile` 等 |
| 影響調査（impact_analysis） | `naibujimu`, `smile` |

## エラーハンドリング

| ケース | 挙動 |
|--------|------|
| コレクション未構築 | `ChromaNotFoundError` → 警告ログ + スキップ（例外にしない） |
| キーワード抽出結果が空 | 空リスト返却 |
| `collection.get()` が空 | 次のコレクションへ continue |
| DB破損等の致命的エラー | 例外を上げる（呼び出し側で try/except 済み） |

## テスト計画

`tests/unit/test_chromadb_keyword_search.py` を新設。
ChromaDB は in-memory client（`chromadb.Client()`）でテストデータ投入。

| テストケース | 内容 |
|-------------|------|
| 単一コレクション検索 | 1コレクション、キーワードヒットあり |
| 複数コレクション横断 | 2コレクション、マッチ数降順ソート確認 |
| キーワードなし | 空クエリ → 空リスト |
| コレクション未存在 | 警告ログ出力、例外なし |
| Scenario ID 互換性 | `row_index + 2` が正しく生成されること |
| 結果フォーマット互換 | 既存の辞書キーがすべて含まれること |

## 削除対象

- `evaluate_revisions.py`: `_load_scenario_excel()` メソッド
- `eval_ui.py`: `SCENARIO_DIR` 定数、Excel glob/read ロジック
- 両ファイル: Excel 直読み用の pandas 関連コード（import は別用途で残る）
