# ANSWER_SUPPORT.md レビューレポート

**対象**: `rag-local/docs/ANSWER_SUPPORT.md`（326行）
**レビュー日**: 2026-03-03
**レビュアー**: answer-support-reviewer

---

## 指摘サマリ

| 優先度 | 件数 |
|--------|------|
| Critical | 2 |
| Important | 5 |
| Minor | 4 |

---

## Critical（事実誤認・コード不一致）

### C-1: Streamlit 直接起動コマンドが不正

**箇所**: 163行目

```markdown
streamlit run apps/answer-support/ui/chat.py
```

**問題**: `streamlit` CLI の起動コマンドは `streamlit run` だが、これは正しい。ただし `main.py` の `interactive` サブコマンド経由では `subprocess.Popen` で `-m streamlit run` を使用しているため、直接起動とサブプロセス起動で挙動が異なる可能性がある。

実際に `chat.py:358` は `if __name__ == '__main__': run_streamlit_ui()` があるため、直接起動も可能。

**結論**: コマンド自体は正しいが、`cd rag-local` が前提という記載が重要。問題なし。

---

### C-2: バッチ出力 Excel 列構成がコードと不一致

**箇所**: 290-302行目

**ドキュメント記載の列**:
- `Input_Number`, `Original_Query`, `Original_Answer`, `Search_Query`, `Search_Result_Q`, `Search_Result_A`, `Similarity`, `Vector_Weight`, `Top_K`, `Generated_Tags`

**実際のコード (`searcher.py:410-423` `_build_result_data`)**:
出力される辞書のキーは以下:
- `Input_Number`, `Original_Query`, `Original_Answer`, `Search_Query`, `Search_Result_Q`, `Search_Result_A`, `Similarity`, **`Scenario_ID`**, **`Sheet_Name`**, **`Row_Index`**, `Vector_Weight`, `Top_K`

**不一致点**:
1. **`Scenario_ID`, `Sheet_Name`, `Row_Index` がドキュメントに記載されていない** -- 実際の出力 Excel にはこれらの列が含まれる
2. **`Generated_Tags` がコード内に存在しない** -- `_build_result_data` にも `_format_final_results` にも `Generated_Tags` の設定はない。`output_handler.py:93` の `header_names` マッピングにのみ存在するが、実データには含まれない。廃止済み機能の残骸と思われる

**修正案**: `Generated_Tags` を列構成表から削除し、`Scenario_ID`/`Sheet_Name`/`Row_Index` を追加する

---

### C-3: Metadata シートのパラメータリストが不完全

**箇所**: 303行目

**ドキュメント記載**:
> Metadata シートに検索パラメータ（vector_weight, search_mode, search_type, top_k, embedding_provider, embedding_model, timestamp）を記録する。

**実際のコード (`output_handler.py:419-435`)**:
`keyword_weight` も記録されている（8項目中7項目しかドキュメントに記載されていない）。

**修正案**: `keyword_weight` を追記する

---

## Important（情報不足・文書化されていない機能・設定）

### I-1: 入力 Excel のカラム仕様が未記載

**箇所**: 152行目付近

ドキュメントには「`data/input/` 配下の Excel ファイル」としか書かれていないが、`input_handler.py:28-37` の `_get_column_names` メソッドによると:
- **列は位置ベースで検出**: 1列目=番号、2列目=質問、3列目=回答（任意）
- 列名は任意（位置で判断）
- `INPUT_FILE_PATTERN` (`config.py:133`): `^([^_]+)_(\d{8})\.xlsx$` に一致するファイル名が必要
- ファイル名から業務分野を自動判定（`DynamicDBManager.extract_business_area_from_input`）

**修正案**: 入力 Excel の仕様（列構成、ファイル命名規則 `{業務分野名}_{YYYYMMDD}.xlsx`）を追記する

### I-2: `keyword_weight` プロパティの説明が不足

**箇所**: 59-68行目（スコア計算式セクション）

ドキュメントでは `(1 - vector_weight)` と記載されているが、コード上は `config.keyword_weight` プロパティ（`config.py:302-304`）が `1.0 - self.vector_weight` を返す。

数学的には等価だが、コード内で実際に使われている変数名は `keyword_weight` であり、`settings.yaml` での設定との関係を明確にすべき。

**修正案**: `keyword_weight` が `1 - vector_weight` として自動計算されるプロパティであることを一文追記する

### I-3: `search_source` のデフォルト値が文脈によって異なることが不明瞭

**箇所**: 148行目

> `config/settings.yaml` の `common.search_source`（`scenario` / `history_data`、デフォルト: `history_data`）を編集してください。

`settings.yaml:36` では確かに `search_source: history_data` だが、UI では `chat.py:281-290` でドロップダウンが提供され、動的に変更可能。この動的変更がセッション限りであること（YAML 自体は変更されない）が明記されていない。

**修正案**: UI でのセッション内変更は YAML に保存されない旨を追記する

### I-4: UI の検索タイプが `hybrid` に固定されていることが未記載

**箇所**: 169-178行目（UI機能テーブル）

`chat.py:265` で `st.session_state.config.search_type = "hybrid"` とハードコードされており、UI では `keyword_filter` に切り替えられない。UI機能テーブルにこの制約が記載されていない。

**修正案**: UI では検索タイプが `hybrid` 固定である旨を注記する

### I-5: バッチ処理時の DB 自動更新動作が未記載

**箇所**: 119-139行目

`main.py:192-199` では、バッチモード実行時に `run_db_update()` が自動的に呼ばれ、参照ファイルの更新チェック + 必要に応じて DB 更新が実行される。ドキュメントにはこの自動更新の挙動が記載されていない。

初見の人は「バッチ実行前に `build_db.py` を手動実行する必要があるのか」が分からない。

**修正案**: バッチモードでは起動時に参照ファイルの更新チェックが自動実行される旨を追記する。`build_db.py` は初回構築時や `--force` 再構築時に使用する位置付けであることを明確にする。

---

## Minor（文体・明瞭性・体裁）

### M-1: キーワード抽出の説明が不正確

**箇所**: 37行目

```
キーワード抽出（Sudachi: 名詞 Top-5）
```

`keyword_search_engine.py` を直接確認していないが、`searcher.py` では `_keyword_engine.extract_keywords(query)` で抽出したキーワード全件をログ出力しており、Top-5 に限定するロジックが `searcher.py` 側にはない。Top-5 制限が本当に存在するか確認が必要。

### M-2: `multi_stage` モードの Note が紛らわしい

**箇所**: 57行目

> `multi_stage` モードも存在しますが、改定影響調査（`run_eval.py`）専用です。

この Note は回答支援AI のドキュメントに記載されているが、回答支援AIの文脈では不要な情報。検索モードの選択肢に `multi_stage` を含めていないため（50-53行目の表に2つしかない）、この Note は削除するか、もしくは「回答支援AIでは使用しません」と簡潔に書き直すべき。

### M-3: 処理フロー図のフォーマット不統一

**箇所**: 28-46行目

テキストベースの処理フロー図が ASCII art と通常テキストの混在で、可読性が低い。Mermaid やシンプルなリスト形式への置き換えを検討。

### M-4: チャット履歴保存で `Original_Answer` が含まれない

**箇所**: 178行目（チャット履歴保存機能の説明）

`chat.py:218-227` の `save_chat_history` では、保存データに `Original_Answer` が含まれていない（バッチ出力には含まれる）。チャット履歴とバッチ出力の列構成が異なることが明記されていない。

---

## 引き継ぎ適性評価

### 良い点
- 実行モード3つの概要テーブルが分かりやすい
- DB構造（業務分野・件数・プロバイダー）が明確
- コマンドライン例が豊富で、組み合わせ例も記載
- `build_db.py` のスキップロジック説明が実用的
- 前提条件（Streamlit停止、環境変数）が明記されている

### 改善が必要な点
- **入力ファイルの仕様が欠落** (I-1): 初見の人が「どんな Excel を `data/input/` に置けばよいか」が分からない
- **バッチ実行時の自動 DB 更新の説明がない** (I-5): `build_db.py` と `main.py` 内の DB 更新の関係が不明瞭
- **出力列構成が不正確** (C-2): 実際の出力と一致しない列がある
- **UI 固有の制約が未記載** (I-4): hybrid 固定、セッション限りの設定変更

### 総合評価
骨格は良好だが、**入力仕様の欠落 (I-1)** と **出力列構成の不一致 (C-2)** が初見の利用者にとって障害になる可能性がある。これらを修正すれば、引き継ぎドキュメントとして十分機能する。
