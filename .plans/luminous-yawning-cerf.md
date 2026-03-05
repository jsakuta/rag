# ドキュメント修正計画: ユーザー変更可能設定の網羅的記載

## Context

引き継ぎ先が VertexAI デフォルト環境で使う想定に対し、ドキュメントに以下5点の記載漏れ・不整合がある。
設定を変更しようとしたユーザーが手順不明で詰まる、またはエラーの原因を特定できない問題を解消する。

## 修正対象ファイル

- `docs/ANSWER_SUPPORT.md` — #1, #4, #5
- `docs/CONFIGURATION.md` — #1, #2, #3, #4
- `docs/TROUBLESHOOTING.md` — #4

## 修正内容

### #1: search_source のバッチ処理変更方法

**ANSWER_SUPPORT.md L133 の後** — CLI引数テーブルの直後に Note 追加:
> バッチ処理の検索対象は CLI引数では変更不可。`config/settings.yaml` の `common.search_source`（`scenario` / `history_data`、デフォルト: `history_data`）を編集する。UIではサイドバーで動的切替可能。

**CONFIGURATION.md L273** — settings.yaml テーブルの common 行に `search_source` を追加。
**CONFIGURATION.md L283 の後** — 新サブセクション `### search_source（検索対象）` を挿入:
- 値の説明テーブル（history_data / scenario）
- 設定場所（settings.yaml）
- UI vs バッチでの挙動の違い

### #2: keyword設定の説明追加

**CONFIGURATION.md** — search_source サブセクションの後に `### keyword設定（キーワード検索パラメータ）` を挿入:
- `position_weight`（float, デフォルト 1.2）: テキスト前半マッチの重み係数
- `stop_words`（list, 13語）: 除外する一般語リスト
- 動作概要（Sudachi → 名詞抽出 → stop_words除外 → Top-5 → Jaccard類似度）

### #3: columns設定の説明追加

**CONFIGURATION.md** — keyword設定サブセクションの後に `### columns設定（Excel列名候補）` を挿入:
- query/answer/tag/correct_id 各候補リストの説明
- 必須/任意とエラー動作テーブル（query/answer: ValueError停止、tag: 警告続行、correct_id: スキップ）
- カスタマイズ例（候補リスト先頭に独自列名を追加）

> **注意**: input_handler.py L140-167 は列名候補をハードコードしており、settings.yaml の値を使っていない。ドキュメントでは settings.yaml を正とし、コード修正は別タスクとして扱う。

### #4: search_mode=llm_enhanced の前提条件

**ANSWER_SUPPORT.md L53 の後** — 検索モードテーブルの直後に Note 追加:
> `llm_enhanced` には LLM 環境変数（`DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL`, `GEMINI_PROJECT_ID` + GCP認証）が必須。未設定時は `RuntimeError` が発生。`original` モードでは LLM 不使用。

**CONFIGURATION.md L195** — LLM拡張検索モードの特徴の後に「前提条件」ブロック追加:
- 必要な環境変数一覧
- 未設定時のエラーメッセージ

**TROUBLESHOOTING.md L276 の後** — 新エントリ追加:
- 症状: `RuntimeError: LLM is not initialized`
- 原因: settings.yaml で llm_enhanced 設定済みだが LLM 環境変数未設定
- 解決策: original に戻す or LLM 環境変数を設定

### #5: ANSWER_SUPPORT.md の azure_openai 限定記載を修正

**L77-79**: 「azure_openai のみを使用する」→ プロバイダーテーブル（azure_openai / vertex_ai）に置換。build_db.py が両方構築する旨を追記。

**L83-92**: ディレクトリ構成図に `vertex_ai/` サブディレクトリを追加。実行時は `DEFAULT_EMBEDDING_PROVIDER` に一致するDBが使用される旨を追記。

**L256-259**: 前提条件の環境変数をプロバイダー別に整理（azure_openai: AOAI認証、vertex_ai: GCP認証）。build_db.py は両方構築するため両方の認証が必要、片方のみ使用する場合は省略可と注記。

## コミット

1コミット: `docs: add missing configuration guides (search_source, keyword, columns, LLM prereq, provider flexibility)`

## 検証

- 各ドキュメントの Markdown 構文が壊れていないこと（見出しレベル、テーブル整合）
- settings.yaml の実際の値とドキュメント記載のデフォルト値が一致すること
