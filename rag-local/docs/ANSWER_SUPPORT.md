# 回答支援AI（類似回答検索）

FAQ およびシナリオデータから類似回答を検索するシステム。

## 概要

### 目的

- ユーザーの質問に対して、既存の FAQ やシナリオボットのデータから類似度の高い質問・回答ペアを検索
- 業務分野ごとに独立した ChromaDB ベクトルデータベースを使用し、ハイブリッド検索で精度を確保

### 実行モード

| モード | 用途 | エントリーポイント |
|--------|------|-------------------|
| バッチ | Excel入力 → 一括検索 → Excel出力 | `apps/answer-support/main.py` |
| Streamlit UI | 対話形式でリアルタイム検索 | `apps/answer-support/ui/chat.py` |
| プレフライト | DB更新可否の事前検証（本番更新なし） | `apps/answer-support/main.py preflight` |

---

## 処理フロー

### ハイブリッド検索

ベクトル検索（意味的類似度）とキーワード検索（語彙的類似度）を加重合算する方式。

```
入力クエリ
    |
    +---> ベクトル検索（埋め込みモデルでクエリをベクトル化 → ChromaDB で類似度検索）
    |         |
    |         v
    |     vector_similarity（コサイン類似度）
    |
    +---> キーワード抽出（Sudachi: 名詞 Top-5）
              |
              v
          keyword_similarity（Jaccard 類似度: 共通キーワード / 全キーワード）
              |
              v
    スコア合算: final_score = vector_weight x vector_sim + (1 - vector_weight) x keyword_sim
              |
              v
    スコア降順でソート → top_k 件を返却
```

### 検索モード

| モード | 説明 |
|--------|------|
| `original` | 原文をそのままクエリとしてハイブリッド検索（デフォルト） |
| `llm_enhanced` | LLM でクエリを拡張・要約してからハイブリッド検索 |

> **Note:** `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL` は全モードで起動時に必須です（`SearchConfig` のバリデーション）。ただし、LLM API の呼び出しが発生するのは `llm_enhanced` モードのみです。`GEMINI_PROJECT_ID` + GCP認証も `llm_enhanced` 使用時に必須となります。

> **Note:** `multi_stage` モードも存在しますが、改定影響調査（`run_eval.py`）専用です。回答支援AIでは `original` または `llm_enhanced` を使用してください。

### スコア計算式

```
final_score = vector_weight x vector_similarity + (1 - vector_weight) x keyword_similarity
```

- `vector_weight`: デフォルト 0.9（`settings.yaml` で変更可能、UI ではスライダーで動的調整）
- `vector_similarity`: 埋め込みモデルによるコサイン類似度（0.0 - 1.0）
- `keyword_similarity`: Jaccard 類似度（Sudachi で抽出したキーワード集合の共通割合）
- 最終スコアは 0.0 - 1.0 にクリップ

---

## DB構造

### 業務分野

| 業務分野 | DB名 | 内容 | 件数 |
|---------|------|------|------|
| 内部事務 | `naibujimu` | 預金+総則 FAQ + naibujimu-bot シナリオ | 11,439件 |
| スマイル | `smile` | スマイル FAQ + smile-bot シナリオ | 9,237件 |

### 埋め込みプロバイダー

| プロバイダー | モデル | 次元数 | 備考 |
|-------------|--------|--------|------|
| `azure_openai` | text-embedding-3-large | 3072 | エンタープライズ向け・高精度 |
| `vertex_ai` | gemini-embedding-001 | 3072 | Google Cloud 統合・MRL対応 |

`build_db.py` は両プロバイダーの DB を構築する。実行時は `DEFAULT_EMBEDDING_PROVIDER` 環境変数に一致するプロバイダーの DB が使用される。

### ディレクトリ構成

```
data/
├── vector_db/
│   ├── update_timestamps.json        # 更新日時記録
│   ├── naibujimu/                    # 内部事務 DB
│   │   ├── azure_openai/
│   │   │   └── chroma.sqlite3
│   │   └── vertex_ai/
│   │       └── chroma.sqlite3
│   └── smile/                        # スマイル DB
│       ├── azure_openai/
│       │   └── chroma.sqlite3
│       └── vertex_ai/
│           └── chroma.sqlite3
├── source/
│   ├── faq/latest/                   # FAQ（履歴データ）
│   │   ├── 内部事務_履歴データ_YYYYMMDD.xlsx
│   │   └── スマイル_履歴データ_YYYYMMDD.xlsx
│   └── scenarios/latest/             # 最新シナリオ
│       ├── 内部事務_シナリオデータ_YYYYMMDD.xlsx
│       └── スマイル_シナリオデータ_YYYYMMDD.xlsx
├── input/                            # バッチ入力 Excel
└── output/latest/answer/             # バッチ出力 Excel
```

---

## 使用方法

### バッチ処理

入力 Excel を一括処理して結果を Excel に出力する。

```bash
cd rag-local

# 全業務分野でバッチ処理
python apps/answer-support/main.py

# 特定業務分野のみ（naibujimu / smile）
python apps/answer-support/main.py --business naibujimu

# 処理件数を制限（先頭 N 件のみ）
python apps/answer-support/main.py --limit 5

# 組み合わせ
python apps/answer-support/main.py --business naibujimu --limit 10
```

#### コマンドライン引数

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--business` | 対象の業務分野（`naibujimu`, `smile`） | 全業務分野 |
| `--limit` | 処理する入力データの件数上限 | 無制限 |

> **Note:** バッチ処理の検索対象は CLI 引数では変更できません。`config/settings.yaml` の `common.search_source`（`scenario` / `history_data`、デフォルト: `history_data`）を編集してください。UI ではサイドバーで動的に切替可能です。

#### 入出力

- **入力**: `data/input/` 配下の Excel ファイル
- **出力**: `data/output/latest/answer/answer_batch_YYYYMMDD_HHMMSS.xlsx`

### Streamlit UI

対話形式で検索を実行する。

```bash
cd rag-local

# 直接起動
streamlit run apps/answer-support/ui/chat.py

# main.py 経由で起動
python apps/answer-support/main.py interactive
```

#### UI機能

| 機能 | 説明 |
|------|------|
| 業務分野選択 | DB に存在する業務分野をプルダウンで選択 |
| ベクトル重み調整 | スライダーで vector_weight を 0.0 - 1.0 の範囲で動的変更 |
| 検索モード切替 | 原文検索（original） / LLMクエリ検索（llm_enhanced） |
| 検索対象切替 | シナリオのみ（scenario） / FAQのみ（history_data） |
| 候補数設定 | 表示する類似候補数（1 - 10） |
| チャット履歴保存 | チャット履歴を Excel ファイルとして保存 |

### プレフライト検証

DB更新の事前検証を実行する（本番のDB更新は行わない）。

```bash
cd rag-local

# 全業務分野の検証
python apps/answer-support/main.py preflight

# 特定業務分野のみ（naibujimu / smile）
python apps/answer-support/main.py preflight --business naibujimu

# サンプル件数を変更（デフォルト: 5）
python apps/answer-support/main.py preflight --sample-size 10
```

#### プレフライト引数

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--business` | 対象の業務分野（`naibujimu`, `smile`） | 全業務分野 |
| `--sample-size` | 検証に使うサンプル件数 | 5 |

---

## DB構築

統合スクリプト `scripts/build_db.py` で回答支援AI用 DB を構築する。

### コマンド

```bash
cd rag-local

# 差分のみ構築（未構築 or 参照ファイル更新時のみ実行）
python scripts/build_db.py

# 既存DB削除して全再構築
python scripts/build_db.py --force

# 指定業務分野のみ
python scripts/build_db.py --business naibujimu
python scripts/build_db.py --business smile

# 回答支援AI用DBのみ（改定別を除外）
python scripts/build_db.py --no-revisions
```

### build_db.py 引数

| 引数 | 説明 |
|------|------|
| `--force` | 既存DBを削除して全再構築 |
| `--business <名前>` | 構築対象の業務分野（例: `naibujimu`, `smile`） |
| `--no-revisions` | 通常業務のみ構築（改定別 `rev*` を除外） |
| `--revisions-only` | 改定別のみ構築（回答支援AI用を除外） |

`--no-revisions` と `--revisions-only` は排他（同時指定不可）。

### スキップロジック（デフォルト動作）

- DB存在 + ドキュメント数 > 0 + 参照ファイル未更新 --> スキップ（APIコスト発生なし）
- DB未存在 or 参照ファイル更新あり --> 構築/更新実行
- `--force` 指定時のみ既存DB削除 --> 全再構築

### 参照データの配置

```
data/source/
├── faq/latest/                # FAQ（履歴データ）
│   ├── 内部事務_履歴データ_YYYYMMDD.xlsx
│   └── スマイル_履歴データ_YYYYMMDD.xlsx
└── scenarios/latest/          # 最新シナリオ
    ├── 内部事務_シナリオデータ_YYYYMMDD.xlsx
    └── スマイル_シナリオデータ_YYYYMMDD.xlsx
```

### ファイル命名規則

```
{業務名}_履歴データ_{YYYYMMDD}.xlsx     # FAQ
{業務名}_シナリオデータ_{YYYYMMDD}.xlsx  # シナリオ
```

- `{業務名}`: 日本語名（スマイル、内部事務等）。`config/business_areas.yaml` のマッピングで英語DB名に自動変換
- `{YYYYMMDD}`: データ日付。同一業務分野に複数ファイルがある場合、最新日付のファイルが使用される

### 前提条件

1. **Streamlit UI の停止**: ChromaDB はファイルロックを使用するため、UI と同時にスクリプトを実行するとロックエラーが発生する
2. **環境変数の設定**: `.env` に以下が必要
   - `DEFAULT_EMBEDDING_PROVIDER`（モデルはプロバイダーから自動解決）
   - **azure_openai 使用時**: `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - **vertex_ai 使用時**: `GEMINI_PROJECT_ID` + GCP認証（`gemini_credentials.json` または Key Vault）
   - `build_db.py` は両プロバイダー（azure_openai / vertex_ai）の DB を常に構築する。未認証のプロバイダー側はエラーになるが、認証済み側の DB は正常に構築されるため、片方のみ使用する場合は未認証側のエラーを無視してよい
   - `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL`（起動時バリデーションで必須。LLM API 呼び出しは `llm_enhanced` モードのみ）
3. **参照データの配置**: `data/source/faq/latest/` と `data/source/scenarios/latest/` に対象 Excel ファイルが必要

---

## 出力ファイル

### 出力先

- バッチ: `data/output/latest/answer/answer_batch_YYYYMMDD_HHMMSS.xlsx`
- チャット履歴: `data/output/latest/answer/answer_chat_YYYYMMDD_HHMMSS.xlsx`

### バッチ出力 Excel 列構成

| 列名（内部キー） | 表示名 | 説明 |
|-----------------|--------|------|
| `Input_Number` | # | 入力番号 |
| `Original_Query` | ユーザーの質問 | 入力された質問文 |
| `Original_Answer` | ユーザーの回答 | 入力された回答文（存在する場合） |
| `Search_Query` | 検索クエリ | 実際に検索に使用したクエリ（LLM拡張時は拡張後） |
| `Search_Result_Q` | 類似質問 | 検索結果の質問文 |
| `Search_Result_A` | 類似回答 | 検索結果の回答文 |
| `Similarity` | 類似度 | ハイブリッドスコア（0.0 - 1.0） |
| `Vector_Weight` | ベクトルの重み | 使用した vector_weight 値 |
| `Top_K` | 候補数 | 返却件数上限 |
| `Generated_Tags` | 生成タグ | 生成されたタグ（存在する場合） |

Metadata シートに検索パラメータ（vector_weight, search_mode, search_type, top_k, embedding_provider, embedding_model, timestamp）を記録する。

---

## 設定パラメータ

`config/settings.yaml` で検索パラメータを設定する。

### セクション構成

| セクション | 対象 | 主な設定 |
|-----------|------|---------|
| `common` | 全プログラム共通 | search_type, vector_weight, search_mode, search_source, keyword 設定 |
| `ui` | Streamlit UI | top_k（デフォルト 3）、vector_weight 初期値 |
| `batch` | バッチ処理 | top_k（デフォルト 4） |

詳細は `docs/CONFIGURATION.md` を参照。

---

## トラブルシューティング

`docs/TROUBLESHOOTING.md` を参照。
