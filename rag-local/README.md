# RAG-Local（ローカル検証・評価基盤）

## 概要

2つのAIアプリケーションと共有コアで構成する、ローカル環境で動作する検索システム（RAG: Retrieval-Augmented Generation 方式）。

| AI | バッチ | UI | 用途 | 詳細ドキュメント |
|----|-------|-----|------|----------------|
| **回答支援AI（類似回答検索）** | `apps/answer-support/main.py` | `apps/answer-support/ui/chat.py` | 問い合わせ履歴やシナリオから似た質問・回答を検索 | [docs/ANSWER_SUPPORT.md](./docs/ANSWER_SUPPORT.md) |
| **運用保守効率化AI（改定影響調査）** | `apps/revision-ops/run_eval.py` | `apps/revision-ops/ui/ops_ui.py` | 事務改定で影響を受けるシナリオ候補を調査 | [docs/REVISION_OPS.md](./docs/REVISION_OPS.md) |

---

## ドキュメント一覧

以下の順序で読むことを推奨します:

| # | ドキュメント | 内容 |
|---|------------|------|
| 1 | **README.md**（本ファイル） | セットアップ手順・全体像 |
| 2 | [docs/ANSWER_SUPPORT.md](./docs/ANSWER_SUPPORT.md) | 回答支援AIの詳細（使い方・DB構築・出力フォーマット） |
| 3 | [docs/REVISION_OPS.md](./docs/REVISION_OPS.md) | 改定影響調査AIの詳細（評価フロー・参照データ管理） |
| 4 | [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) | 設定リファレンス（環境変数・YAML詳細） |
| 5 | [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | 技術アーキテクチャ・API仕様・プロンプト |
| 6 | [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) | トラブルシューティング |

設計書アーカイブは `docs/plans/` に格納されています。

---

## セットアップ手順（ゼロから動かすまで）

### Step 1: Python環境の構築

**前提:** Python 3.9 以上が必要です（依存ライブラリの要件）。

```bash
cd rag-local
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### Step 2: 認証情報の準備

> **どの認証が必要か？** Step 4 で設定する `DEFAULT_EMBEDDING_PROVIDER` に依存します。`vertex_ai` のみ使用する場合は Azure OpenAI の認証（Step 2b）は不要です。`azure_openai` のみ使用する場合でも、LLM（大規模言語モデル。検索語の補強や関連性判定に使用）は Gemini を使用するため Google Cloud 認証（Step 2a）は必須です。

#### Step 2a: Google Cloud（Vertex AI / Gemini）

LLMおよび埋め込みモデル（文章を数値に変換するAIモデル）で VertexAI を使用する場合に必要です。**LLM は Gemini のみ対応のため、全環境で必須です。**

1. Google Cloud Console でサービスアカウントキーを作成（必要権限: `Vertex AI User`）
2. JSON キーファイルをダウンロード
3. プロジェクトルートに `gemini_credentials.json` として配置

環境変数の設定（OS別）:

```powershell
# Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\gemini_credentials.json"
```

```bash
# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gemini_credentials.json"
```

> **注意**: `gemini_credentials.json` は機密情報です。共有フォルダに置かず、アクセス権限を制限してください。Git にコミットしないでください（`.gitignore` 登録済み）。

#### Step 2b: Azure OpenAI

埋め込みモデルで Azure OpenAI を使用する場合に必要です。`DEFAULT_EMBEDDING_PROVIDER=vertex_ai` で回答支援AIのみ使用する場合はスキップできます。

- Azure Portal で API キーとエンドポイント URL を確認
- `.env` ファイルに設定（Step 3 参照）

### Step 3: 環境変数の設定

```bash
cp .env.example .env
```

`.env` を編集して以下の主要変数を設定:

| 変数 | 説明 | 例 |
|------|------|-----|
| `DEFAULT_LLM_PROVIDER` | LLM（言語モデル）のプロバイダー | `gemini` |
| `DEFAULT_LLM_MODEL` | LLMのモデル名 | `gemini-2.5-flash-lite` |
| `DEFAULT_EMBEDDING_PROVIDER` | 埋め込みモデルのプロバイダー（モデル名はプロバイダーから自動決定） | `azure_openai` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API キー | `your-api-key` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI エンドポイント | `https://your-resource.openai.azure.com/` |
| `GEMINI_PROJECT_ID` | Google Cloud プロジェクトID | `your-project-id` |

全変数の詳細は [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) を参照。

### Step 4: ソースデータの配置

以下のディレクトリにExcelファイルを配置します（別途提供）:

```
data/source/
├── faq/latest/                    # 問い合わせ履歴データ（FAQ）
│   ├── 内部事務_履歴データ_YYYYMMDD.xlsx
│   └── スマイル_履歴データ_YYYYMMDD.xlsx
├── scenarios/
│   ├── latest/                    # 最新シナリオ（回答支援AI用）
│   │   ├── 内部事務_シナリオデータ_YYYYMMDD.xlsx
│   │   └── スマイル_シナリオデータ_YYYYMMDD.xlsx
│   └── revisions/                 # 変更前シナリオ（改定影響調査用）
│       ├── rev01smile_シナリオデータ_YYYYMMDD.xlsx
│       └── ...
```

**ファイル命名規則**:
- 問い合わせ履歴データ（FAQ）: `{業務名}_履歴データ_{YYYYMMDD}.xlsx`
- シナリオ: `{業務名}_シナリオデータ_{YYYYMMDD}.xlsx`
- `{業務名}` は日本語名（スマイル、内部事務等）。`config/business_areas.yaml` のマッピングで英語DB名に自動変換
- `{YYYYMMDD}` はデータ日付。同一業務分野に複数ファイルがある場合、最新日付のファイルが使用される

改定影響調査に必要な参照資料（`reference/`）は別途提供します。

### Step 5: DB構築

```bash
# 回答支援AI用のみ構築（推奨）
python scripts/build_db.py --no-revisions --force

# 改定影響調査用のみ構築
python scripts/build_db.py --revisions-only --force

# 全DB構築（回答支援AI + 改定影響調査）
python scripts/build_db.py --force

# 2回目以降: 差分のみ構築（更新があるDBのみ）
python scripts/build_db.py
```

- 回答支援AI用DBの詳細は [docs/ANSWER_SUPPORT.md](./docs/ANSWER_SUPPORT.md) を参照
- 改定DB構築は [docs/REVISION_OPS.md](./docs/REVISION_OPS.md) を参照

> **注意**: DB構築中は Streamlit UI を停止してください（検索用データベース ChromaDB のファイルロック競合を防止）。

> **Note:** `build_db.py` は azure_openai と vertex_ai の**両プロバイダー**でDBを構築します。片方の認証が未設定の場合、その側はエラーになりますが、認証済み側のDBは正常に構築されます。VertexAI のみ使用する場合、Azure 側のエラーは無視して構いません。

### Step 6: 動作確認

```bash
# 回答支援AI（バッチ）
python apps/answer-support/main.py

# 回答支援AI（UI）
streamlit run apps/answer-support/ui/chat.py

# 改定影響調査（バッチ）
python apps/revision-ops/run_eval.py

# 改定影響調査（UI）
streamlit run apps/revision-ops/ui/ops_ui.py
```

各AIの詳細な使い方は専用ドキュメントを参照してください。

### テスト実行

```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt
```

**開発用パッケージ** (`requirements-dev.txt`):
- `pytest` — テストフレームワーク
- `pytest-cov` — カバレッジ計測
- `pytest-mock` — モックライブラリ
- `pytest-asyncio` — 非同期テスト対応
- `faker` — テストデータ生成

```bash
# 全テスト実行
pytest

# カバレッジ付き
pytest --cov=src
```

テスト構成: `tests/unit/`（ユニットテスト）、`tests/integration/`（統合テスト）、`tests/fixtures/`（テスト用データ）。

---

## ディレクトリ構造

```
rag-local/
├── apps/                         # アプリケーション
│   ├── answer-support/           # 回答支援AI（類似回答検索）
│   │   ├── main.py               # バッチ処理エントリーポイント
│   │   └── ui/
│   │       └── chat.py           # Streamlit UI
│   └── revision-ops/             # 運用保守効率化AI（改定影響調査）
│       ├── run_eval.py           # バッチExcel出力
│       └── ui/
│           └── ops_ui.py         # 改定影響調査 Streamlit UI
│
├── config.py                     # 設定管理
├── requirements.txt              # 依存パッケージ
├── .env.example                  # 環境変数テンプレート
├── .streamlit/
│   └── config.toml              # Streamlit 設定
│
├── docs/                         # ドキュメント
│   ├── ANSWER_SUPPORT.md         # 回答支援AI詳細
│   ├── REVISION_OPS.md           # 改定影響調査詳細
│   ├── CONFIGURATION.md          # 設定リファレンス
│   ├── ARCHITECTURE.md           # アーキテクチャ・API仕様
│   ├── TROUBLESHOOTING.md        # トラブルシューティング
│   └── plans/                    # 設計書アーカイブ
│
├── src/                          # 共有コアライブラリ
│   ├── core/                     # コアロジック
│   │   ├── processor.py          # データ処理エンジン
│   │   ├── searcher.py           # 検索統合（Processor から使用）
│   │   ├── judgment_support.py   # LLM判断支援
│   │   └── search/               # 検索エンジン
│   │       ├── multi_stage_orchestrator.py  # 多段階検索（改定影響調査AI専用）
│   │       ├── search_strategy.py           # 検索戦略切替
│   │       ├── query_enhancer.py            # クエリ拡張
│   │       ├── vector_search_engine.py      # ベクトル検索
│   │       ├── keyword_search_engine.py     # キーワード検索
│   │       ├── chromadb_keyword_search.py   # ChromaDBキーワード検索
│   │       └── text_combiner.py             # テキスト結合
│   │
│   ├── types/                    # 型定義
│   │   └── search_types.py       # 検索関連の型定義
│   │
│   ├── handlers/                 # 入出力処理
│   │   ├── input_handler.py      # 入力処理
│   │   └── output_handler.py     # 出力処理
│   │
│   └── utils/                    # ユーティリティ
│       ├── dynamic_db_manager.py # DB管理
│       ├── vector_db.py          # ChromaDB 操作の共通インターフェース
│       ├── base_embedding.py     # 埋め込みモデル共通の基底クラス
│       ├── gemini_embedding.py   # Gemini埋め込み
│       ├── azure_embedding.py    # Azure埋め込み
│       ├── auth.py               # Google Cloud認証
│       ├── business_area_translator.py  # 業務領域変換
│       └── logger.py             # ログ設定
│
├── prompt/                       # プロンプト
│   ├── summarize_v1.0.txt        # クエリ拡張
│   └── judgment_support.txt      # 関連性判定
│
├── scripts/                      # ユーティリティスクリプト
│   ├── build_db.py               # DB構築（回答支援AI + 改定別 統合）
│   ├── generate_correct_ids.py   # 正解ID生成
│   ├── prepare_before_scenario.py # データ前処理
│   ├── check_db_content.py       # DB内容確認
│   └── create_handover_package.py # 引き継ぎパッケージ作成
│
├── config/
│   ├── settings.yaml             # 検索・UI設定
│   └── business_areas.yaml       # 業務分野定義
│
├── ui/
│   └── shared.py                 # 共通UI部品
│
├── data/                         # データディレクトリ
│   ├── vector_db/                # 検索用データベース（自動生成、ベクトルDB）
│   ├── source/
│   │   ├── scenarios/            # シナリオExcel
│   │   │   ├── latest/
│   │   │   └── revisions/
│   │   └── faq/                  # 問い合わせ履歴データ（FAQ）
│   │       └── latest/
│   ├── input/                    # 入力ファイル
│   └── output/                   # 出力ファイル
│       ├── latest/
│       └── archive/
│
├── tests/                        # テスト
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── logs/                         # ログファイル（自動生成）
```

---

## AI使用箇所マップ

| 処理 | AIモデル | 設定環境変数 |
|-----|---------|-------------|
| 文章の数値変換（ベクトル化） | text-embedding-3-large / gemini-embedding-001 | `DEFAULT_EMBEDDING_PROVIDER` |
| 検索語の自動補強（クエリ拡張） | gemini-2.5-flash-lite | `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL` |
| 検索結果の関連度チェック（関連性判定） | gemini-2.5-flash-lite | `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL` |

---

## 引き継ぎ時の注意

### 同梱しないもの（別途渡す / 除外）

| 対象 | 理由 | 対処 |
|------|------|------|
| `.env` | 認証情報 | `.env.example` から作成 |
| `gemini_credentials.json` | 認証情報 | 別経路で受け渡し |
| `data/vector_db/` | ベクトルDB | `build_db.py` で再構築 |
| `data/.keyword_cache/` | キャッシュ | 自動生成 |
| `data/output/` | 出力ファイル | 実行時生成 |
| `reference/` | 改定資料 | 別途提供 |
| `.venv/` | Python仮想環境 | `pip install` で再作成 |
| `logs/` | ログ | 実行時生成 |
| `CLAUDE.md` | 開発メモ | 引き継ぎ対象外 |

> **注記:** 引き継ぎパッケージは許可リスト方式で生成されます（`create_handover_package.py:INCLUDE`）。`CLAUDE.md` は開発者用のプロジェクトメモであり、運用には不要なため許可リストから除外されています。

### 同梱するもの

- **ソースコード**: `apps/`, `src/`, `ui/`, `scripts/`, `config/`, `config.py`
- **プロンプト**: `prompt/`
- **テスト**: `tests/`
- **ドキュメント**: `README.md`, `docs/`
- **設定テンプレート**: `.env.example`, `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.streamlit/`
- **ソースデータ**: `data/source/`, `data/input/`（別途渡す場合は空ディレクトリ）

引き継ぎパッケージの作成には `scripts/create_handover_package.py` を使用できます（許可リスト方式で秘密情報の混入を防止）:

```bash
# パッケージ作成（コードのみ、data/ は空ディレクトリ構造のみ）
python scripts/create_handover_package.py ./handover_package

# パッケージ作成（data/source/ と data/input/ の実データも含む）
python scripts/create_handover_package.py ./handover_package --include-data

# パッケージ作成（出力例を含む — 下記4種から最新1件ずつ選定）
python scripts/create_handover_package.py ./handover_package --include-examples

# 事前確認（コピーせず対象一覧とサイズを表示）
python scripts/create_handover_package.py ./handover_package --dry-run
```

| フラグ | 動作 |
|--------|------|
| （なし） | 許可リストのファイルをコピー。`data/` は空ディレクトリ構造のみ作成 |
| `--include-data` | `data/source/` と `data/input/` の実データも含める |
| `--include-examples` | `data/output/examples/` から種類ごとに最新1件を含める（回答支援×バッチ/UI、運用保守×バッチ/UI の4種） |
| `--dry-run` | コピーせず対象ファイル一覧と合計サイズを表示 |

> **Note:** 出力先ディレクトリが既に存在する場合はエラーになります。コピー後に秘密情報チェック（`.env`, `*credentials*`, `*.key`）を自動実行します。

---

## トラブルシューティング

よくある問題は [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) を参照。

```bash
# DB内容確認
python scripts/check_db_content.py

# DB再構築（破損時）
python scripts/build_db.py --force
```
