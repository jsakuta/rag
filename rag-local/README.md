# RAG-Local（ローカル検証・評価基盤）

## 概要

2つのAIアプリケーション + 共有コアで構成するローカル RAG 検索システム。

| AI | バッチ | UI | 用途 |
|----|-------|-----|------|
| **回答支援AI（類似回答検索）** | `apps/answer-support/main.py` | `apps/answer-support/ui/chat.py` | FAQ/シナリオ検索 |
| **運用保守効率化AI（改定影響調査）** | `apps/revision-ops/run_eval.py` | `apps/revision-ops/ui/ops_ui.py` | 改定影響候補の調査 |

---

## 前提条件 · 環境構築

```bash
cd rag-local
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 環境変数

```bash
cp .env.example .env
# .env を編集して以下を設定:
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
# GEMINI_PROJECT_ID, GEMINI_CREDENTIALS_PATH
```

詳細は [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) を参照。

### データ配置

```
data/source/scenarios/latest/    ← シナリオExcel
data/source/faq/latest/          ← FAQデータ
data/input/multi_stage_input.xlsx ← 改定影響調査用入力
```

---

## クイックスタート

### 回答支援AI（類似回答検索）

```bash
# バッチ処理
python apps/answer-support/main.py

# UI起動（Streamlit）
python apps/answer-support/main.py interactive
# または直接起動:
streamlit run apps/answer-support/ui/chat.py
```

### 運用保守効率化AI（改定影響調査）

```bash
# バッチ（Excel出力）
python apps/revision-ops/run_eval.py

# 評価UI
streamlit run apps/revision-ops/ui/ops_ui.py
```

---

## ディレクトリ構造

```
rag-local/
├── apps/
│   ├── answer-support/           # 回答支援AI（類似回答検索）
│   │   ├── main.py               # バッチ処理エントリポイント
│   │   └── ui/
│   │       └── chat.py           # 回答支援 Streamlit UI
│   └── revision-ops/             # 運用保守効率化AI（改定影響調査）
│       ├── run_eval.py           # バッチExcel出力
│       └── ui/
│           └── ops_ui.py         # 改定影響調査 Streamlit UI
├── ui/
│   └── shared.py                 # 共通UI部品（apps/*/ui/ から import）
├── src/                          # 共有コア（検索エンジン、DB管理等）
│   ├── core/                     # 検索・処理ロジック
│   └── utils/                    # 埋め込みモデル、DB管理等
├── config.py                     # 設定管理
├── config/
│   ├── settings.yaml             # 検索・UI設定
│   └── business_areas.yaml       # 業務分野定義
├── scripts/                      # ユーティリティ（DB構築等）
├── data/                         # データ（ベクトルDB、入出力）
├── prompt/                       # プロンプトテンプレート
└── tests/                        # テスト
```

---

## AI使用箇所マップ

| 処理 | AIモデル | 設定環境変数 |
|-----|---------|-------------|
| ベクトル化 | text-embedding-3-large / gemini-embedding-001 | `DEFAULT_EMBEDDING_PROVIDER` |
| クエリ拡張 | gemini-2.5-flash-lite | `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL` |
| 関連性判定 | gemini-2.5-flash-lite | `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL` |

---

## 改定影響調査システム

改定内容→変更対象シナリオを Azure OpenAI / VertexAI 両方で検索し、正解IDとのマッチ率を評価。

```bash
# Step 1: 改定前シナリオDB構築
python scripts/build_db.py --revisions-only

# Step 2: 評価実行
python apps/revision-ops/run_eval.py
# → data/output/latest/rev/rev_eval_batch_YYYYMMDD.xlsx
```

詳細は [docs/REVISION_OPS.md](./docs/REVISION_OPS.md) を参照。

---

## トラブルシューティング

よくある問題は [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) を参照。

```bash
# DB内容確認
python scripts/check_db_content.py

# DB再構築（破損時）
# rm -rf data/vector_db/  から再実行
```

---

## rag-maintenance との関係

| | rag-local | rag-maintenance |
|--|-----------|----------------|
| 役割 | ローカル開発・検証・評価 | 本番 Teams Bot |
| 技術 | Python + ChromaDB | TypeScript + Azure AI Search |
| 目的 | 検索パラメータ最適化 | 運用保守効率化 |

---

## ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | システムアーキテクチャ |
| [docs/API_REFERENCE.md](./docs/API_REFERENCE.md) | API仕様 |
| [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) | 環境変数・設定オプション |
| [docs/DB_BUILD_GUIDE.md](./docs/DB_BUILD_GUIDE.md) | DB構築ガイド |
| [docs/GOOGLE_CLOUD_AUTH.md](./docs/GOOGLE_CLOUD_AUTH.md) | Google Cloud 認証設定 |
| [docs/PROMPTS.md](./docs/PROMPTS.md) | プロンプト詳細 |
| [docs/REVISION_OPS.md](./docs/REVISION_OPS.md) | 改定影響調査システム |
| [docs/SECURITY.md](./docs/SECURITY.md) | セキュリティガイド |
| [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) | トラブルシューティング |
