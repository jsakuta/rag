# RAG プロジェクト — 引き継ぎ資料

> 最終更新: 2026-02-17
> リポジトリ: `C:\VSCode\rag\`

---

## 目次

- [プロジェクト全体マップ](#プロジェクト全体マップ)
- [進化の系譜](#進化の系譜)
- [rag-maintenance（Phase2 PoC 本番）](#rag-maintenancephase2-poc-本番)
- [rag-local（Phase1 ローカル検証・評価基盤）](#rag-localphase1-ローカル検証評価基盤)
- [アーカイブ対象](#アーカイブ対象)
- [技術的注意事項](#技術的注意事項)
- [環境構築リンク集](#環境構築リンク集)
- [アーカイブ実施手順](#アーカイブ実施手順)

---

## プロジェクト全体マップ

```
rag/
├── rag-maintenance/     [現行] Phase2 PoC 本番 — Teams Bot (TypeScript)
├── rag-local/          [現行] Phase1 ローカル検証・評価基盤 (Python)
├── archive/             [非推奨] 旧版プロジェクト群
│   ├── rag-batch/       第2世代: Excel一括バッチ処理
│   ├── rag-streamlit/   第2世代: Streamlit対話型UI
│   └── rag-reranker/    第1世代: Cross-Encoder Reranking PoC
└── docs/                共通ドキュメント
```

| ディレクトリ | 状態 | 目的 | 技術 |
|---|---|---|---|
| **rag-maintenance** | 現行（引き継ぎ対象） | Teams Bot: 事務改定影響候補検出 | TypeScript, M365 Agents SDK, Azure AI Search, Cosmos DB |
| **rag-local** | 現行（引き継ぎ対象） | ローカル検証: バッチ処理 + Streamlit UI + 事務改定評価 | Python, ChromaDB, Gemini/Azure OpenAI |
| archive/rag-batch | 非推奨 | Excel一括バッチ処理 → rag-localに吸収済み | Python, multilingual-e5 |
| archive/rag-streamlit | 非推奨 | Streamlit対話型UI → rag-localに吸収済み | Python, Streamlit |
| archive/rag-reranker | 非推奨 | Cross-Encoder Reranking PoC → DEPRECATED明記 | Python, SentenceTransformer |

---

## 進化の系譜

```
rag-reranker (第1世代)
  │  融資業務Q&A + Cross-Encoder + PDF処理
  │  技術: SentenceTransformer, janome
  │
  ├─→ rag-batch (第2世代A)
  │     Excel一括バッチ処理特化
  │     技術: Sudachi, Factory Pattern
  │
  ├─→ rag-streamlit (第2世代B)
  │     rag-batchのStreamlit UI版(フォーク)
  │
  └─→ rag-local (第3世代/現行ローカル)
        全機能統合 + ChromaDB + Gemini/Azure OpenAI
        技術: SearchStrategy, MultiStageOrchestrator
        │
        └─→ rag-maintenance (Phase2 PoC/現行本番)
              rag-localの検索知見をAzure環境に移植
              技術: M365 Agents SDK, Azure AI Search, Cosmos DB
```

---

## rag-maintenance（Phase2 PoC 本番）

千葉銀行 デジタル戦略部（B&DX）の Phase2 PoC。事務改定時にシナリオ・FAQへの影響候補をAI検索で検出するTeams Botシステム。

### Botソースコード構成（計2,013行）

| ファイル | 行数 | 責務 |
|---------|------|------|
| `agent.ts` | 688 | メインBotロジック、全Actionハンドラ、検索実行、キャッシュ管理 |
| `cards.ts` | 969 | Adaptive Cardビルダー全種（検索/結果/削除/Excel出力完了等） |
| `excel.ts` | 187 | カテゴリ別Excel生成（Lv/文字数形式、LEN数式、黄色ハイライト） |
| `cosmos.ts` | 81 | Cosmos DB操作（FAQ論理削除、要修正フラグ保存） |
| `sharepoint.ts` | 57 | SPOアップロード（Graph API、4MBチェック） |
| `config.ts` | 28 | 設定定数・環境変数参照 |
| `index.ts` | 3 | Expressサーバー起動 |

### 実装済み機能要件

| FR | 内容 | 状態 |
|----|------|------|
| FR-001 | テキスト入力（2000文字制限） | 完了 |
| FR-002 | 検索カード（タブUI: シナリオ/FAQ） | 完了 |
| FR-003 | 意味検索（ハイブリッド: BM25+ベクトルRRF） | 完了 |
| FR-004 | キーワード検索 | 完了 |
| FR-005 | 検索結果表示（ページネーション、UTF-8サイズ25KB上限） | 完了 |
| FR-013 | FAQ論理削除 | 完了 |
| FR-014 | シナリオ要修正フラグ保存 | 完了 |
| FR-015 | カテゴリ別Excel出力+SPOアップロード+戻るボタン | 完了（動作確認中） |

### Azure環境（命名: `*-maintenance-poc`）

| リソース | 名前 | SKU |
|---------|------|-----|
| Resource Group | `rg-maintenance-poc` | - |
| Azure OpenAI | `aoai-maintenance-poc` | S0 |
| AI Search | `srch-maintenance-poc` | Basic |
| Cosmos DB | `cosmos-maintenance-poc` | Serverless |
| Web App | `app-maintenance-bot-poc` | B1 |
| Bot Service | `bot-maintenance-poc` | F0 Single-Tenant |
| Key Vault | `kv-maintenance-poc` | - |

### RBAC構成（7ロール）

| 付与先 | ターゲット | ロール |
|--------|-----------|--------|
| AI Search MI | Azure OpenAI | Cognitive Services OpenAI User |
| AI Search MI | Cosmos DB | Cosmos DB Account Reader Role |
| AI Search MI | Cosmos DB | Built-in Data Reader |
| Web App MI | AI Search | Search Index Data Reader |
| Web App MI | Cosmos DB | Built-in Data Contributor |
| Web App MI | Key Vault | Key Vault Secrets User |

> Web App → Azure OpenAI のロールは不要（BotはAIを直接呼ばない。EmbeddingはAI Search Skillset経由）

### 環境変数

| 変数名 | デフォルト | 用途 |
|--------|-----------|------|
| `AI_SEARCH_ENDPOINT` | (必須) | AI Searchエンドポイント |
| `AI_SEARCH_INDEX_NAME` | `maintenance-search-index` | インデックス名 |
| `COSMOS_DB_ENDPOINT` | (必須) | Cosmos DBエンドポイント |
| `COSMOS_DB_DATABASE` | `maintenance-db` | DB名 |
| `SPO_SITE_ID` | (必須) | SharePointサイトID |
| `SPO_DRIVE_ID` | (必須) | SharePointドライブID |
| `SPO_UPLOAD_FOLDER` | `影響候補シナリオ` | アップロード先フォルダ名 |

認証: 全て`DefaultAzureCredential`使用（Managed Identity / 開発者CLI）

### 未完了タスク

1. **F5デバッグ動作確認**（最優先）: UTF-8サイズ計測修正後の検索結果表示確認
2. **Excel出力E2Eテスト**: 検索→要修正保存→Excel出力→SPOリンク→Excelファイル確認
3. **Toolkit再Deploy**: `manifest.json`の`validDomains`変更反映が必要
4. **手順書完成**: スクリーンショット追加、最終Word化

### ドキュメント

| 文書 | バージョン | パス |
|------|-----------|------|
| 要件定義書 | v3.4 | `rag-maintenance/docs/要件定義書.md` |
| 導入手順書 | v1.5 | `rag-maintenance/docs/導入手順書.md` |
| データベース設計書 | v1.2 | `rag-maintenance/docs/データベース設計書.md` |
| 検索ロジック比較 | - | `rag-maintenance/docs/検索ロジック比較_Phase1_vs_Phase2.md` |

---

## rag-local（Phase1 ローカル検証・評価基盤）

事務改定のシナリオ影響をローカルで検証・評価するPython基盤。2つのAIアプリが `apps/` 配下に論理的に分離されている。

| 項目 | 内容 |
|------|------|
| 言語 | Python 3.11 |
| ベクトルDB | ChromaDB（永続化） |
| 埋め込み | Azure OpenAI text-embedding-3-large / Gemini Embedding |
| LLM | gemini-2.5-flash-lite |
| UI | Streamlit |
| データ規模 | scenarios 2,318件 + faqs 18,744件 = 計21,047件 |

### 3つの機能（論理的分離）

#### 機能A: バッチ処理（`main.py batch`）

入力Excelの質問に対してRAG検索し、結果をExcel出力する。

- エントリポイント: `main.py` → `Processor` → `Searcher` → `SearchStrategy`
- 固有モジュール: `src/handlers/input_handler.py`(610行), `src/handlers/output_handler.py`(431行)

#### 機能B: Streamlit回答支援UI（`main.py interactive`）

対話的にRAG検索結果を閲覧。通常検索 + 事務改定評価の2モード搭載。

- エントリポイント: `ui/chat.py`
- 固有依存: `streamlit`, `streamlit_elements`

#### 機能C: 事務改定評価スクリプト（`scripts/evaluate_revisions.py`）

事務改定の正解ID発見率を定量評価。Azure/VertexAI両プロバイダーで比較し、評価Excelを出力。

- エントリポイント: `scripts/evaluate_revisions.py`(1,217行)
- 固有モジュール: `src/core/judgment_support.py`(121行)
- 固有依存: `rich`

#### 共通コード（3機能間で共有、計3,000行以上）

| モジュール | 行数 | 役割 |
|-----------|------|------|
| `config.py` | 289 | SearchConfig + YAML設定読込 |
| `src/utils/auth.py` | 176 | LLM/埋め込みモデルファクトリー |
| `src/utils/vector_db.py` | 277 | ChromaDBラッパー |
| `src/utils/dynamic_db_manager.py` | 1,044 | 業務分野別DB管理 |
| `src/core/search/` | - | 検索エンジン群（keyword, vector, query_enhancer, text_combiner, multi_stage_orchestrator） |
| `config/settings.yaml` | 243 | common/ui/batch/evaluation 4セクション |

> 共通コード量が多いため物理的分割はせず、論理的境界を文書化。将来分離する場合は`rag-common`パッケージ化（pip editable install）を推奨。

### Phase1 → Phase2 の技術移行

| 観点 | Phase1（rag-local） | Phase2（rag-maintenance） |
|------|---------------------|--------------------------|
| ベクトル化 | ChromaDB/3072次元 | AI Search HNSW/3072次元 |
| テキスト検索 | Sudachi+Jaccard（1フィールド） | ja.microsoft+BM25（5フィールド） |
| スコア統合 | 加重平均（0.9v+0.1k） | RRF（k=60固定） |
| 実行形式 | Pythonバッチ（Excel入出力） | Teams Bot（リアルタイム） |
| Embedding更新 | スクリプト手動実行 | Indexer自動（1時間毎） |
| LLMクエリ拡張 | 3戦略あり | なし（精度向上未確認のため省略） |

---

## アーカイブ対象

### rag-reranker（第1世代）

| 項目 | 内容 |
|------|------|
| 状態 | READMEにDEPRECATED明記 |
| サイズ | 85MB (.venv除く) |
| テスト | なし（tests/は空） |
| 固有価値 | Cross-Encoder: `hotchpotch/japanese-reranker-cross-encoder-large-v1`、Azure Document Intelligence PDF処理 |

### rag-batch（第2世代A）

| 項目 | 内容 |
|------|------|
| 状態 | rag-localに全機能吸収済み |
| サイズ | 521MB (.venv除く) — old/のExcel・キャッシュが大半 |
| テスト | なし（tests/は空） |
| 固有価値 | `old/OrganizeFAQ*.py`: Claude3を使ったFAQ整理の初期実験（歴史的参考のみ） |

### rag-streamlit（第2世代B）

| 項目 | 内容 |
|------|------|
| 状態 | rag-localに全機能吸収済み、固有価値なし |
| サイズ | 172MB (.venv除く) |
| テスト | なし（tests/は空） |

---

## 技術的注意事項

### M365 Agents SDK

- `actionExecute`の`data`は `{type, verb, data: {ユーザーデータ}}` 形式。ユーザーデータは `data.data` にネスト
- 全extract関数（`extractQuery`, `extractSelectedIds`等）でネスト対応必須
- `adaptiveCardsActions.ts:130`: `(a.value as any).action` をハンドラ第3引数に渡す

### Adaptive Card サイズ制限

- Teams上限: 約28KB（UTF-8バイト）、安全マージンで25KB制限
- **正しい計測**: `Buffer.byteLength(JSON.stringify(card), "utf8")`
- `JSON.stringify().length`はUTF-16コードユニット数。日本語コンテンツでは実サイズの60-70%しか計測できない
- 二分探索で最適perPage値を特定する実装済み（`cards.ts`）

### Azure環境

- **遅延初期化必須**: SearchClient/CosmosClient/GraphClientは全てlazy init。`env-cmd`より先にモジュール評価されるとInvalid URLエラー
- **Git Bashパス変換問題**: Azure CLI使用時は `MSYS_NO_PATHCONV=1` を設定
- **OpenAIモデル再デプロイ**: ソフトデリート状態のモデルはパージが必要
- **Toolkit v6系**: `m365agents.yml`（旧`teamsapp.yml`）。manifest.json変更後はToolkit再Deployが必要

### Windows環境

- **予約語ファイル**: `nul` ファイルが `rag-maintenance/` と `rag-local/` に存在。特殊ツールでの削除が必要
- **パス**: 日本語ユーザー名含むパスでのGit操作に注意

### データ規模

| コンテナ | 件数 |
|---------|------|
| scenarios | 2,318 |
| faqs | 18,744 |
| **合計** | **21,047** |

---

## 環境構築リンク集

| 文書 | パス | 内容 |
|------|------|------|
| 導入手順書 v1.5 | `rag-maintenance/docs/導入手順書.md` | Azure環境構築 Step 1〜13 |
| 要件定義書 v3.4 | `rag-maintenance/docs/要件定義書.md` | 全機能要件+非機能要件 |
| DB設計書 v1.2 | `rag-maintenance/docs/データベース設計書.md` | 3コンテナ+Excel列マッピング |
| 検索ロジック比較 | `rag-maintenance/docs/検索ロジック比較_Phase1_vs_Phase2.md` | Phase1→Phase2移行の技術比較 |
| rag-local README | `rag-local/README.md` | クイックスタート、ディレクトリ構造 |
| rag-local 設定 | `rag-local/config/settings.yaml` | 4セクション設定リファレンス |
| Google Cloud認証 | `rag-local/docs/GOOGLE_CLOUD_AUTH.md` | Vertex AI認証設定 |

---

## アーカイブ実施手順

### Step 1: 機密ファイル削除

```bash
# 各プロジェクトの.env削除
rm rag-reranker/.env rag-batch/.env rag-streamlit/.env

# rag-reranker: 業務データExcel・Azurite設定
rm rag-reranker/__azurite_db_table__.json
# rm rag-reranker/問い合わせ履歴データ*.xlsx 等

# rag-batch: old/内のExcel・ベクトルキャッシュ
# rm -rf rag-batch/old/

# rag-streamlit: ログ
rm rag-streamlit/app.log
```

### Step 2: venv削除（サイズ削減）

```bash
rm -rf rag-reranker/.venv rag-batch/.venv rag-streamlit/.venv
```

### Step 3: ディレクトリ移動

```bash
mkdir archive
git mv rag-reranker archive/rag-reranker
git mv rag-batch archive/rag-batch
git mv rag-streamlit archive/rag-streamlit
```

### Step 4: 整理後の確認（実施済み）

```
rag/
├── rag-maintenance/     [現行] Phase2 PoC Teams Bot
├── rag-local/           [現行] Phase1 ローカル検証・評価（旧rag-gemini）
│   ├── apps/answer-support/   回答支援AI
│   └── apps/revision-eval/    事務改定評価AI
├── archive/
│   ├── rag-batch/
│   ├── rag-streamlit/
│   └── rag-reranker/
├── docs/
│   ├── DOCKER.md
│   ├── SECURITY.md
│   └── TROUBLESHOOTING.md
└── README.md            ← 本ファイル（引き継ぎ資料）
```
