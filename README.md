# RAG プロジェクト — 引き継ぎ資料

> 最終更新: 2026-03-02
> リポジトリ: `C:\VSCode\rag\`

---

## 目次

- [プロジェクト全体マップ](#プロジェクト全体マップ)
- [進化の系譜](#進化の系譜)
- [rag-maintenance（Phase2 PoC 本番）](#rag-maintenancephase2-poc-本番)
- [rag-local（Phase1 ローカル検証・評価基盤）](#rag-localphase1-ローカル検証評価基盤)
- [技術的注意事項](#技術的注意事項)
- [環境構築リンク集](#環境構築リンク集)

---

## プロジェクト全体マップ

```
rag/
├── rag-maintenance/     [現行] Phase2 PoC 本番 — Teams Bot (TypeScript)
├── rag-local/           [現行] Phase1 ローカル検証・評価基盤 (Python)
│   ├── apps/
│   │   ├── answer-support/   回答支援AI（バッチ + Streamlit UI）
│   │   └── revision-eval/    事務改定評価AI（バッチ + 評価UI）
│   ├── src/              共有コア（検索エンジン、DB管理等）
│   ├── config/           設定ファイル
│   ├── scripts/          ユーティリティスクリプト
│   └── docs/             ドキュメント
├── archive/              [Git管理外] 旧版プロジェクト群（ローカル参照用）
└── docs/                 共通ドキュメント（SECURITY.md, TROUBLESHOOTING.md）
```

| ディレクトリ | 状態 | 目的 | 技術 |
|---|---|---|---|
| **rag-maintenance** | 現行（引き継ぎ対象） | Teams Bot: 事務改定影響候補検出 | TypeScript, M365 Agents SDK, Azure AI Search, Cosmos DB |
| **rag-local** | 現行（引き継ぎ対象） | ローカル検証: 回答支援AI + 事務改定評価AI | Python, ChromaDB, Gemini/Azure OpenAI |

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

### 残存タスク

1. **Toolkit再Deploy**: コミット済み修正のデプロイ実施が必要
2. **手順書完成**: スクリーンショット追加、最終Word化

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

### 2つのAIアプリケーション

#### 回答支援AI（`apps/answer-support/`）

FAQ/シナリオを対象にハイブリッド検索（ベクトル+キーワード）を実行。

| モード | コマンド |
|--------|---------|
| バッチ（Excel入出力） | `python apps/answer-support/main.py` |
| Streamlit UI | `python apps/answer-support/main.py interactive` |

#### 事務改定評価AI（`apps/revision-eval/`）

改定内容→変更対象シナリオを Azure/VertexAI 両方で検索し、正解IDとのマッチ率を評価。

| モード | コマンド |
|--------|---------|
| バッチ（Excel出力） | `python apps/revision-eval/evaluate_revisions.py` |
| 評価UI（Streamlit） | `streamlit run apps/revision-eval/ui/eval_ui.py` |

#### 共有コア（`src/`）

| モジュール | 役割 |
|-----------|------|
| `config.py` | SearchConfig + YAML設定読込 |
| `src/utils/auth.py` | LLM/埋め込みモデルファクトリー |
| `src/utils/vector_db.py` | ChromaDBラッパー |
| `src/utils/dynamic_db_manager.py` | 業務分野別DB管理 |
| `src/core/search/` | 検索エンジン群（keyword, vector, query_enhancer, text_combiner） |
| `config/settings.yaml` | common/ui/batch/evaluation 4セクション |

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

