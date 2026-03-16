# 運用保守効率化AI（事務改定影響検知システム Phase 2 PoC）

事務改定時にシナリオ・FAQ への影響候補を Azure AI Search で自動検出する Teams Bot システム。

Azure AI Search のハイブリッド検索（BM25 + ベクトル検索 + Semantic Ranker）により、改定内容に関連するシナリオ・FAQ を高精度で特定する。

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| Bot SDK | M365 Agents SDK (`@microsoft/agents-hosting`) |
| 言語 | TypeScript |
| 検索 | Azure AI Search (Basic) — ハイブリッド検索 + Semantic Ranker |
| ベクトル化 | Azure OpenAI `text-embedding-3-large`（3,072次元）— AI Search Skillset/Vectorizer 経由 |
| DB | Azure Cosmos DB (Serverless, NoSQL API) |
| UI | Adaptive Card（Teams） |
| Excel出力 | ExcelJS（要修正シナリオのハイライト出力） |
| 認証 | Bot認証は Entra ID アプリ + clientSecret、Azure リソースアクセスは Managed Identity |

## ディレクトリ構成

```
rag-maintenance/
├── maintenance-bot/           # Bot アプリケーション（メイン）
│   ├── src/
│   │   ├── index.ts           # エントリポイント（Express + Agent Server + Application Insights）
│   │   ├── agent.ts           # Bot ロジック（検索・カード操作ハンドラ）
│   │   ├── cards.ts           # Adaptive Card 生成（検索UI・結果表示・ページネーション）
│   │   ├── config.ts          # 環境変数・アプリ定数
│   │   ├── cosmos.ts          # Cosmos DB 操作（影響評価の保存・FAQ削除）
│   │   ├── excel.ts           # Excel 出力（要修正シナリオ一覧）
│   │   └── sharepoint.ts      # SharePoint Online アップロード
│   ├── .vscode/               # F5デバッグ設定（launch.json / tasks.json）
│   ├── appPackage/            # Teams アプリマニフェスト
│   ├── env/                   # 環境変数ファイル（.env.dev / .env.local）※引き継ぎ時は .env.dev.example のみ同梱
│   ├── infra/                 # Azure Bicep テンプレート
│   ├── m365agents.yml         # M365 Agents Toolkit 設定（Provision / Deploy）
│   └── package.json
├── scripts/                   # AI Search リソース定義 JSON
│   ├── index-definition.json  # インデックス定義（15フィールド）
│   ├── datasource-*.json      # データソース定義（scenarios / faqs）
│   ├── skillset.json          # Skillset（ベクトル化）
│   ├── indexer-*.json         # Indexer 定義（scenarios / faqs）
│   ├── convert-excel-to-json.py  # Excel → Cosmos DB 用 JSON 変換
│   └── seed-cosmos.ts         # Cosmos DB テストデータ投入
└── docs/                      # 設計ドキュメント
    ├── 要件定義書.docx         # 機能要件・非機能要件・システム構成
    ├── 導入手順書.docx         # Azure 環境構築 Step 1〜14
    ├── 検索設計書.docx         # 検索アルゴリズム・パラメータ・チューニング経緯
    └── データベース設計書.docx  # Cosmos DB スキーマ定義・データ構造
```

## セットアップ

環境構築の詳細手順は導入手順書（`docs/導入手順書.docx`）を参照。

### 前提条件

- Node.js 18 / 20 / 22
- Azure サブスクリプション（リソース作成権限）
- Microsoft 365 テナント（Teams 管理者権限）
- Visual Studio Code + [M365 Agents Toolkit 拡張機能](https://aka.ms/teams-toolkit)

### 環境構築の流れ

1. **Azure リソース作成**（Step 1〜6）: リソースグループ、Azure OpenAI、AI Search、Cosmos DB、App Insights、App Service
2. **Bot / アプリ登録**（Step 7〜10）: Toolkit Provision、Azure Bot 登録、サービスプリンシパル作成、RBAC / Graph 権限付与（引き継ぎコードがある場合、Step 7 のプロジェクト作成・yml カスタマイズはスキップ — 詳細は導入手順書参照）
3. **AI Search 設定**（Step 11）: インデックス・データソース・Skillset・Indexer の作成
4. **Bot 実装・デプロイ**（Step 12〜13）: アプリ設定、Toolkit Deploy、Teams アプリ Publish
5. **動作確認**（Step 14）: F5 ローカルデバッグと Teams 上での検索テスト

### ローカル開発

```bash
cd maintenance-bot
npm install
# env/.env.local に環境変数を設定後:
npm run dev:teamsfx    # ローカル起動（VSCode の F5 デバッグからも実行可）
```

必要な環境変数と `.localConfigs` の扱いは導入手順書の Step 10、Step 12、Step 14 を参照。

## アーキテクチャ

```
Teams ユーザー
    │
    ▼
Azure Bot Service ──── Azure Web App (maintenance-bot)
                           │
                     ┌─────┼─────┐
                     ▼     ▼     ▼
              AI Search  Cosmos DB  SharePoint Online
                 │
                 ▼
            Azure OpenAI
          (Skillset/Vectorizer 経由のみ)
```

- Bot から Azure OpenAI を**直接呼び出さない**（Embedding は AI Search Skillset/Vectorizer 経由）
- Teams / Bot Framework から Bot への認証は Entra ID アプリ + clientSecret
- Bot から Azure AI Search / Cosmos DB / Microsoft Graph へのアクセスは Managed Identity（RBAC）

## 検索の仕組み

詳細は検索設計書（`docs/検索設計書.docx`）を参照。

| 項目 | 値 |
|------|-----|
| 検索方式 | ハイブリッド検索（BM25 + ベクトル + RRF統合） |
| ベクトル次元数 | 3,072（text-embedding-3-large） |
| スコア統合 | RRF（Reciprocal Rank Fusion） — ベクトル重み 4.5 |
| リランキング | Semantic Ranker（L2: Cross-Encoder） |
| HNSW パラメータ | m=10, efSearch=1000, efConstruction=400 |
| デフォルト表示件数 | 30件（10〜100件で選択可） |
| 検索モード | ハイブリッド検索 / キーワード検索 |

## データ構成

Cosmos DB に3コンテナを配置。

| コンテナ | 用途 | パーティションキー | 件数目安 |
|---------|------|-----------------|---------|
| `scenarios` | シナリオデータ（4カテゴリ） | `/categoryId` | 約 2,300件 |
| `faqs` | FAQ データ（3カテゴリ） | `/categoryId` | 約 18,700件 |
| `impactAssessments` | 影響評価結果の保存 | `/searchId` | — |

## ドキュメント一覧

| ドキュメント | 内容 |
| ------------ | ------ |
| `docs/要件定義書.docx` | 機能要件 FR-001〜FR-009、非機能要件、システム構成 |
| `docs/導入手順書.docx` | Azure 環境構築〜デプロイの全手順（Step 1〜14） |
| `docs/検索設計書.docx` | 検索アルゴリズム、パラメータ、チューニング経緯 |
| `docs/データベース設計書.docx` | Cosmos DB スキーマ定義、combinedContent 生成ルール |

> 各ドキュメントの役割分担: 「何を作るか」→ 要件定義書、「どう構築するか」→ 導入手順書、「検索をどう設計したか」→ 検索設計書、「データをどう持つか」→ データベース設計書

