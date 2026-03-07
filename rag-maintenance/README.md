# 運用保守効率化AI（事務改定影響検知システム Phase2 PoC）

事務改定時にシナリオ・FAQ への影響候補を AI 検索で自動検出する Teams Bot システム。

Azure AI Search のハイブリッド検索（BM25 + ベクトル検索 + Semantic Ranker）により、改定内容に関連するシナリオ・FAQ を高精度で特定する。

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| Bot SDK | M365 Agents SDK (`@microsoft/agents-hosting`) |
| 言語 | TypeScript |
| 検索 | Azure AI Search (Basic) — ハイブリッド検索 + Semantic Ranker |
| ベクトル化 | Azure OpenAI `text-embedding-3-large`（3,072次元）— AI Search Skillset 経由 |
| DB | Azure Cosmos DB (Serverless, NoSQL API) |
| UI | Adaptive Card（Teams） |
| Excel出力 | ExcelJS（要修正シナリオのハイライト出力） |
| 認証 | Managed Identity（パスワードレス） |

## ディレクトリ構成

```
rag-maintenance/
├── maintenance-bot/           # Bot アプリケーション（メイン）
│   ├── src/
│   │   ├── index.ts           # エントリポイント（Express + Agent Server）
│   │   ├── agent.ts           # Bot ロジック（検索・カード操作ハンドラ）
│   │   ├── cards.ts           # Adaptive Card 生成（検索UI・結果表示・ページネーション）
│   │   ├── config.ts          # 環境変数・アプリ定数
│   │   ├── cosmos.ts          # Cosmos DB 操作（影響評価の保存・FAQ削除）
│   │   ├── excel.ts           # Excel 出力（要修正シナリオ一覧）
│   │   └── sharepoint.ts      # SharePoint Online アップロード
│   ├── appPackage/            # Teams アプリマニフェスト
│   ├── env/                   # 環境変数ファイル（.env.dev / .env.local）
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
    ├── 要件定義書.md           # 機能要件・非機能要件・システム構成
    ├── 導入手順書.md           # Azure 環境構築 Step 1〜15
    ├── 検索設計書.md           # 検索アルゴリズム・パラメータ・チューニング経緯
    ├── データベース設計書.md    # Cosmos DB スキーマ定義
    ├── シナリオ情報設計.md      # シナリオ/FAQ データ構造・Excel I/O 仕様
    └── screenshots/            # 手順書用スクリーンショット
```

## セットアップ

環境構築の詳細手順は [導入手順書](docs/導入手順書.md) を参照。

### 前提条件

- Node.js 18 / 20 / 22
- Azure サブスクリプション（リソース作成権限）
- Microsoft 365 テナント（Teams 管理者権限）
- Visual Studio Code + [M365 Agents Toolkit 拡張機能](https://aka.ms/teams-toolkit)

### 環境構築の流れ

1. **Azure リソース作成**（Step 1〜7）: リソースグループ、Azure OpenAI、AI Search、Cosmos DB、Key Vault、App Insights、App Service
2. **Bot プロジェクト設定**（Step 8〜10）: Toolkit でプロジェクト作成、Azure Bot 登録、サービスプリンシパル作成
3. **RBAC 設定**（Step 11）: Managed Identity 間のロール付与（7ロール）
4. **AI Search 設定**（Step 12）: インデックス・データソース・Skillset・Indexer の作成
5. **Bot 実装・デプロイ**（Step 13〜14）: ソースコード配置、Provision、Deploy、Publish
6. **動作確認**（Step 15）: Teams 上での検索テスト

### ローカル開発

```bash
cd maintenance-bot
npm install
# env/.env.local に環境変数を設定後:
npm run dev:teamsfx    # F5 デバッグ（VSCode から実行推奨）
```

必要な環境変数は [導入手順書 Step 13](docs/導入手順書.md) を参照。

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
          (Skillset 経由のみ)
```

- Bot から Azure OpenAI を**直接呼び出さない**（Embedding は AI Search Skillset/Vectorizer 経由）
- 全サービス間の認証は Managed Identity（RBAC）

## 検索の仕組み

詳細は [検索設計書](docs/検索設計書.md) を参照。

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

詳細は [データベース設計書](docs/データベース設計書.md)、[シナリオ情報設計](docs/シナリオ情報設計.md) を参照。

| コンテナ | 用途 | パーティションキー | 件数目安 |
|---------|------|-----------------|---------|
| `scenarios` | シナリオデータ（4カテゴリ） | `/categoryId` | 約 2,300件 |
| `faqs` | FAQ データ（3カテゴリ） | `/categoryId` | 約 18,700件 |
| `impactAssessments` | 影響評価結果の保存 | `/searchId` | — |

## ドキュメント一覧

| ドキュメント | 文書番号 | 内容 |
|------------|---------|------|
| [要件定義書](docs/要件定義書.md) | REQ-FAQ-IMPACT-002 | 機能要件 FR-001〜015、非機能要件、システム構成 |
| [導入手順書](docs/導入手順書.md) | SETUP-FAQ-IMPACT-001 | Azure 環境構築〜デプロイの全手順（Step 1〜15） |
| [検索設計書](docs/検索設計書.md) | DESIGN-SEARCH-001 | 検索アルゴリズム、パラメータ、チューニング経緯 |
| [データベース設計書](docs/データベース設計書.md) | DB-FAQ-IMPACT-001 | Cosmos DB スキーマ定義、combinedContent 生成ルール |
| [シナリオ情報設計](docs/シナリオ情報設計.md) | — | シナリオ/FAQ の JSON・Excel 入出力仕様 |

> 各ドキュメントの役割分担: 「何を作るか」→ 要件定義書、「どう構築するか」→ 導入手順書、「検索をどう設計したか」→ 検索設計書、「データをどう持つか」→ データベース設計書・シナリオ情報設計

