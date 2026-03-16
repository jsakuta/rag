# maintenance-bot

事務改定影響検知 Teams Bot アプリケーション。

[M365 Agents SDK](https://github.com/Microsoft/Agents) をベースに、Azure AI Search へのハイブリッド検索と Adaptive Card による結果表示を実装している。

## ソースコード構成

| ファイル | 役割 |
|---------|------|
| `src/index.ts` | Express サーバー起動 + Agent Server セットアップ |
| `src/agent.ts` | Bot メインロジック（メッセージ受信・Action.Execute ハンドラ） |
| `src/cards.ts` | Adaptive Card 生成（検索UI・結果表示・ページネーション・タブ切替） |
| `src/config.ts` | 環境変数・検索パラメータ・カテゴリ定義 |
| `src/cosmos.ts` | Cosmos DB 操作（影響評価保存・FAQ 削除） |
| `src/excel.ts` | ExcelJS による要修正シナリオ一覧の Excel 出力 |
| `src/sharepoint.ts` | SharePoint Online への Excel ファイルアップロード |

## 環境変数

| 変数名 | 用途 |
|--------|------|
| `AI_SEARCH_ENDPOINT` | Azure AI Search エンドポイント URL |
| `AI_SEARCH_INDEX_NAME` | インデックス名（デフォルト: `maintenance-search-index`） |
| `COSMOS_DB_ENDPOINT` | Cosmos DB エンドポイント URL |
| `COSMOS_DB_DATABASE` | データベース名（デフォルト: `maintenance-db`） |
| `SPO_SITE_ID` | SharePoint サイト ID |
| `SPO_DRIVE_ID` | SharePoint ドキュメントライブラリ ID |
| `SPO_UPLOAD_FOLDER` | アップロード先フォルダ名（デフォルト: `影響候補シナリオ`） |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | Application Insights 接続文字列（設定時のみ監視有効化） |

認証は Managed Identity を使用するため、API キーの設定は不要。ローカル開発時は `DefaultAzureCredential` が Azure CLI ログインを自動検出する。

## ローカル開発

```bash
npm install
# VSCode で F5（Debug in Teams）を推奨
# または:
npm run dev:teamsfx
```

## ビルド・デプロイ

```bash
npm run build          # TypeScript コンパイル
npm start              # コンパイル済み JS を実行
```

デプロイは M365 Agents Toolkit の Provision → Deploy → Publish フローで実行する。詳細は [導入手順書](../docs/導入手順書.md) Step 12〜13 を参照。

## Toolkit 設定ファイル

| ファイル | 用途 |
|---------|------|
| `m365agents.yml` | Provision / Deploy の定義（Azure 環境用） |
| `m365agents.local.yml` | ローカルデバッグ用オーバーライド |
