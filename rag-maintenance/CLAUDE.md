# CLAUDE.md — Phase2 PoC 影響候補検出システム

> このファイルは Claude Code が起動時に自動で読み込むプロジェクトコンテキストです。
> 最終更新: 2026-02-10

---

## Azure環境 リソース保護ルール（絶対順守）

**既存のAzureリソースを削除・変更してはならない。**

- 現在のサブスクリプション内の既存リソースグループ・リソースを一切削除・変更しない
- `rg-impact-poc` 内のリソースのみ新規作成する
- 他のリソースグループへの操作は禁止
- `az group delete`、`az resource delete` 等の破壊的コマンドは実行しない
- ロール付与は手順書に記載されたスコープのみに限定する

**Azure CLIステータス:** ログイン済み（`admin@bdxcorp.onmicrosoft.com` / `Azure subscription 1`）

---

## プロジェクト概要

千葉銀行 デジタル戦略部（B&DX）の Phase2 PoC プロジェクト。
事務改定時にシナリオ・FAQ への影響候補を AI 検索（ハイブリッド検索 + Semantic Ranker）で検出する Teams Bot システムを構築する。

## ファイル構成

```
rag-maintenance/
├── CLAUDE.md                 ← このファイル
├── docs/
│   ├── phase2-poc-requirements-definition.md   # 要件定義書 v3.2
│   └── phase2-poc-setup-guide.md               # 導入手順書 v1.1
└── (今後追加)
    ├── screenshots/          # Azure構築時のスクリーンショット
    └── src/                  # Botアプリケーションソース
```

## 要件定義書（v3.2）概要

- 文書番号: REQ-FAQ-IMPACT-002
- Azure リソース: OpenAI (S0) / AI Search (Basic) / Cosmos DB (Serverless) / Bot Service (F0, Single-Tenant) / Web App (B1)
- 検索方式: ハイブリッド検索（テキスト + ベクトル 3,072次元） + Semantic Ranker
- SDK: **M365 Agents SDK**（`@microsoft/agents-hosting`）が基本方針
- Cosmos DB コンテナ: `scenarios`、`faqs`（別コンテナ、パーティションキー `/categoryId`）、`impactAssessments`
- UI: Adaptive Card（ToggleVisibility でタブ切替、ScrollArea でスクロール）
- 8.4節: **BotアプリからAzure OpenAIを直接呼び出さない**。EmbeddingはすべてAI Search経由（Skillset/Vectorizer）

## 導入手順書（v1.1）概要

- 文書ID: SETUP-FAQ-IMPACT-001
- 全13ステップ、約1,300行
- v1.0 → v1.1 でセルフレビューにより10件修正済み

### v1.1 修正済み内容

| # | 重大度 | 修正内容 | 状態 |
|---|--------|---------|------|
| ① | 🔴 | Azure OpenAIの`Cognitive Services OpenAI User`付与先: Web App → **AI Search** MI | ✅済 |
| ② | 🔴 | AI Search → Azure OpenAI ロール付与手順の新規追加 | ✅済 |
| ③ | 🔴 | データソース2系統分離（`cosmos-scenarios-ds` + `cosmos-faqs-ds`）、Indexerも2つ | ✅済 |
| ④ | 🟡 | SDK選定を M365 Agents SDK に統一（`@microsoft/agents-hosting`） | ✅済 |
| ⑤ | 🟡 | Web App環境変数から `AZURE_OPENAI_*` 削除（Bot→OpenAI直接呼出なし） | ✅済 |
| ⑥ | 🟢 | インデックス定義にWeek 3 `imageVector`追加注記 | ✅済 |
| ⑦ | 🟢 | Key Vault `Key Vault Secrets User` ロール付与追加 | ✅済 |
| ⑧ | 🟢 | データソースの `identity` フィールド削除（System Assigned MI時は省略が正） | ✅済 |
| ⑨ | 🟢 | Cosmos DB 管理プレーン `Cosmos DB Account Reader Role` 追加 | ✅済 |
| ⑩ | 🟢 | Bot実装ステップに「詳細は別途実装ガイド参照」注記 | ✅済 |

## 技術的な重要決定事項

### SDK選定経緯
- 要件定義書: M365 Agents SDK を基本方針
- WEB調査結果: M365 Agents SDK は GA（JS/C#/Python）。npmパッケージは `@microsoft/agents-hosting` + `@microsoft/agents-hosting-express`。Teams拡張は `@microsoft/agents-hosting-extensions-teams`
- Teams SDK（旧 Teams AI Library v2、`@microsoft/teams-ai`）も選択肢だが、要件定義書に合わせ M365 Agents SDK を採用
- Bot Framework SDK は 2025/12/31 サポート終了、新規開発には使用しない

### RBAC設計のポイント
- AI Search の MI → Azure OpenAI: `Cognitive Services OpenAI User`（Skillset/Vectorizer用）
- AI Search の MI → Cosmos DB: `Cosmos DB Account Reader Role`（管理プレーン）+ `Built-in Data Reader`（データプレーン、CLI必須）
- Web App の MI → AI Search: `Search Index Data Reader`
- Web App の MI → Cosmos DB: `Built-in Data Contributor`（データプレーン、CLI必須）
- Web App の MI → Key Vault: `Key Vault Secrets User`
- **Web App → Azure OpenAI のロールは不要**（直接呼出なし）

### データソース設計
- `scenarios` と `faqs` は **別コンテナ** → データソース2つ + Indexer2つ
- 同一インデックス `maintenance-search-index` に両方書き込む（AI Search公式サポート）
- System Assigned MI 使用時、データソースの `identity` フィールドは省略

## Azure環境構築 進捗（2026-02-10）

### ✅ 完了（Step 1〜7, 9〜10）
- リソースグループ `rg-maintenance-poc` + 全サービス8リソース作成済み
- RBAC 7ロール（管理プレーン5 + データプレーン2）付与済み
- AI Search: インデックス + DS2 + Skillset + Indexer2 設定済み
- 詳細値はメモリ参照: `memory/azure-deployment-progress.md`

### ⬜ 未完了（次セッションで実施）
1. **Step 8: Entra IDアプリ登録 + Bot Service**（CLI or ポータル）
   - `az ad app create --display-name app-maintenance-bot-poc --sign-in-audience AzureADMyOrg`
   - クライアントシークレット作成、Bot Service(F0)作成、Teamsチャネル有効化
   - Web Appアプリケーション設定に AppId/Password/TenantId 登録
2. **Step 11: Botアプリデプロイ**（M365 Agents SDK）
3. **Step 12-13: Teamsサイドロード + 動作確認**
4. **スクショ付き手順書 v2.0 完成**（最終的にWord化）

## コーディング規約・ドキュメント規約

### ドキュメント
- Excel/Word: Meiryo UI、本文10.5pt、見出し黒太字
- MD: テーブル形式維持、セル内改行は `<br>`、末尾改行1つ
- 設計書記法: 初出定義「正式名称（略称X）」→以降略称使用、出典 [1][2] 形式
- 図表番号: 「図2-1」形式（章番号+通番）

### 命名規則（Azureリソース）
- リソースグループ: `rg-maintenance-<env>`
- Azure OpenAI: `aoai-maintenance-<env>`
- AI Search: `srch-maintenance-<env>`
- Cosmos DB: `cosmos-maintenance-<env>`
- Key Vault: `kv-maintenance-<env>`
- Web App: `app-maintenance-bot-<env>`
- Bot Service: `bot-maintenance-<env>`
- PoC環境の `<env>` は `poc`
