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
- 同一インデックス `impact-search-index` に両方書き込む（AI Search公式サポート）
- System Assigned MI 使用時、データソースの `identity` フィールドは省略

## 次のタスク: Azure環境構築 + スクショ付き手順書作成

### 目的
導入手順書の各Stepを実際のAzureポータルで実行し:
1. 手順の正確性を検証
2. スクリーンショットを取得
3. 不備があれば手順書を修正
4. 最終的にスクショ付き手順書 v2.0 を完成

### ツール構成（Claude Code + Chrome連携）

```
Claude Code（CLI / VS Code）
  ├── Chrome連携（--chrome フラグ）
  │   └── Azureポータル操作 + スクリーンショット取得
  ├── Azure MCP Server（補助）
  │   └── リソース設定値の自動検証
  └── ファイル操作
      └── 手順書MD直接編集 + スクショ保存
```

**Chrome連携セットアップ:**
```bash
# Claude Code起動時に --chrome フラグを付ける
claude --chrome

# または起動後に /chrome コマンドで有効化
/chrome
```

**Azure MCP Server セットアップ（補助、オプション）:**
```bash
# Azure CLIでログイン
az login

# Claude Codeに Azure MCP を追加
claude mcp add-json "Azure MCP Server" '{"command":"npx","args":["-y","@azure/mcp@latest","server","start"]}'
```

### 進め方
1. 作田さんが Chrome で Azureポータルにログイン
2. Claude Code に `--chrome` で Chrome連携を有効化
3. 手順書の Step 1 から順に実行
4. 各画面でスクリーンショット取得 → `docs/screenshots/` に保存
5. 実画面と手順書の記載を比較 → 差異があれば手順書を修正
6. 全Step完了後、スクショ付き手順書 v2.0 を完成

## コーディング規約・ドキュメント規約

### ドキュメント
- Excel/Word: Meiryo UI、本文10.5pt、見出し黒太字
- MD: テーブル形式維持、セル内改行は `<br>`、末尾改行1つ
- 設計書記法: 初出定義「正式名称（略称X）」→以降略称使用、出典 [1][2] 形式
- 図表番号: 「図2-1」形式（章番号+通番）

### 命名規則（Azureリソース）
- リソースグループ: `rg-impact-<env>`
- Azure OpenAI: `aoai-impact-<env>`
- AI Search: `srch-impact-<env>`
- Cosmos DB: `cosmos-impact-<env>`
- Key Vault: `kv-impact-<env>`
- Web App: `app-impact-bot-<env>`
- Bot Service: `bot-impact-<env>`
- PoC環境の `<env>` は `poc`
