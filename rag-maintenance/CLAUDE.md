# CLAUDE.md — Phase2 PoC 影響候補検出システム

> このファイルは Claude Code が起動時に自動で読み込むプロジェクトコンテキストです。
> 最終更新: 2026-02-12

---

## Azure環境 リソース保護ルール（絶対順守）

**既存のAzureリソースを削除・変更してはならない。**

- 現在のサブスクリプション内の既存リソースグループ・リソースを一切削除・変更しない
- `rg-maintenance-poc` 内のリソースのみ新規作成する
- 他のリソースグループへの操作は禁止
- `az group delete`、`az resource delete` 等の破壊的コマンドは実行しない
- ロール付与は手順書に記載されたスコープのみに限定する

**Azure CLIステータス:** ログイン済み（`admin@bdxcorp.onmicrosoft.com` / `Azure subscription 1`）

---

## 手順書メンテナンスルール（必須）

### 実装後の手順書反映判断

実装・修正・トラブルシューティングを行った後は、以下を必ず判断する:

1. **顧客が引き継ぎ時に実行すべき手順か？** → `docs/導入手順書.md` に追記・修正する
   - 環境変数の設定変更、ロール付与、パッケージ変更、設定ファイルの編集 → 手順書に反映
   - 開発時のみのデバッグ修正、一時的なワークアラウンド → 手順書には不要
2. **判断基準**: 「顧客が新規環境でゼロから構築する際に、この情報がないと詰まるか？」がYesなら手順書に記載

### 手順書変更後の整合性レビュー（必須）

`docs/導入手順書.md` を変更した後は、**手順書全体を通読して以下を確認すること**:

- セクション番号の連番が正しいか（重複・欠番がないか）
- 変数名（`$DEVELOPER_OID` 等）が文書全体で統一されているか
- 前のStepで定義した値が後のStepで正しく参照されているか
- 新規追加した手順が他セクションと矛盾していないか
- トラブルシューティング表に関連するエントリが追加されているか

---

## プロジェクト概要

千葉銀行 デジタル戦略部（B&DX）の Phase2 PoC プロジェクト。
事務改定時にシナリオ・FAQ への影響候補を AI 検索（ハイブリッド検索 + Semantic Ranker）で検出する Teams Bot システムを構築する。

## ファイル構成

```
rag-maintenance/
├── CLAUDE.md                 ← このファイル
├── docs/
│   ├── 要件定義書.md                            # 要件定義書 v3.3
│   ├── 導入手順書.md                            # 導入手順書 v1.4
│   └── screenshots/                            # スクリーンショット
├── scripts/                  # AI Search設定用JSON（index, datasource, skillset, indexer）
├── maintenance-bot/          # Botアプリケーション（M365 Agents Toolkit生成）
└── (今後追加)
```

## 要件定義書（v3.3）概要

- 文書番号: REQ-FAQ-IMPACT-002
- Azure リソース: OpenAI (S0) / AI Search (Basic) / Cosmos DB (Serverless) / Bot Service (F0, Single-Tenant) / Web App (B1)
- 検索方式: ハイブリッド検索（テキスト + ベクトル 3,072次元） + Semantic Ranker
- SDK: **M365 Agents SDK**（`@microsoft/agents-hosting`）が基本方針
- Cosmos DB コンテナ: `scenarios`、`faqs`（別コンテナ、パーティションキー `/categoryId`）、`impactAssessments`
- UI: Adaptive Card（ToggleVisibility でタブ切替、ページネーション）
- FR-015: シナリオ要修正Excel出力（要修正行を黄色ハイライト、ExcelJS + インライン添付）
- FR-006: 差分ベクトル化（High Water Mark方式で変更分のみ自動再ベクトル化）
- 8.4節: **BotアプリからAzure OpenAIを直接呼び出さない**。EmbeddingはすべてAI Search経由（Skillset/Vectorizer）

## 導入手順書（v1.4）概要

- 文書ID: SETUP-FAQ-IMPACT-001
- 全13ステップ（Step 1: RG作成 〜 Step 13: 動作確認）
- v1.1: セルフレビュー10件修正（ロール付与先修正、DS分離、SDK統一 等）— 全件反映済み
- v1.3: Step 8/11/12をM365 Agents Toolkitベースに書換え、F5デバッグ追加
- v1.4: Toolkit v6系対応（`teamsapp.yml`→`m365agents.yml`、テンプレート選択フロー修正）

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

### 検索UIの設計原則
- シナリオとFAQは「同じデータカテゴリ」として統一的に扱う
- 7カテゴリ（シナリオ4 + FAQ3）をグループ表示し、1回の検索で選択カテゴリを横断検索する
- 検索ボタンは1セット（ハイブリッド検索 + キーワード一致検索）のみ。セクション別に分離しない
- 検索結果は全タイプをスコア順にマージし、統一ページネーション（1系統）で表示
- 同一ページにシナリオとFAQが混在する場合はタイプ別セクションで表示
- アクションボタン（「要修正を保存」「選択したFAQを削除」）は結果内容に応じて両方表示

## 基本方針

**運用保守効率化AIの導入がメインタスク。** Azure環境構築 + Bot実装 + 手順書完成を並行して進める。

- 手順書の該当Stepを読む → 実行 → 差異があれば手順書修正 → 次のStepへ
- スクリーンショットは `docs/screenshots/` に保存
- 実装後は「手順書メンテナンスルール」に従い手順書反映を判断する

---

## Azure環境構築 進捗（2026-02-12）

### ✅ 完了（Step 1〜12）

- リソースグループ `rg-maintenance-poc` + 全サービス8リソース作成済み
- RBAC 7ロール（管理プレーン5 + データプレーン2）付与済み
- AI Search: インデックス + DS2 + Skillset + Indexer2 設定済み
- Bot: M365 Agents SDK 実装 + Toolkit Provision/Deploy/Publish 完了
- 詳細値はメモリ参照: `memory/azure-deployment-progress.md`

### 🔧 Step 13: 動作確認（進行中）

- **F5ローカルデバッグ起動**: ✅ 完了（遅延初期化修正、AZURE_OPENAI_*削除、.localConfigs設定済み）
- **Bot基本通信（カード表示）**: ✅ 完了（モード選択カード表示まで動作確認）
- **検索機能**: 🔧 403→開発者にSearch Index Data Readerロール付与済み（反映待ち）
- **Cosmos DBテストデータ投入**: ✅ 投入済み（scenarios 2,318件 + faqs 18,744件 = 計21,062件、AI Searchインデックス反映済み）
- **想定外データ**: `cat-yokin`/`cat-kawase`/`cat-yushi` の15件が初期テスト残骸として残存（要削除）
- **Adaptive Card UIテスト**: ⬜ 未着手
- **FR-015 Excel出力**: ⬜ 未実装

### ⬜ 未完了

1. **Step 13残タスク**: 想定外データ削除 + 検索テスト + UI確認
2. **FR-015実装**: シナリオ要修正Excel出力（ExcelJS + インライン添付）
3. **手順書 v2.0 完成**（スクショ付き、最終Word化）

## コーディング規約・ドキュメント規約

### TypeScript（maintenance-bot）
- **`ts-node` は `tsc --noEmit` より厳格**: `Error as Record<string, unknown>` は TS2352 エラー。`as unknown as Record<string, unknown>` で二段キャストが必要
- **M365 Agents SDK の `actionExecute` コールバック**: 第3引数 `data` は Action全体（`{type, title, verb, data: {...}}`）が渡される。ユーザーデータは `data.data.query` でアクセスする（`data.query` ではない）
- **Azure SDKクライアントは遅延初期化必須**: `SearchClient` / `CosmosClient` をモジュールトップレベルで初期化すると `env-cmd` の環境変数読み込み前に評価され `Invalid URL` エラーになる

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
