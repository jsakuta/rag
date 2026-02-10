# Phase2 PoC Azure環境 導入手順書

**文書ID**: SETUP-FAQ-IMPACT-001

**版数**: 1.1

**作成日**: 2026/2/10

**作成者**: デジタル戦略部

**関連文書**: REQ-FAQ-IMPACT-002（Phase2 PoC 要件定義書 v3.1）

---

## 改訂履歴

| 版数 | 日付 | 変更内容 | 担当 |
|------|------|---------|------|
| 1.0 | 2026/2/10 | 初版作成 | デジタル戦略部 |
| 1.1 | 2026/2/10 | セルフレビューによる10件修正: ロール付与先修正（Azure OpenAI→AI Search MI）、データソース/Indexer設計を2系統に変更、SDK選定を要件定義書と整合、Key Vaultロール付与追加、Cosmos DB管理プレーンロール追加、不要な環境変数削除、Week 3注記追加 | デジタル戦略部 |

---

## 1. 概要

### 1.1 目的

本手順書は、Phase2 PoC「事務改定 影響候補検出システム」のAzure環境を構築するための手順を記載する。要件定義書（REQ-FAQ-IMPACT-002 v3.1）に基づき、Week 1〜2で必要なリソースの作成・設定を行う。

### 1.2 構築対象リソース

| # | リソース | SKU/プラン | 用途 |
|---|---------|-----------|------|
| 1 | リソースグループ | — | 全リソースの論理グループ |
| 2 | Azure OpenAI | Standard S0 | text-embedding-3-large（3,072次元） |
| 3 | Azure AI Search | Basic | ハイブリッド検索、Semantic Ranker、Indexer |
| 4 | Cosmos DB | Serverless（NoSQL API） | シナリオ・FAQ・影響判定マスタデータ |
| 5 | Key Vault | Standard | シークレット管理 |
| 6 | Application Insights | — | 監視・ログ |
| 7 | App Service（Web App） | Basic B1 | Botバックエンド |
| 8 | Azure Bot Service | F0（Free） | Teamsチャネル登録（Single-Tenant） |

### 1.3 全体構築フロー

```
Step 1: リソースグループ作成
  ↓
Step 2: Azure OpenAI 作成 + Embeddingモデルデプロイ
  ↓
Step 3: Azure AI Search 作成 + Semantic Ranker有効化
  ↓
Step 4: Cosmos DB 作成 + コンテナ作成
  ↓
Step 5: Key Vault 作成
  ↓
Step 6: Application Insights 作成
  ↓
Step 7: App Service (Web App) 作成 + Managed Identity有効化
  ↓
Step 8: Azure Bot Service 作成 + Teamsチャネル有効化
  ↓
Step 9: Managed Identityロール付与
  ↓
Step 10: AI Search インデックス・データソース・Skillset・Indexer設定
  ↓
Step 11: Botアプリケーションデプロイ
  ↓
Step 12: Teamsアプリ登録・サイドロード
  ↓
Step 13: 動作確認
```

---

## 2. 前提条件

### 2.1 Azureサブスクリプション

| 項目 | 要件 |
|------|------|
| サブスクリプション | 有効なAzureサブスクリプション |
| 権限 | サブスクリプションの「所有者」または「共同作成者」ロール |
| リージョン | Japan East（東日本）を推奨。Semantic RankerとAzure OpenAIが利用可能であること |
| Azure OpenAI利用申請 | 事前にAzure OpenAIの利用申請が承認済みであること |

### 2.2 Microsoft 365 / Teams

| 項目 | 要件 |
|------|------|
| Microsoft 365ライセンス | Teams利用可能なライセンス |
| Teams管理者権限 | カスタムアプリのサイドロードが許可されていること |
| Entra ID | アプリ登録の権限（アプリケーション管理者以上） |

### 2.3 開発環境

| ツール | バージョン | 用途 |
|--------|----------|------|
| Azure CLI | 2.60以上 | リソース作成・ロール付与 |
| Node.js | 20 LTS以上 | Botアプリケーション開発 |
| Visual Studio Code | 最新版 | 開発エディタ |
| M365 Agents Toolkit拡張機能 | 最新版 | VS Code拡張（旧Teams Toolkit） |

### 2.4 命名規則

本手順書では以下の命名規則を使用する。`<env>`はPoC環境を示す識別子（例: `poc`）に置き換えること。

| リソース | 命名例 |
|---------|--------|
| リソースグループ | `rg-impact-<env>` |
| Azure OpenAI | `aoai-impact-<env>` |
| Azure AI Search | `srch-impact-<env>` |
| Cosmos DB | `cosmos-impact-<env>` |
| Key Vault | `kv-impact-<env>` |
| Application Insights | `appi-impact-<env>` |
| App Service Plan | `asp-impact-<env>` |
| Web App | `app-impact-bot-<env>` |
| Azure Bot Service | `bot-impact-<env>` |

---

## 3. Step 1: リソースグループ作成

**Azureポータル手順:**

1. Azure Portal（https://portal.azure.com）にサインイン
2. 「リソースグループ」→「作成」を選択
3. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ名 | `rg-impact-poc` |
| リージョン | Japan East |

4. 「確認および作成」→「作成」を選択

**Azure CLI:**

```bash
az group create \
  --name rg-impact-poc \
  --location japaneast
```

---

## 4. Step 2: Azure OpenAI リソース作成

### 4.1 リソース作成

1. Azure Portal →「リソースの作成」→「Azure OpenAI」を検索
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| リージョン | Japan East |
| 名前 | `aoai-impact-poc` |
| 価格レベル | Standard S0 |

3. ネットワーク: 「すべてのネットワーク」（PoCのため。本番ではプライベートエンドポイント推奨）
4. 「確認して作成」→「作成」

**Azure CLI:**

```bash
az cognitiveservices account create \
  --name aoai-impact-poc \
  --resource-group rg-impact-poc \
  --kind OpenAI \
  --sku S0 \
  --location japaneast
```

### 4.2 Embeddingモデルのデプロイ

1. 作成したAzure OpenAIリソースに移動
2. 「モデルデプロイ」→「デプロイの管理」→ Azure AI Foundry（旧Azure OpenAI Studio）が開く
3. 「デプロイ」→「モデルのデプロイ」→「基本モデルをデプロイする」を選択
4. 以下を設定:

| 項目 | 値 |
|------|-----|
| モデル | text-embedding-3-large |
| デプロイ名 | `text-embedding-3-large` |
| デプロイの種類 | Standard |
| TPMレート制限 | 120K（推奨。Indexer実行中にスロットリングを避けるため） |

5. 「デプロイ」を選択

**注意:** デプロイ名はAI SearchのSkillset/Vectorizer設定で使用するため、正確に記録すること。

### 4.3 確認事項

- [ ] リソースが正常に作成された
- [ ] エンドポイントURLを記録した（例: `https://aoai-impact-poc.openai.azure.com/`）
- [ ] Embeddingモデルがデプロイ済み
- [ ] デプロイ名を記録した

---

## 5. Step 3: Azure AI Search 作成

### 5.1 リソース作成

1. Azure Portal →「リソースの作成」→「Azure AI Search」を検索
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| サービス名 | `srch-impact-poc` |
| 場所 | Japan East |
| 価格レベル | **Basic** |

3. 「確認して作成」→「作成」

**Azure CLI:**

```bash
az search service create \
  --name srch-impact-poc \
  --resource-group rg-impact-poc \
  --sku basic \
  --location japaneast
```

**価格レベルにBasic以上を選択する理由:**
Semantic RankerはBasic以上で利用可能。Free tierでは使用不可。

### 5.2 Semantic Rankerの有効化

1. 作成したAI Searchリソースに移動
2. 左メニュー →「設定」→「Semantic ranker」を選択
3. **Free**プランを選択（PoCでは月1,000クエリ以内のため十分）
4. 「プランの選択」を押下

**注記:** Free プランは月1,000クエリまで。超過した場合はStandardプラン（従量課金）に切り替え可能。PoCの検証規模ではFreeプランで十分。

### 5.3 Managed Identity（システム割り当て）の有効化

AI SearchのIndexerがCosmos DBに接続する際にManaged Identityを使用するため、有効化する。

1. AI Searchリソース →「設定」→「ID」を選択
2. 「システム割り当て」タブ →「状態」を**オン**に切り替え
3. 「保存」を選択

### 5.4 確認事項

- [ ] AI Searchリソースが正常に作成された（Basic SKU）
- [ ] Semantic Rankerが有効化された（Freeプラン）
- [ ] システム割り当てManaged Identityが有効
- [ ] サービスURLを記録した（例: `https://srch-impact-poc.search.windows.net`）
- [ ] 管理キーを記録した（Key Vault格納用）

---

## 6. Step 4: Cosmos DB 作成

### 6.1 リソース作成

1. Azure Portal →「リソースの作成」→「Azure Cosmos DB」を検索
2. 「Azure Cosmos DB for NoSQL」を選択 →「作成」
3. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| アカウント名 | `cosmos-impact-poc` |
| 場所 | Japan East |
| 容量モード | **サーバーレス** |

4. 「確認して作成」→「作成」

**Azure CLI:**

```bash
az cosmosdb create \
  --name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --locations regionName=japaneast \
  --capabilities EnableServerless
```

### 6.2 データベースとコンテナの作成

Cosmos DBリソース作成後、データエクスプローラーで以下を作成する。

**データベース作成:**

1. 「データ エクスプローラー」→「New Database」
2. Database id: `impact-db`

**コンテナ作成（3つ）:**

| コンテナ名 | パーティションキー | 用途 |
|-----------|-----------------|------|
| `scenarios` | `/categoryId` | シナリオマスタデータ |
| `faqs` | `/categoryId` | FAQマスタデータ |
| `impactAssessments` | `/searchId` | 要修正フラグの判定記録 |

各コンテナの作成手順:

1. `impact-db`を右クリック →「New Container」
2. Container id と Partition key を上表の通り入力
3. 「OK」を選択

**Azure CLI:**

```bash
# データベース作成
az cosmosdb sql database create \
  --account-name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --name impact-db

# scenariosコンテナ
az cosmosdb sql container create \
  --account-name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --database-name impact-db \
  --name scenarios \
  --partition-key-path /categoryId

# faqsコンテナ
az cosmosdb sql container create \
  --account-name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --database-name impact-db \
  --name faqs \
  --partition-key-path /categoryId

# impactAssessmentsコンテナ
az cosmosdb sql container create \
  --account-name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --database-name impact-db \
  --name impactAssessments \
  --partition-key-path /searchId
```

### 6.3 Soft Delete検知用プロパティの確認

`scenarios`と`faqs`コンテナのドキュメントには`isDeleted`フィールドを含める。AI Search Indexerはこのフィールドを参照してSoft Delete検知を行う。データ投入時にデフォルト値`false`を設定すること。

### 6.4 確認事項

- [ ] Cosmos DBアカウントが正常に作成された（Serverless）
- [ ] `impact-db`データベースが作成された
- [ ] 3つのコンテナが作成された（`scenarios`, `faqs`, `impactAssessments`）
- [ ] エンドポイントURLを記録した（例: `https://cosmos-impact-poc.documents.azure.com:443/`）

---

## 7. Step 5: Key Vault 作成

1. Azure Portal →「リソースの作成」→「Key Vault」を検索
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| Key Vault名 | `kv-impact-poc` |
| リージョン | Japan East |
| 価格レベル | Standard |
| アクセス許可モデル | Azure ロールベースのアクセス制御（RBAC） |

3. 「確認して作成」→「作成」

**シークレット登録（必要に応じて）:**

API キーをManaged Identityではなくキーで認証する場合、以下をシークレットとして登録する。

| シークレット名 | 値 |
|--------------|-----|
| `AiSearchAdminKey` | AI Searchの管理キー |
| `CosmosDbKey` | Cosmos DBの主キー（Managed Identity使用時は不要） |

**注記:** 本手順書ではManaged Identityによる認証を基本方針とするため、Key Vaultのシークレットは最小限とする。

---

## 8. Step 6: Application Insights 作成

1. Azure Portal →「リソースの作成」→「Application Insights」を検索
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| 名前 | `appi-impact-poc` |
| リージョン | Japan East |
| ワークスペース | 既存のLog Analyticsワークスペース、またはデフォルト |

3. 「確認して作成」→「作成」
4. 作成後、「接続文字列」を記録（Botアプリケーション設定で使用）

---

## 9. Step 7: App Service (Web App) 作成

### 9.1 App Serviceプランの作成

1. Azure Portal →「リソースの作成」→「App Service」を検索
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| 名前 | `app-impact-bot-poc` |
| 公開 | コード |
| ランタイムスタック | Node 20 LTS |
| オペレーティングシステム | Linux |
| 地域 | Japan East |
| App Serviceプラン | 新規作成: `asp-impact-poc`、SKU: **Basic B1** |

3. 「確認して作成」→「作成」

### 9.2 システム割り当てManaged Identityの有効化

1. 作成したWeb Appに移動
2. 左メニュー →「設定」→「ID」
3. 「システム割り当て」タブ → 状態を**オン**
4. 「保存」→ 確認ダイアログで「はい」
5. 表示される**オブジェクトID（プリンシパルID）** を記録（ロール付与で使用）

### 9.3 アプリケーション設定

Web App →「設定」→「環境変数」で以下を設定:

| 名前 | 値 | 備考 |
|------|-----|------|
| `MicrosoftAppId` | （Step 8で取得） | BotのアプリケーションID |
| `MicrosoftAppPassword` | （Step 8で取得） | Botのクライアントシークレット |
| `MicrosoftAppTenantId` | （テナントID） | Single-Tenant構成で必須 |
| `MicrosoftAppType` | `SingleTenant` | Single-Tenant構成を明示 |
| `AI_SEARCH_ENDPOINT` | `https://srch-impact-poc.search.windows.net` | |
| `AI_SEARCH_INDEX_NAME` | `impact-search-index` | |
| `COSMOS_DB_ENDPOINT` | `https://cosmos-impact-poc.documents.azure.com:443/` | |
| `COSMOS_DB_DATABASE` | `impact-db` | |
| `APPLICATIONINSIGHTS_CONNECTION_STRING` | （Step 6で取得） | |

**注記:**
- `MicrosoftAppId`と`MicrosoftAppPassword`はStep 8のBot Service作成後に設定する
- Azure OpenAI関連の環境変数（エンドポイント、デプロイ名）は不要。Embeddingの呼び出しはAI Search経由（ビルトインSkill/Vectorizer）で行い、BotアプリからAzure OpenAIを直接呼び出すことはない（要件定義書8.4参照）

### 9.4 確認事項

- [ ] Web Appが正常に作成された（Node 20 LTS, Linux, B1）
- [ ] Managed Identityが有効化され、オブジェクトIDを記録した
- [ ] メッセージングエンドポイントURL: `https://app-impact-bot-poc.azurewebsites.net/api/messages`

---

## 10. Step 8: Azure Bot Service 作成

### 10.1 Entra IDアプリ登録

Azure Bot Serviceを作成する前に、Single-Tenant用のアプリ登録を行う。

1. Azure Portal →「Microsoft Entra ID」→「アプリの登録」→「新規登録」
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| 名前 | `app-impact-bot-poc` |
| サポートされているアカウントの種類 | **この組織ディレクトリのみに含まれるアカウント（シングルテナント）** |
| リダイレクトURI | 空欄のまま |

3. 「登録」を選択
4. 以下を記録:
   - **アプリケーション（クライアント）ID** → `MicrosoftAppId`として使用
   - **ディレクトリ（テナント）ID** → `MicrosoftAppTenantId`として使用

### 10.2 クライアントシークレットの作成

1. 登録したアプリに移動 →「証明書とシークレット」
2. 「新しいクライアントシークレット」を選択
3. 説明: `bot-secret`、有効期限: 任意（PoCでは6ヶ月推奨）
4. 「追加」を選択
5. 表示される**値**を即座に記録（一度しか表示されない）→ `MicrosoftAppPassword`として使用

**重要:** シークレットの値はこの時点でしか確認できない。必ず安全な場所に記録すること。

### 10.3 Azure Bot Service リソース作成

1. Azure Portal →「リソースの作成」→「Azure Bot」を検索
2. 以下を入力:

| 項目 | 値 |
|------|-----|
| ボットハンドル | `bot-impact-poc` |
| サブスクリプション | 対象サブスクリプション |
| リソースグループ | `rg-impact-poc` |
| 価格レベル | **F0（Free）** |
| アプリの種類 | **シングルテナント** |
| 作成の種類 | 「既存のアプリの登録を使用する」 |
| アプリID | Step 10.1で取得したアプリケーションID |
| アプリテナントID | Step 10.1で取得したテナントID |

3. 「確認して作成」→「作成」

**注記:** Multi-Tenant構成は2025年7月31日以降、新規作成が非推奨となった。本手順ではSingle-Tenant構成を使用する。

### 10.4 メッセージングエンドポイントの設定

1. 作成したBot Serviceリソースに移動
2. 左メニュー →「構成」
3. メッセージングエンドポイントに以下を入力:

```
https://app-impact-bot-poc.azurewebsites.net/api/messages
```

4. 「適用」を選択

### 10.5 Teamsチャネルの有効化

1. Bot Serviceリソース →「チャネル」
2. 「Microsoft Teams」を選択
3. 利用規約に同意 → 「適用」
4. Teams チャネルが「実行中」になっていることを確認

### 10.6 MicrosoftAppId / Password の設定（Step 7に戻る）

Step 10.1〜10.2で取得した以下の値を、Step 9.3のWeb Appアプリケーション設定に登録する:

- `MicrosoftAppId` = アプリケーション（クライアント）ID
- `MicrosoftAppPassword` = クライアントシークレットの値
- `MicrosoftAppTenantId` = ディレクトリ（テナント）ID
- `MicrosoftAppType` = `SingleTenant`

### 10.7 確認事項

- [ ] Entra IDアプリ登録が完了した（Single-Tenant）
- [ ] クライアントシークレットを記録した
- [ ] Azure Bot Service（F0）が作成された
- [ ] メッセージングエンドポイントが設定された
- [ ] Teamsチャネルが有効化された
- [ ] Web Appのアプリケーション設定にAppId/Password/TenantIdを登録した

---

## 11. Step 9: Managed Identity ロール付与

各リソースのManaged Identityに対して、必要なアクセス権限を付与する。

### 11.1 ロール付与一覧

**AI SearchのManaged Identity向け:**

| 対象リソース | ロール名 | 付与先 | 用途 |
|-------------|---------|--------|------|
| Azure OpenAI | `Cognitive Services OpenAI User` | **AI Search**のManaged Identity | Skillset（インデクシング時）・Vectorizer（クエリ時）からのEmbedding呼び出し |
| Cosmos DB | `Cosmos DB Account Reader` | **AI Search**のManaged Identity | Indexerのデータソース接続メタデータ読み取り（管理プレーン） |
| Cosmos DB | `Cosmos DB Built-in Data Reader` | **AI Search**のManaged Identity | IndexerによるCosmos DBドキュメント読み取り（データプレーン） |

**Web AppのManaged Identity向け:**

| 対象リソース | ロール名 | 付与先 | 用途 |
|-------------|---------|--------|------|
| Azure AI Search | `Search Index Data Reader` | Web AppのManaged Identity | 検索クエリ実行 |
| Azure AI Search | `Search Service Contributor` | Web AppのManaged Identity | インデックス操作（任意） |
| Cosmos DB | `Cosmos DB Built-in Data Contributor` | Web AppのManaged Identity | FAQ削除・判定記録の書き込み（データプレーン） |
| Key Vault | `Key Vault Secrets User` | Web AppのManaged Identity | シークレット読み取り |

**注記:** BotアプリケーションからAzure OpenAIを直接呼び出すことはない（要件定義書8.4参照）。Embeddingの呼び出しはすべてAI Search経由（ビルトインSkill/Vectorizer）で行うため、`Cognitive Services OpenAI User`はAI SearchのMIに付与する。

### 11.2 Azure OpenAI ロール付与（Azureポータル）

**AI SearchのMIに付与する**（BotアプリからAzure OpenAIは直接呼び出さないため）。

1. Azure OpenAIリソースに移動
2. 「アクセス制御（IAM）」→「ロールの割り当ての追加」
3. ロール: `Cognitive Services OpenAI User`
4. メンバー: 「マネージドID」→ 対象のAI Search（`srch-impact-poc`）を選択
5. 「確認と割り当て」

**Azure CLI:**

```bash
# AI SearchのプリンシパルIDを取得
SEARCH_PRINCIPAL_ID=$(az search service show \
  --name srch-impact-poc \
  --resource-group rg-impact-poc \
  --query identity.principalId -o tsv)

# Azure OpenAIのリソースIDを取得
AOAI_RESOURCE_ID=$(az cognitiveservices account show \
  --name aoai-impact-poc \
  --resource-group rg-impact-poc \
  --query id -o tsv)

# AI SearchのMIにCognitive Services OpenAI Userを付与
az role assignment create \
  --assignee-object-id $SEARCH_PRINCIPAL_ID \
  --assignee-principal-type ServicePrincipal \
  --role "Cognitive Services OpenAI User" \
  --scope $AOAI_RESOURCE_ID
```

### 11.3 Azure AI Search ロール付与（Azureポータル）

1. AI Searchリソースに移動
2. 「アクセス制御（IAM）」→「ロールの割り当ての追加」
3. ロール: `Search Index Data Reader`
4. メンバー: Web AppのManaged Identityを選択
5. 「確認と割り当て」

**Azure CLI:**

```bash
# Web AppのプリンシパルIDを取得
WEB_APP_PRINCIPAL_ID=$(az webapp identity show \
  --name app-impact-bot-poc \
  --resource-group rg-impact-poc \
  --query principalId -o tsv)

SEARCH_RESOURCE_ID=$(az search service show \
  --name srch-impact-poc \
  --resource-group rg-impact-poc \
  --query id -o tsv)

az role assignment create \
  --assignee-object-id $WEB_APP_PRINCIPAL_ID \
  --assignee-principal-type ServicePrincipal \
  --role "Search Index Data Reader" \
  --scope $SEARCH_RESOURCE_ID
```

### 11.4 Key Vault ロール付与（Azureポータル）

1. Key Vaultリソースに移動
2. 「アクセス制御（IAM）」→「ロールの割り当ての追加」
3. ロール: `Key Vault Secrets User`
4. メンバー: Web AppのManaged Identityを選択
5. 「確認と割り当て」

**Azure CLI:**

```bash
KV_RESOURCE_ID=$(az keyvault show \
  --name kv-impact-poc \
  --resource-group rg-impact-poc \
  --query id -o tsv)

az role assignment create \
  --assignee-object-id $WEB_APP_PRINCIPAL_ID \
  --assignee-principal-type ServicePrincipal \
  --role "Key Vault Secrets User" \
  --scope $KV_RESOURCE_ID
```

### 11.5 Cosmos DB ロール付与（Azure CLI必須）

Cosmos DBには**管理プレーン**と**データプレーン**の2種類のロールがある。

**管理プレーンロール:**
AzureポータルのIAMから付与可能。AI SearchのIndexerがデータソース接続を確立する際に必要。

**データプレーンロール:**
AzureポータルのIAMには表示されない。Azure CLIまたはPowerShellで付与する必要がある。

**Built-in Data ContributorのロールID:** `00000000-0000-0000-0000-000000000002`

**Built-in Data ReaderのロールID:** `00000000-0000-0000-0000-000000000001`

#### 11.5.1 AI Search → Cosmos DB（管理プレーン: Account Reader）

AI SearchのIndexerがCosmos DBアカウントのメタデータを読み取るために必要。

1. Cosmos DBリソースに移動
2. 「アクセス制御（IAM）」→「ロールの割り当ての追加」
3. ロール: `Cosmos DB Account Reader Role`
4. メンバー: AI SearchのManaged Identity（`srch-impact-poc`）を選択
5. 「確認と割り当て」

**Azure CLI:**

```bash
COSMOS_RESOURCE_ID=$(az cosmosdb show \
  --name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --query id -o tsv)

az role assignment create \
  --assignee-object-id $SEARCH_PRINCIPAL_ID \
  --assignee-principal-type ServicePrincipal \
  --role "Cosmos DB Account Reader Role" \
  --scope $COSMOS_RESOURCE_ID
```

#### 11.5.2 AI Search → Cosmos DB（データプレーン: Data Reader）

```bash
az cosmosdb sql role assignment create \
  --account-name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --scope "/" \
  --principal-id $SEARCH_PRINCIPAL_ID \
  --role-definition-id 00000000-0000-0000-0000-000000000001
```

#### 11.5.3 Web App → Cosmos DB（データプレーン: Data Contributor）

```bash
az cosmosdb sql role assignment create \
  --account-name cosmos-impact-poc \
  --resource-group rg-impact-poc \
  --scope "/" \
  --principal-id $WEB_APP_PRINCIPAL_ID \
  --role-definition-id 00000000-0000-0000-0000-000000000002
```

**注意事項:**
- Cosmos DBのデータプレーンロールはAzureポータルのIAMには表示されない（管理プレーンのロールとは別系統のため）
- ロール付与が反映されるまで数分かかる場合がある
- データプレーンロール付与の確認: `az cosmosdb sql role assignment list --account-name cosmos-impact-poc --resource-group rg-impact-poc`
- 管理プレーンロール付与の確認: `az role assignment list --scope $COSMOS_RESOURCE_ID`

### 11.6 確認事項

- [ ] Azure OpenAI: **AI Search**に`Cognitive Services OpenAI User`を付与した
- [ ] AI Search: Web Appに`Search Index Data Reader`を付与した
- [ ] Key Vault: Web Appに`Key Vault Secrets User`を付与した
- [ ] Cosmos DB（管理プレーン）: AI Searchに`Cosmos DB Account Reader Role`を付与した
- [ ] Cosmos DB（データプレーン）: AI Searchに`Built-in Data Reader`を付与した（CLI）
- [ ] Cosmos DB（データプレーン）: Web Appに`Built-in Data Contributor`を付与した（CLI）
- [ ] 各ロール付与をCLIで確認した

---

## 12. Step 10: AI Search インデックス・データソース・Skillset・Indexer設定

AI Search REST APIを使用してインデクシングパイプラインを構築する。以下の順序で作成する。

### 12.1 概要

```
(1) インデックス定義 → (2) データソース → (3) Skillset → (4) Indexer
```

以下のREST API呼び出しでは、AI Searchの管理キー（または Managed Identity + Entra ID認証）を使用する。

**共通ヘッダー:**

```
Content-Type: application/json
api-key: <AI Searchの管理キー>
```

### 12.2 インデックス作成

```
PUT https://srch-impact-poc.search.windows.net/indexes/impact-search-index?api-version=2025-09-01
```

```json
{
  "name": "impact-search-index",
  "fields": [
    { "name": "id", "type": "Edm.String", "key": true, "filterable": true },
    { "name": "dataType", "type": "Edm.String", "filterable": true },
    { "name": "categoryId", "type": "Edm.String", "filterable": true },
    { "name": "categoryName", "type": "Edm.String", "searchable": true, "filterable": true, "analyzer": "ja.microsoft" },
    { "name": "title", "type": "Edm.String", "searchable": true, "analyzer": "ja.microsoft" },
    { "name": "content", "type": "Edm.String", "searchable": true, "analyzer": "ja.microsoft" },
    { "name": "combinedContent", "type": "Edm.String", "searchable": true, "analyzer": "ja.microsoft" },
    {
      "name": "contentVector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "dimensions": 3072,
      "vectorSearchProfile": "vector-profile"
    },
    { "name": "keywords", "type": "Collection(Edm.String)", "searchable": true, "filterable": true },
    { "name": "updatedAt", "type": "Edm.DateTimeOffset", "filterable": true, "sortable": true },
    { "name": "isDeleted", "type": "Edm.Boolean", "filterable": true }
  ],
  "vectorSearch": {
    "algorithms": [
      {
        "name": "hnsw-algorithm",
        "kind": "hnsw",
        "hnswParameters": {
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500,
          "metric": "cosine"
        }
      }
    ],
    "vectorizers": [
      {
        "name": "openai-vectorizer",
        "kind": "azureOpenAI",
        "azureOpenAIParameters": {
          "resourceUri": "https://aoai-impact-poc.openai.azure.com",
          "deploymentId": "text-embedding-3-large",
          "modelName": "text-embedding-3-large"
        }
      }
    ],
    "profiles": [
      {
        "name": "vector-profile",
        "algorithm": "hnsw-algorithm",
        "vectorizer": "openai-vectorizer"
      }
    ]
  },
  "semantic": {
    "configurations": [
      {
        "name": "semantic-config",
        "prioritizedFields": {
          "titleField": { "fieldName": "title" },
          "prioritizedContentFields": [
            { "fieldName": "content" },
            { "fieldName": "combinedContent" }
          ],
          "prioritizedKeywordsFields": [
            { "fieldName": "keywords" }
          ]
        }
      }
    ]
  }
}
```

### 12.3 データソース作成

要件定義書の設計に基づき、`scenarios`と`faqs`は別コンテナであるため、データソースを2つ作成する。

**データソース1: scenarios**

```
POST https://srch-impact-poc.search.windows.net/datasources?api-version=2025-09-01
```

```json
{
  "name": "cosmos-scenarios-ds",
  "type": "cosmosdb",
  "credentials": {
    "connectionString": "ResourceId=/subscriptions/<subscription-id>/resourceGroups/rg-impact-poc/providers/Microsoft.DocumentDB/databaseAccounts/cosmos-impact-poc;Database=impact-db;"
  },
  "container": {
    "name": "scenarios",
    "query": null
  },
  "dataChangeDetectionPolicy": {
    "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
    "highWaterMarkColumnName": "_ts"
  },
  "dataDeletionDetectionPolicy": {
    "@odata.type": "#Microsoft.Azure.Search.SoftDeleteColumnDeletionDetectionPolicy",
    "softDeleteColumnName": "isDeleted",
    "softDeleteMarkerValue": "true"
  }
}
```

**データソース2: faqs**

```
POST https://srch-impact-poc.search.windows.net/datasources?api-version=2025-09-01
```

```json
{
  "name": "cosmos-faqs-ds",
  "type": "cosmosdb",
  "credentials": {
    "connectionString": "ResourceId=/subscriptions/<subscription-id>/resourceGroups/rg-impact-poc/providers/Microsoft.DocumentDB/databaseAccounts/cosmos-impact-poc;Database=impact-db;"
  },
  "container": {
    "name": "faqs",
    "query": null
  },
  "dataChangeDetectionPolicy": {
    "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
    "highWaterMarkColumnName": "_ts"
  },
  "dataDeletionDetectionPolicy": {
    "@odata.type": "#Microsoft.Azure.Search.SoftDeleteColumnDeletionDetectionPolicy",
    "softDeleteColumnName": "isDeleted",
    "softDeleteMarkerValue": "true"
  }
}
```

**注記:**
- `connectionString`に`ResourceId`形式を使用することで、AI SearchのSystem Assigned Managed Identityで認証する（Step 11.5で`Cosmos DB Account Reader Role`+`Built-in Data Reader`を付与済み）
- `apiKey`や`identity`フィールドの指定は不要（System Assigned MI使用時は自動で適用される）
- `impactAssessments`コンテナはインデックス対象外のため、データソースを作成しない

### 12.4 Skillset作成

AzureOpenAIEmbeddingSkillを使用して、Indexer実行時に自動でベクトル化する。

```
PUT https://srch-impact-poc.search.windows.net/skillsets/impact-skillset?api-version=2025-09-01
```

```json
{
  "name": "impact-skillset",
  "skills": [
    {
      "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
      "name": "embedding-skill",
      "context": "/document",
      "resourceUri": "https://aoai-impact-poc.openai.azure.com",
      "deploymentId": "text-embedding-3-large",
      "modelName": "text-embedding-3-large",
      "dimensions": 3072,
      "inputs": [
        {
          "name": "text",
          "source": "/document/combinedContent"
        }
      ],
      "outputs": [
        {
          "name": "embedding",
          "targetName": "contentVector"
        }
      ]
    }
  ]
}
```

**注記:**
- `combinedContent`フィールド（title + content の結合テキスト）をベクトル化の入力とする。Cosmos DB側でドキュメント格納時に`combinedContent`を事前に生成しておく。
- 上記インデックス定義はWeek 1〜2のテキスト検索用。Week 3で画像検索を追加する際は、`imageVector`フィールド（`Collection(Edm.Single)`, 1,024次元, Azure Vision multimodal embedding用）と対応するvectorSearchプロファイルをインデックスに追加する。

### 12.5 Indexer作成

2つのデータソースに対応するIndexerをそれぞれ作成する。両方とも同一インデックス（`impact-search-index`）と同一Skillset（`impact-skillset`）を使用する。

**Indexer 1: scenarios**

```
PUT https://srch-impact-poc.search.windows.net/indexers/impact-scenarios-indexer?api-version=2025-09-01
```

```json
{
  "name": "impact-scenarios-indexer",
  "dataSourceName": "cosmos-scenarios-ds",
  "targetIndexName": "impact-search-index",
  "skillsetName": "impact-skillset",
  "schedule": {
    "interval": "PT1H"
  },
  "parameters": {
    "batchSize": 100,
    "maxFailedItems": -1,
    "maxFailedItemsPerBatch": -1
  },
  "fieldMappings": [
    { "sourceFieldName": "id", "targetFieldName": "id" },
    { "sourceFieldName": "dataType", "targetFieldName": "dataType" },
    { "sourceFieldName": "categoryId", "targetFieldName": "categoryId" },
    { "sourceFieldName": "categoryName", "targetFieldName": "categoryName" },
    { "sourceFieldName": "title", "targetFieldName": "title" },
    { "sourceFieldName": "content", "targetFieldName": "content" },
    { "sourceFieldName": "combinedContent", "targetFieldName": "combinedContent" },
    { "sourceFieldName": "keywords", "targetFieldName": "keywords" },
    { "sourceFieldName": "updatedAt", "targetFieldName": "updatedAt" },
    { "sourceFieldName": "isDeleted", "targetFieldName": "isDeleted" }
  ],
  "outputFieldMappings": [
    {
      "sourceFieldName": "/document/contentVector",
      "targetFieldName": "contentVector"
    }
  ]
}
```

**Indexer 2: faqs**

```
PUT https://srch-impact-poc.search.windows.net/indexers/impact-faqs-indexer?api-version=2025-09-01
```

```json
{
  "name": "impact-faqs-indexer",
  "dataSourceName": "cosmos-faqs-ds",
  "targetIndexName": "impact-search-index",
  "skillsetName": "impact-skillset",
  "schedule": {
    "interval": "PT1H"
  },
  "parameters": {
    "batchSize": 100,
    "maxFailedItems": -1,
    "maxFailedItemsPerBatch": -1
  },
  "fieldMappings": [
    { "sourceFieldName": "id", "targetFieldName": "id" },
    { "sourceFieldName": "dataType", "targetFieldName": "dataType" },
    { "sourceFieldName": "categoryId", "targetFieldName": "categoryId" },
    { "sourceFieldName": "categoryName", "targetFieldName": "categoryName" },
    { "sourceFieldName": "title", "targetFieldName": "title" },
    { "sourceFieldName": "content", "targetFieldName": "content" },
    { "sourceFieldName": "combinedContent", "targetFieldName": "combinedContent" },
    { "sourceFieldName": "keywords", "targetFieldName": "keywords" },
    { "sourceFieldName": "updatedAt", "targetFieldName": "updatedAt" },
    { "sourceFieldName": "isDeleted", "targetFieldName": "isDeleted" }
  ],
  "outputFieldMappings": [
    {
      "sourceFieldName": "/document/contentVector",
      "targetFieldName": "contentVector"
    }
  ]
}
```

**`schedule.interval: "PT1H"`** で1時間ごとにIndexerが実行される。初回はCosmos DBの全ドキュメントをインデクシングし、以降は`_ts`の変更分のみ処理する。

**注記:** 2つのIndexerが同一インデックスに書き込む構成はAI Searchで正式にサポートされている。各Indexerは独立したスケジュールで動作し、同一ドキュメント（同一`id`）を更新する場合は後勝ち（last-write-wins）となる。

### 12.6 Indexer手動実行と確認

両方のIndexerを手動実行する:

```
POST https://srch-impact-poc.search.windows.net/indexers/impact-scenarios-indexer/run?api-version=2025-09-01
POST https://srch-impact-poc.search.windows.net/indexers/impact-faqs-indexer/run?api-version=2025-09-01
```

実行状態の確認:

```
GET https://srch-impact-poc.search.windows.net/indexers/impact-scenarios-indexer/status?api-version=2025-09-01
GET https://srch-impact-poc.search.windows.net/indexers/impact-faqs-indexer/status?api-version=2025-09-01
```

`lastResult.status`が`success`であれば正常。`transientFailure`の場合はAzure OpenAIのTPM超過による一時的なエラーの可能性があり、再実行で解消する。

### 12.7 確認事項

- [ ] インデックス `impact-search-index` が作成された
- [ ] データソースが2つ作成された（`cosmos-scenarios-ds`, `cosmos-faqs-ds`）
- [ ] Skillsetが作成された
- [ ] Indexerが2つ作成され、初回実行が成功した
- [ ] 検索エクスプローラーでデータが検索できる

---

## 13. Step 11: Botアプリケーションのデプロイ

### 13.1 SDK選定

| SDK | 状態 | npmパッケージ | 備考 |
|-----|------|-------------|------|
| M365 Agents SDK | **GA**（JS, C#, Python） | `@microsoft/agents-hosting`, `@microsoft/agents-hosting-express` | 要件定義書の基本方針。Teams以外のチャネルにも対応可能 |
| Teams SDK（旧Teams AI Library v2） | GA（JS, C#） | `@microsoft/teams-ai` | Teams特化のBot開発SDK。M365 Agents SDKと併用可能 |
| Bot Framework SDK | **サポート終了**（2025/12/31） | `botbuilder` | 新規開発には使用しない |

要件定義書に基づき、本PoCでは**M365 Agents SDK**（TypeScript）を基本方針とする。Teams固有機能が必要な場合は`@microsoft/agents-hosting-extensions-teams`を併用する。

### 13.2 プロジェクト初期化

```bash
mkdir impact-bot && cd impact-bot
npm init -y
npm install @microsoft/agents-hosting @microsoft/agents-hosting-express
npm install @microsoft/agents-hosting-extensions-teams
npm install @azure/cosmos @azure/identity
npm install typescript ts-node @types/node --save-dev
npx tsc --init
```

**最小限のエントリポイント（index.mjs）:**

```javascript
import { startServer } from '@microsoft/agents-hosting-express'
import { AgentApplication, MemoryStorage } from '@microsoft/agents-hosting'

const app = new AgentApplication({
  storage: new MemoryStorage()
})

app.activity('message', async (context) => {
  // ここに検索ロジック・Adaptive Card生成ロジックを実装
  await context.sendActivity(`受信: ${context.activity.text}`)
})

startServer(app)
```

**注記:** 上記は動作確認用の最小構成。検索ロジック、Adaptive Card生成、Cosmos DB書き込み等の実装詳細は別途実装ガイドを参照すること。

### 13.3 デプロイ方法

VS CodeのM365 Agents Toolkit拡張機能を使用するか、Azure CLIでデプロイする。

**Azure CLI でのデプロイ:**

```bash
# ZIPデプロイ
cd impact-bot
zip -r deploy.zip . -x "node_modules/*" ".git/*"
az webapp deploy \
  --resource-group rg-impact-poc \
  --name app-impact-bot-poc \
  --src-path deploy.zip \
  --type zip
```

### 13.4 確認事項

- [ ] Botアプリケーションがデプロイされた
- [ ] Web Appのログストリームでエラーが出ていない
- [ ] `/api/messages`エンドポイントが応答している

**注記:** 本ステップは環境構築（デプロイ先の準備と初回デプロイ）を対象とする。検索ロジック、Adaptive Card生成、Cosmos DB書き込み、エラーハンドリング等の実装詳細は別途作成する実装ガイドで扱う。

---

## 14. Step 12: Teamsアプリ登録・サイドロード

### 14.1 アプリマニフェストの作成

`manifest.json`を以下の内容で作成する:

```json
{
  "$schema": "https://developer.microsoft.com/en-us/json-schemas/teams/v1.17/MicrosoftTeams.schema.json",
  "manifestVersion": "1.17",
  "version": "1.0.0",
  "id": "<MicrosoftAppId>",
  "developer": {
    "name": "デジタル戦略部",
    "websiteUrl": "https://chibabank.co.jp",
    "privacyUrl": "https://chibabank.co.jp/privacy",
    "termsOfUseUrl": "https://chibabank.co.jp/terms"
  },
  "name": {
    "short": "影響候補検出Bot",
    "full": "事務改定 影響候補検出システム (Phase2 PoC)"
  },
  "description": {
    "short": "事務改定時の影響候補をAI検索で検出します",
    "full": "改定内容をテキスト入力すると、影響を受ける可能性のあるシナリオとFAQを検出し、一覧表示します。"
  },
  "icons": {
    "outline": "outline.png",
    "color": "color.png"
  },
  "accentColor": "#2B5292",
  "bots": [
    {
      "botId": "<MicrosoftAppId>",
      "scopes": ["personal"],
      "supportsFiles": false,
      "isNotificationOnly": false,
      "commandLists": [
        {
          "scopes": ["personal"],
          "commands": [
            {
              "title": "help",
              "description": "使い方を表示します"
            }
          ]
        }
      ]
    }
  ],
  "permissions": ["identity", "messageTeamMembers"],
  "validDomains": [
    "app-impact-bot-poc.azurewebsites.net"
  ]
}
```

`<MicrosoftAppId>`をStep 10.1で取得したアプリケーションIDに置き換えること。

### 14.2 アプリパッケージの作成

以下の3ファイルをZIPファイルにまとめる:

- `manifest.json`
- `color.png`（192x192ピクセルのカラーアイコン）
- `outline.png`（32x32ピクセルの透明背景アウトラインアイコン）

```bash
zip impact-bot-app.zip manifest.json color.png outline.png
```

### 14.3 Teamsへのサイドロード

1. Microsoft Teamsを開く
2. 左サイドバーの「アプリ」→「アプリを管理」→「アプリのアップロード」
3. 「カスタムアプリをアップロード」を選択（組織の管理者がサイドロードを許可している必要がある）
4. 作成した`impact-bot-app.zip`を選択
5. アプリの詳細が表示されたら「追加」を選択

**注記:** サイドロードが無効な場合、Teams管理者に依頼してTeams管理センター（https://admin.teams.microsoft.com）→「Teamsアプリ」→「セットアップポリシー」でカスタムアプリのアップロードを許可する必要がある。

### 14.4 確認事項

- [ ] アプリマニフェストが正しく作成された
- [ ] アプリパッケージ（ZIP）が作成された
- [ ] Teamsにサイドロードでアプリがインストールされた
- [ ] 1:1チャットでBotに話しかけられる

---

## 15. Step 13: 動作確認チェックリスト

### 15.1 基本通信確認

| # | 確認項目 | 手順 | 期待結果 |
|---|---------|------|---------|
| 1 | Bot応答 | Teamsで1:1チャットを開き「hello」と入力 | Botが応答メッセージを返す |
| 2 | エラーログ確認 | Application InsightsまたはWeb Appのログストリームを確認 | エラーが出ていない |

### 15.2 検索機能確認

| # | 確認項目 | 手順 | 期待結果 |
|---|---------|------|---------|
| 3 | テキスト検索 | 改定内容を入力（例:「口座開設の本人確認書類が変更」） | Adaptive Cardで候補一覧が表示される |
| 4 | タブ切り替え | シナリオタブ/FAQタブのボタンを押す | ToggleVisibilityで表示が切り替わる |
| 5 | スクロール表示 | 候補数が多い検索結果を確認 | スクロールバーが表示され、全件を確認できる |

### 15.3 書き込み機能確認

| # | 確認項目 | 手順 | 期待結果 |
|---|---------|------|---------|
| 6 | FAQ削除 | FAQタブでチェック →「選択したFAQを削除」→ 確認 →「削除実行」 | 完了カードが表示され、Cosmos DBの`isDeleted`が`true`に更新 |
| 7 | 要修正フラグ | シナリオタブでチェック →「要修正を保存」 | 完了カードが表示され、Cosmos DBに`impactAssessments`レコードが作成 |
| 8 | Indexer反映 | FAQ削除後、Indexerを手動実行し再検索 | 削除したFAQが検索結果から除外 |

### 15.4 トラブルシューティング

| 症状 | 原因の可能性 | 対処法 |
|------|------------|--------|
| Botが応答しない | メッセージングエンドポイントの設定誤り | Bot Service「構成」のURLを確認 |
| 403 Forbidden | Managed Identityのロール付与不足 | Step 11のロール付与を再確認 |
| Indexer失敗 | Azure OpenAI TPM超過 | Embeddingモデルのレート制限を引き上げ |
| Indexer失敗（Cosmos DB） | データソースのManaged Identity設定不備 | AI SearchのMI→Cosmos DB Data Readerを確認 |
| 検索結果が0件 | Indexerが未実行/データ未投入 | Indexerを手動実行、Cosmos DBのデータを確認 |
| Adaptive Cardが表示されない | カードサイズ28KB超過 | 候補件数を確認し、ページネーションフォールバックを検討 |

---

## 付録A: リソースID・エンドポイント記録シート

環境構築完了後、以下の情報を記録して管理すること。

| 項目 | 値 |
|------|-----|
| リソースグループ名 | |
| Azure OpenAI エンドポイント | |
| Azure OpenAI デプロイ名 | |
| AI Search エンドポイント | |
| AI Search 管理キー | |
| AI Search インデックス名 | |
| Cosmos DB エンドポイント | |
| Cosmos DB データベース名 | |
| Key Vault名 | |
| Application Insights 接続文字列 | |
| Web App URL | |
| Web App Managed Identity オブジェクトID | |
| Bot Service アプリID（MicrosoftAppId） | |
| Bot Service テナントID | |
| Bot Service クライアントシークレット | |

---

## 付録B: 月額コスト概算

| リソース | 月額概算 | 備考 |
|---------|---------|------|
| Azure AI Search (Basic) | ~¥10,000 | Semantic Ranker含む |
| Azure OpenAI (S0) | ~¥2,000 | Embedding利用のみ |
| Azure Web App (B1) | ~¥2,000 | Linux, 1コア, 1.75GB |
| Cosmos DB (Serverless) | ~¥1,000 | RU消費量依存 |
| Key Vault (Standard) | ~¥500 | |
| Application Insights | ~¥500 | |
| Azure Bot Service (F0) | ¥0 | Free |
| **合計** | **~¥16,000/月** | |

---

## 付録C: 参考リンク

| リソース | URL |
|---------|-----|
| Azure Bot Service（Single-Tenant作成） | https://learn.microsoft.com/en-us/azure/bot-service/abs-quickstart |
| Teams SDK（旧Teams AI Library） | https://github.com/microsoft/teams-sdk |
| M365 Agents SDK | https://github.com/microsoft/Agents |
| M365 Agents Toolkit（旧Teams Toolkit） | https://github.com/OfficeDev/microsoft-365-agents-toolkit |
| AI Search Integrated Vectorization | https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization |
| AI Search Cosmos DB Indexer | https://learn.microsoft.com/en-us/azure/search/search-how-to-index-cosmosdb-sql |
| AI Search Managed Identity設定 | https://learn.microsoft.com/en-us/azure/search/search-how-to-managed-identities |
| Semantic Ranker有効化 | https://learn.microsoft.com/en-us/azure/search/semantic-how-to-enable-disable |
| Cosmos DB RBAC設定 | https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/how-to-connect-role-based-access-control |
| Adaptive Card スクロールコンテナ | https://learn.microsoft.com/en-us/microsoftteams/platform/task-modules-and-cards/cards/cards-format |