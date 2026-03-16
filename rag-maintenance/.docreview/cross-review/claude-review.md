# クロスドキュメント整合性レビュー

**レビュー日**: 2026-03-16
**レビュアー**: Claude Opus 4.6
**対象文書**:
- 要件定義書.md（以下「要件」）
- 検索設計書.md（以下「検索」）
- 導入手順書.md（以下「手順」）

---

## 検出された不整合

### CONS-001: [MEDIUM] インデックス定義の `keywords` フィールドにアナライザー記載漏れ
- **文書A**: 要件定義書 7.2 インデックス設計（Line 688付近）
  > `keywords | Collection(Edm.String) | ✓ | — | — | キーワード`
- **文書B**: 検索設計書 10-B（Line 474付近）
  > `keywords フィールド改善 | FAQ に質問文・回答文を追加、ja.microsoft アナライザ適用`
- **実装**: `scripts/index-definition.json` Line 18
  > `"analyzer": "ja.microsoft"`
- **矛盾**: 要件定義書のインデックスフィールド定義表で `keywords` の検索可能欄が `✓` のみ。`title`, `content`, `combinedContent`, `path`, `tags` には `✓（ja.microsoft）` と明記されているのに対し、`keywords` だけアナライザーの記載がない。検索設計書と実装JSONではともに `ja.microsoft` が適用されている。
- **修正提案**: 要件定義書 7.2 の `keywords` 行の「検索可能」列を `✓（ja.microsoft）` に修正する。

---

### CONS-002: [MEDIUM] インデックス定義の `categoryName` フィールドにアナライザー記載漏れ
- **文書A**: 要件定義書 7.2 インデックス設計（Line 683付近）
  > `categoryName | Edm.String | ✓ | ✓ | — | カテゴリ名`
- **実装**: `scripts/index-definition.json` Line 7
  > `"analyzer": "ja.microsoft"`
- **矛盾**: `categoryName` の検索可能欄が `✓` のみでアナライザー未記載。実装JSONでは `ja.microsoft` が設定されている。他の日本語テキストフィールド（`title`, `content`, `path`, `tags`）にはすべて `✓（ja.microsoft）` と明記されており、`categoryName` だけ表記が欠落している。
- **修正提案**: 要件定義書 7.2 の `categoryName` 行の「検索可能」列を `✓（ja.microsoft）` に修正する。

---

### CONS-003: [LOW] 導入手順書 Step 10 ロール一覧表と CLI コマンドでロール名が不一致
- **文書A**: 導入手順書 Step 10 ロール付与一覧表（Line 861付近）
  > `Cosmos DB Account Reader`
- **文書B**: 導入手順書 Step 10 CLI コマンド（Line 982付近）
  > `--role "Cosmos DB Account Reader Role"`
- **矛盾**: 一覧表では `Cosmos DB Account Reader` だが、CLI コマンドでは `Cosmos DB Account Reader Role` と末尾に `Role` が付いている。Azureの正式なロール名は `Cosmos DB Account Reader Role`（"Role" 付き）である。
- **修正提案**: 一覧表のロール名を `Cosmos DB Account Reader Role` に修正して CLI コマンドと統一する。

---

### CONS-004: [MEDIUM] 要件定義書の bicep 記述と導入手順書の構築方式が矛盾
- **文書A**: 要件定義書 6.3 注記（Line 597付近）
  > `maintenance-bot/infra/azure.bicep` がこの repo で直接作成するのは Managed Identity、App Service Plan、Azure Web App である。
- **文書B**: 導入手順書 Step 7 m365agents.yml カスタマイズ（Line 686付近）
  > `arm/deploy`アクション（Bicepによるリソース作成）を**削除**。Web App・App Service Plan等はStep 1〜6で作成済みのため
- **矛盾**: 要件定義書は bicep がリソースを「直接作成する」と記述しているが、導入手順書ではその bicep デプロイアクションを明示的に削除し、Azure CLI で個別にリソースを作成する手順を採用している。要件定義書の記述は「bicep ファイルに定義されている」という意味であれば誤解を招く表現であり、実際の構築フローとの齟齬がある。
- **修正提案**: 要件定義書 6.3 の注記を「`maintenance-bot/infra/azure.bicep` には Managed Identity、App Service Plan、Azure Web App の定義が含まれるが、本手順では CLI で個別に作成する」等に修正し、bicep が自動実行されるかのような誤解を防ぐ。

---

### CONS-005: [HIGH] 要件定義書の Managed Identity ロール記述が導入手順書のロール一覧と不完全
- **文書A**: 要件定義書 10.1 認証・認可（Line 855-866付近）
  > - Azure OpenAI認証: Managed Identity（Cognitive Services OpenAI Userロール付与）
  > - Cosmos DB書き込み認証: Web AppのManaged Identityに`Cosmos DB Built-in Data Contributor`ロールを付与
- **文書B**: 導入手順書 Step 10 ロール付与一覧（Line 856-876付近）
  > AI Search MI向け:
  > - Azure OpenAI: `Cognitive Services OpenAI User`
  > - Cosmos DB: `Cosmos DB Account Reader Role`（管理プレーン）
  > - Cosmos DB: `Cosmos DB Built-in Data Reader`（データプレーン）
  >
  > Web App MI向け:
  > - AI Search: `Search Index Data Reader`
  > - Cosmos DB: `Cosmos DB Built-in Data Contributor`
  >
  > Graph API:
  > - `Files.ReadWrite.All`
- **矛盾**: 要件定義書のセキュリティ要件セクションでは、AI Search MI に必要な Cosmos DB 関連ロール2つ（`Cosmos DB Account Reader Role` と `Cosmos DB Built-in Data Reader`）、Web App MI に必要な AI Search ロール（`Search Index Data Reader`）、および Graph API 権限（`Files.ReadWrite.All`）が記載されていない。導入手順書には全6種のロール付与が漏れなく記述されている。要件定義書だけを見た読者は必要なロールの全体像を把握できない。
- **修正提案**: 要件定義書 10.1 に以下のロールを追記する。
  - AI Search MI → Cosmos DB: `Cosmos DB Account Reader Role`（管理プレーン）+ `Cosmos DB Built-in Data Reader`（データプレーン）
  - Web App MI → AI Search: `Search Index Data Reader`
  - Web App MI → Microsoft Graph: `Files.ReadWrite.All`（SPO連携用）

---

### CONS-006: [LOW] 要件定義書のリソース構成表に App Service Plan が欠落
- **文書A**: 要件定義書 6.3 リソース構成（Line 587-596付近）
  > リソース一覧: AI Search, Azure OpenAI, Web App, Bot Service, Cosmos DB, Application Insights, Log Analytics Workspace の7項目
- **文書B**: 導入手順書 1.3 構築対象リソース（Line 37-46付近）および 2.4 命名規則（Line 101-113付近）
  > App Service Plan: `asp-maintenance-<env>` が独立したリソースとして命名規則に含まれている
- **矛盾**: App Service Plan は Azure 上で Web App とは独立したリソース（別途課金対象）であり、導入手順書 Step 6 では `az appservice plan create` で明示的に作成している。しかし要件定義書のリソース構成表には Web App のみが記載され、App Service Plan が独立項目として存在しない。
- **修正提案**: 要件定義書 6.3 のリソース構成表に App Service Plan（Basic B1）を追加する。あるいは Web App の備考に「App Service Plan B1 上で稼働」と補足する（現在の「App Service B1（Basic）」は SKU の記載があるため実質的に含まれているとも読めるが、明示が望ましい）。

---

### CONS-007: [LOW] 導入手順書の構築対象リソースに Log Analytics Workspace が欠落
- **文書A**: 要件定義書 6.3 リソース構成（Line 595付近）
  > `Log Analytics Workspace | — | Application Insightsのバックエンド`
- **文書B**: 導入手順書 1.3 構築対象リソース（Line 37-46付近）
  > Log Analytics Workspace が一覧に含まれていない
- **矛盾**: 要件定義書ではリソースとして明記されているが、導入手順書の構築対象一覧に含まれていない。Step 5 の Application Insights 作成時に暗黙的に作成される（またはデフォルトが使われる）が、手順書の構築対象リソース表に記載がないため、読者が全体像を把握しにくい。
- **修正提案**: 導入手順書 1.3 の表に `Log Analytics Workspace | — | Application Insights バックエンド（Application Insights 作成時に自動作成）` を追記する。あるいは Step 5 の補足として明記する。

---

### CONS-008: [MEDIUM] 要件定義書の「Azure OpenAI認証」ロール付与先が不明確
- **文書A**: 要件定義書 10.1 認証・認可（Line 865付近）
  > `Azure OpenAI認証 | Managed Identity（Cognitive Services OpenAI Userロール付与）`
- **文書B**: 導入手順書 Step 10（Line 860付近）
  > `Azure OpenAI | Cognitive Services OpenAI User | **AI Search**のManaged Identity | Skillset（インデクシング時）・Vectorizer（クエリ時）からのEmbedding呼び出し`
- **矛盾**: 要件定義書では「Managed Identity（Cognitive Services OpenAI Userロール付与）」とだけ書かれており、どの Managed Identity（AI Search の MI なのか Web App の MI なのか）に付与するかが明記されていない。導入手順書では明確に「AI Search の Managed Identity」と指定されている。要件定義書の記述だけでは、Web App の MI に付与すべきと誤読する可能性がある。
- **修正提案**: 要件定義書 10.1 を「AI SearchのManaged Identityに`Cognitive Services OpenAI User`ロールを付与」と付与先を明記する。

---

### CONS-009: [HIGH] 検索設計書の `searchFields` に `combinedContent` が含まれないことが要件定義書と暗黙的に矛盾
- **文書A**: 要件定義書 7.2 インデックス設計（Line 686付近）
  > `combinedContent | Edm.String | ✓（ja.microsoft） | — | — | title + content（結合テキスト）。Skillsetのベクトル化入力ソース`
- **文書B**: 検索設計書 9-B / 9-C（Line 393, 417付近）
  > `searchFields: ["title", "content", "keywords"]`
- **矛盾**: 要件定義書で `combinedContent` は「検索可能（searchable）」かつ `ja.microsoft` アナライザーが設定されているが、検索設計書の実際の検索クエリの `searchFields` パラメータには含まれていない。`searchFields` が明示指定されると、BM25 テキスト検索はそこに列挙されたフィールドのみを対象とするため、`combinedContent` は BM25 検索から除外されている。これ自体は設計上の意図的な選択（ベクトル化入力 + Semantic Ranker 用途に限定）だが、要件定義書で「検索可能」と記載しつつ実際のテキスト検索対象外であることが明示されておらず、誤解を招く。
- **修正提案**: 要件定義書 7.2 の `combinedContent` の説明に「BM25テキスト検索の `searchFields` には含めない（ベクトル化入力およびSemantic Rankerのコンテキスト用途）」と補足する。または検索設計書に `combinedContent` を `searchFields` に含めない設計判断の理由を明記する。

---

### CONS-010: [LOW] Embedding コスト記述の備考が微妙に異なる
- **文書A**: 要件定義書 6.4 コスト表（Line 606付近）
  > `Azure OpenAI (S0) | ~¥2,000 | — | Embedding（text-embedding-3-large）のみ。$0.13/1M tokens`
- **文書B**: 導入手順書 付録B コスト表（Line 2156付近）
  > `Azure OpenAI (S0) | ~¥2,000 | — | Embedding利用のみ（初期構築時は Indexer による生成でコスト増加）`
- **矛盾**: 要件定義書にはトークン単価 `$0.13/1M tokens` が記載されているが、導入手順書には記載されていない。逆に、導入手順書には「初期構築時は Indexer による生成でコスト増加」という注記があるが、要件定義書にはない。内容としては矛盾していないが、同じコスト表が2文書に存在する以上、備考レベルの情報も統一されていることが望ましい。
- **修正提案**: 両文書のコスト表備考を統一し、トークン単価と初期コスト注記の両方を含める。例: `Embedding（text-embedding-3-large）のみ。$0.13/1M tokens。初期構築時は全件ベクトル化でコスト増加`

---

## 不整合なし（確認済み項目）

以下の項目は3文書間で整合していることを確認した。

| 確認項目 | 結果 |
|---------|------|
| リソース SKU（AI Search Basic, OpenAI S0, Web App B1, Bot F0, Cosmos Serverless） | 全文書一致 |
| インデックス名 `maintenance-search-index` | 全文書一致 |
| Cosmos DB コンテナ名（scenarios, faqs, impactAssessments） | 全文書一致 |
| パーティションキー（/categoryId, /categoryId, /searchId） | 全文書一致 |
| HNSW パラメータ（m=10, efConstruction=400, efSearch=1000） | 検索設計書 + index-definition.json 一致 |
| VECTOR_WEIGHT = 4.5 | 検索設計書 + 導入手順書 一致 |
| Semantic Ranker 有効（semantic-config） | 全文書一致 |
| Semantic Config フィールド設定（title, content+combinedContent, keywords） | 検索設計書 + index-definition.json 一致 |
| Indexer 実行間隔 PT10M（10分）、最短5分 | 全文書一致 |
| データ件数（シナリオ2,313 + FAQ18,734 = 21,047） | 全文書一致 |
| 月額コスト合計 ~¥20,700/月 | 要件定義書 + 導入手順書 一致 |
| 為替レート前提 1 USD = 155円 | 要件定義書 + 導入手順書 一致 |
| Embedding モデル text-embedding-3-large（3,072次元） | 全文書一致 |
| インデックスフィールド数 15 | 要件定義書 + 検索設計書 + JSON 一致 |
| scenarios Indexer fieldMappings 13フィールド | 検索設計書 + JSON 一致 |
| faqs Indexer fieldMappings 11フィールド | 検索設計書 + JSON 一致 |
| Soft Delete フィールド `isDeleted` | 全文書一致 |
| High Water Mark `_ts` 列 | 全文書一致 |
| Bot 認証方式 Single-Tenant | 全文書一致 |
| メッセージングエンドポイント `/api/messages` | 全文書一致 |
| 命名規則（rg/aoai/srch/cosmos/app/bot-maintenance-\<env\>） | 要件定義書（CLAUDE.md）+ 導入手順書 一致 |
| FR-009 Excel 出力方式（ExcelJS + SPO アップロード） | 要件定義書 + 導入手順書（パッケージ一覧）一致 |
| Adaptive Card サイズ上限 25KB（運用上限）/ 約28KB（Teams上限） | 要件定義書 + 検索設計書 + 導入手順書 一致 |
| searchSessionId キャッシュ TTL 30分 | 要件定義書 + 導入手順書 一致 |

---

## サマリ

| 重要度 | 件数 |
|--------|------|
| HIGH | 2 |
| MEDIUM | 4 |
| LOW | 4 |
| **合計** | **10** |

**HIGH（要対応）:**
- CONS-005: 要件定義書のロール付与一覧が不完全（6種中2種しか記載なし）
- CONS-009: `combinedContent` が searchable だが実際の BM25 searchFields に含まれない点が未説明

**MEDIUM（改善推奨）:**
- CONS-001/002: `keywords` と `categoryName` のアナライザー記載漏れ
- CONS-004: bicep 記述と実際の構築方式の齟齬
- CONS-008: Azure OpenAI ロール付与先の MI が不明確

**LOW（軽微）:**
- CONS-003/006/007/010: ロール名の末尾不統一、リソース表の差異、コスト備考の差異
