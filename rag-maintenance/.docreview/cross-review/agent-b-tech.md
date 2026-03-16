# 技術正確性・用語統一レビュー結果

**レビュー担当**: Agent B（技術正確性・用語統一）
**対象文書**: 要件定義書.md / 検索設計書.md / 導入手順書.md
**実施日**: 2026-03-16

---

## 検出された問題

### TECH-001: [LOW] 導入手順書内でのロール名表記揺れ「Cosmos DB Account Reader」vs「Cosmos DB Account Reader Role」
- **文書A**: 導入手順書 Step 10 ロール付与一覧（Line 861付近）
  > `Cosmos DB Account Reader`
- **文書B**: 導入手順書 Step 10 CLIコマンド・確認事項（Line 982, 990, 1134付近）
  > `Cosmos DB Account Reader Role`
- **矛盾**: 同一文書内でロール名の表記が不統一。Azure の正式ロール名は `Cosmos DB Account Reader Role`（CLIの `--role` に渡す値）。Line 861のロール付与一覧テーブルだけ末尾の「Role」が欠落している。
- **修正提案**: 導入手順書 Line 861 のテーブル内を `Cosmos DB Account Reader Role` に統一する。

---

### TECH-002: [MEDIUM] 要件定義書で Azure OpenAI の Cognitive Services OpenAI User ロールの付与先が不明確
- **文書A**: 要件定義書 10.1 認証・認可（Line 865付近）
  > `Azure OpenAI認証 | Managed Identity（Cognitive Services OpenAI Userロール付与）`
- **文書B**: 導入手順書 Step 10 ロール付与一覧（Line 860付近）
  > `Azure OpenAI | Cognitive Services OpenAI User | **AI Search**のManaged Identity`
- **矛盾**: 導入手順書では「AI SearchのManaged Identity」に付与すると明記しているが、要件定義書では単に「Managed Identity」としか記載しておらず、どのリソースのMIかが不明確。読者がWeb AppのMIに付与すると誤解するリスクがある。
- **修正提案**: 要件定義書 Line 865 を `Managed Identity（AI SearchのManaged Identityに Cognitive Services OpenAI User ロール付与）` に変更する。

---

### TECH-003: [MEDIUM] 要件定義書にAI Search関連のロール付与記載が欠落
- **文書A**: 要件定義書 10.1 認証・認可（Line 855-866付近）
  > Bot認証、Azure OpenAI認証、Cosmos DB書き込み認証のみ記載
- **文書B**: 導入手順書 Step 10 ロール付与一覧（Line 856-876付近）
  > 上記に加え、以下も記載:
  > - AI Search: Web Appに `Search Index Data Reader`
  > - Cosmos DB: AI Searchに `Cosmos DB Account Reader Role`
  > - Cosmos DB: AI Searchに `Cosmos DB Built-in Data Reader`
  > - Microsoft Graph: Web Appに `Files.ReadWrite.All`
- **矛盾**: 要件定義書の認証・認可セクションに、Web App → AI Search の検索クエリ実行用ロール（`Search Index Data Reader`）、AI Search → Cosmos DB の Indexer 接続用ロール（`Cosmos DB Account Reader Role` + `Built-in Data Reader`）、Graph API権限（`Files.ReadWrite.All`）の記載がない。導入手順書には全て記載されている。
- **修正提案**: 要件定義書 10.1 に以下のロール付与を追記する:
  - `Search Index Data Reader`（Web App MI → AI Search）
  - `Cosmos DB Account Reader Role`（AI Search MI → Cosmos DB、管理プレーン）
  - `Cosmos DB Built-in Data Reader`（AI Search MI → Cosmos DB、データプレーン）
  - `Files.ReadWrite.All`（Web App MI → Microsoft Graph）

---

### TECH-004: [LOW] 検索設計書のインデックスフィールド数「15フィールド」と要件定義書のフィールド定義の整合性
- **文書A**: 検索設計書 3. アーキテクチャ全体像（Line 77付近）
  > `maintenance-search-index (15フィールド、21,047件)`
- **文書B**: 要件定義書 7.2 インデックス設計（Line 678-694付近）
  > フィールド定義テーブルに15フィールドが列挙されている（id, dataType, categoryId, categoryName, title, content, combinedContent, contentVector, keywords, updatedAt, isDeleted, path, order, isFinalAnswer, tags）
- **矛盾**: なし。数えると15フィールドで一致している。
- **結果**: 整合性に問題なし。

---

### TECH-005: [LOW] rerankerScore の値域表記の軽微な不統一
- **文書A**: 要件定義書 12. 用語定義（Line 936付近）
  > `rerankerScore | Semantic Rankerが付与する関連度スコア（0〜4の範囲）`
- **文書B**: 検索設計書 8-A. 仕組み（Line 337付近）
  > `@search.rerankerScore (0.0〜4.0) で並び替え`
- **矛盾**: Azure公式ドキュメントでは rerankerScore の範囲を「1 to 4.00」としている箇所がある一方、「0〜4」としている記述もある。要件定義書は「0〜4」、検索設計書は「0.0〜4.0」で表記揺れがある。公式ドキュメントの最新記述では、関連性がないと判断されたドキュメントにはスコアが付与されない（結果から除外される）ため、実質的に返されるスコアは0より大きい値になる。
- **修正提案**: 実用上の影響は小さいが、統一するなら両文書とも「0〜4」に揃えるのが簡潔。公式の表記に厳密に合わせるなら「0.0〜4.0」とする。

---

## 問題が見つからなかった観点

### Azure サービス仕様との整合: 整合性に問題なし

以下の項目を Azure 公式ドキュメントと照合し、矛盾がないことを確認した:

- **Indexer 最短スケジュール間隔**: 要件定義書 Line 902, 910 に「最短5分（全SKU共通の制約）」と記載。[公式ドキュメント](https://learn.microsoft.com/en-us/azure/search/search-howto-schedule-indexers)と一致。
- **HNSW m パラメータ上限**: 検索設計書 Line 376, 469 に「m=10（Azure上限）」と記載。[公式ドキュメント](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-create-index)で m の範囲は 4〜10 と確認済み。
- **Semantic Ranker SKU要件**: 要件定義書 Line 589, 892 に「Basic以上」、導入手順書 Line 242 に「Basic SKU以上で利用可能」と記載。[公式ドキュメント](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-enable-disable)と一致。
- **Bot Framework SDK サポート終了**: 要件定義書 Line 583, 導入手順書 Line 1335 に「2025年12月でサポート終了」と記載。[公式ドキュメント](https://learn.microsoft.com/en-us/azure/bot-service/what-is-new?view=azure-bot-service-4.0)で2025年12月31日終了と確認済み。
- **Multi-Tenant非推奨**: 要件定義書 Line 773, 導入手順書 Line 526 に「2025年7月31日以降非推奨」と記載。[公式Q&A](https://learn.microsoft.com/en-us/answers/questions/5555057/)と一致。
- **AI Search REST API バージョン**: 導入手順書で使用している `api-version=2025-09-01` は[公式ドキュメント](https://learn.microsoft.com/en-us/rest/api/searchservice/search-service-api-versions)で最新安定版と確認済み。
- **text-embedding-3-large 次元数**: 3文書とも「3,072次元」で統一。仕様通り。

### SDK・ライブラリ: 整合性に問題なし

- **M365 Agents SDK パッケージ名**: 導入手順書 Line 1333 に `@microsoft/agents-hosting`, `@microsoft/agents-hosting-express` と記載。要件定義書・検索設計書では SDK 名として参照しており、パッケージ名レベルの矛盾なし。
- **ExcelJS**: 要件定義書 Line 513, 822、導入手順書 Line 1355 でいずれも `exceljs` と記載。統一されている。
- **追加パッケージ一覧**: 導入手順書 Line 1346 の `npm install` コマンドに含まれるパッケージ（`@azure/cosmos`, `@azure/identity`, `@azure/search-documents`, `@azure/monitor-opentelemetry`, `@microsoft/microsoft-graph-client`, `exceljs`）と、要件定義書の外部インターフェース要件で言及されているサービス利用が整合している。

### データフロー整合性: 整合性に問題なし

- 3文書とも「Bot → AI Search REST API（直接呼び出し）」「AI Search Skillset/Vectorizer → Azure OpenAI」「Bot → Cosmos DB SDK（直接書き込み）」のデータフローで一致。
- LLMクエリ拡張不採用（原文検索方式）が3文書で一貫している。
- Indexer構成（scenarios用 + faqs用の2 Indexer → 同一インデックス）が3文書で一致。

### フィールド名・変数名: 整合性に問題なし

- **インデックスフィールド名**: 要件定義書 7.2 のフィールド定義、検索設計書 9-B/9-C の `select` 配列、導入手順書 Step 11 の補足説明が全て一致。
- **環境変数名**: 導入手順書 Step 12 の環境変数テーブル（`AI_SEARCH_ENDPOINT`, `AI_SEARCH_INDEX_NAME`, `COSMOS_DB_ENDPOINT`, `COSMOS_DB_DATABASE`, `SPO_DRIVE_ID`, `SPO_UPLOAD_FOLDER`）と Step 14 の `.localConfigs` が一致。
- **Cosmos DB コンテナ名**: 3文書とも `scenarios`, `faqs`, `impactAssessments` で統一。パーティションキーも `/categoryId`, `/categoryId`, `/searchId` で一致。
- **インデックス名**: 3文書とも `maintenance-search-index` で統一。

### 用語統一（上記で指摘した以外）: 整合性に問題なし

- 「コンテナ」で統一されており、「コレクション」との混在なし（要件定義書 Line 952 で「Cosmos DBコンテナ」の用語定義あり）。
- 「Skillset」は英語表記で3文書とも統一。
- 「Indexer」は英語表記で3文書とも統一。
- 「Managed Identity」は3文書とも英語表記で統一。

---

## サマリ

| 重要度 | 件数 |
|--------|------|
| HIGH | 0 |
| MEDIUM | 2 |
| LOW | 2 |

- TECH-001: 導入手順書内のロール名表記揺れ（`Cosmos DB Account Reader` vs `Cosmos DB Account Reader Role`）
- TECH-002: 要件定義書で Cognitive Services OpenAI User の付与先 MI が不明確
- TECH-003: 要件定義書にAI Search / Cosmos DB Indexer用 / Graph API のロール付与記載が欠落
- TECH-005: rerankerScore 値域の軽微な表記揺れ（実用上の影響は小さい）

Azure サービス仕様との整合、SDK・ライブラリ、データフロー、フィールド名・変数名については3文書間で整合性に問題なし。

Sources:
- [Schedule indexer execution](https://learn.microsoft.com/en-us/azure/search/search-howto-schedule-indexers)
- [Create a Vector Index](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-create-index)
- [Semantic Ranking Overview](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview)
- [Enable or disable semantic ranker](https://learn.microsoft.com/en-us/azure/search/semantic-how-to-enable-disable)
- [Set up an Indexer Connection to Azure Cosmos DB Using a Managed Identity](https://learn.microsoft.com/en-us/azure/search/search-howto-managed-identities-cosmos-db)
- [API Versions of Search Service REST APIs](https://learn.microsoft.com/en-us/rest/api/searchservice/search-service-api-versions)
- [Bot Framework SDK - What's new](https://learn.microsoft.com/en-us/azure/bot-service/what-is-new?view=azure-bot-service-4.0)
