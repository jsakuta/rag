# 導入手順書 網羅的レビュー修正 実装計画（v2）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 導入手順書のレビュー指摘32件を修正し、整合性を確保する

**Architecture:** 文書を上から下へ順にセクション単位で修正。Key Vault（旧Step 5）削除に伴うStep番号振り直しは最後に実行し、編集中の参照ずれを防ぐ。全タスクで**現行のStep番号**を使用する。

**Tech Stack:** Markdown 編集のみ（コード変更なし）

**設計書:** `docs/plans/2026-03-11-setup-guide-review-design.md`

---

## 修正順序の方針

1. 各タスクは文書の上から下への順序で編集する
2. Step 5（Key Vault）削除は Task 4 で実施するが、番号振り直しは Task 11 で一括実行
3. **Task 1〜10 は全て現行のStep番号で参照する**（例: 現Step 12 → まだ「Step 12」のまま）
4. Task 11 で全体の Step 番号を一括振り直し
5. Task 12 で CLAUDE.md を修正
6. Task 13 で全体の整合性レビュー

---

### Task 1: 文書ヘッダー・概要セクション修正

**Files:**
- Modify: `docs/導入手順書.md:1-65`

**Step 1: CLI基本方針の明記（#29）**

L19の直後（L20）に以下を挿入:

```markdown
> **実行方針:** 本手順書は Azure CLI での実行を基本とする。一部ステップではポータル手順も参考として併記している。
```

**Step 2: 所要時間の修正（#2, #3）**

L25の所要時間を修正（現行Step番号のまま。Task 11で振り直す）:

- Before: `Step 1〜12（環境構築）: 約2〜3時間、Step 13〜15（Bot実装・テスト）: 別途`
- After: `Step 1〜12（環境構築）: 約3〜4時間（Indexer ベクトル化待ち時間含む）、Step 13〜15（Bot実装・デプロイ・テスト）: 約1〜2時間`

**Step 3: 確認**

修正箇所を目視確認。

---

### Task 2: 前提条件セクション修正

**Files:**
- Modify: `docs/導入手順書.md:87-95`

**Step 1: Azure CLI バージョン更新（#5）**

L91:
- Before: `| Azure CLI | 2.60以上 | リソース作成・ロール付与 |`
- After: `| Azure CLI | 2.64以上 | リソース作成・ロール付与（Cosmos DB データプレーンロール付与に必要） |`

**Step 2: VS Code バージョン更新（#6）**

L93:
- Before: `| Visual Studio Code | 最新版 | 開発エディタ |`
- After: `| Visual Studio Code | 1.90以降 | 開発エディタ（M365 Agents Toolkit v6 対応版） |`

**Step 3: Python 前提条件追加（#27）**

L94の後に行追加:

```markdown
| Python | 3.10以上 | データ変換スクリプト（Step 15 で使用。本環境では 3.13.5 で実行） |
```

**Step 4: rag-local 前提条件追加（#26）**

L76の前提条件テーブル付近に注記追加:

```markdown
> **補足:** Step 15 のテストデータ変換では rag-local プロジェクトの Excel データファイルを参照する。事前にセットアップ済みであること。
```

**Step 5: コミット**

```bash
git add docs/導入手順書.md
git commit -m "docs: update prerequisites and header (review items #2,3,5,6,26,27,29)"
```

---

### Task 3: Step 3（AI Search）修正

**Files:**
- Modify: `docs/導入手順書.md:230-275` 付近

**Step 1: API キー無効化の警告追加（#8）**

L246の注記を以下に置換:

Before:
```
**注記:** `aadOrApiKey` はRBAC認証（Azure AD/Entra ID）とAPIキー認証の両方を許可する設定 [7]。本番環境でAPIキーを完全に無効化する場合は `--disable-local-auth true` を使用する（`--auth-options` とは排他）。
```

After:
```markdown
**注記:** `aadOrApiKey` はRBAC認証（Azure AD/Entra ID）とAPIキー認証の両方を許可する設定 [7]。

> **⚠️ 注意:** 本番環境で API キーを完全に無効化する場合は `--disable-local-auth true` を使用する（`--auth-options` とは排他）。
> ただし、Step 12 の AI Search REST API 操作（インデックス作成等）では管理キーによる認証を使用するため、
> **環境構築が完了するまでは API キー認証を無効化しないこと。**
```

**Step 2: 管理キー記録の注記修正（#9の一部）**

L272:
- Before: `管理キーを記録した（Key Vault格納用）`
- After: `管理キーを記録した（Step 12 の REST API 操作で使用）`

**Step 3: 確認**

修正箇所を目視確認。

---

### Task 4: Key Vault 関連の全削除

**Files:**
- Modify: `docs/導入手順書.md` 複数箇所

Key Vault 削除は文書の**6箇所**に波及する。漏れなく対応する。

**Step 1: 構築対象リソース表から削除（L40）**

L40の行を削除:
```
| 5 | Key Vault | Standard | シークレット管理 |
```
後続の番号を繰り上げ（6→5, 7→6, 8→7）。

**Step 2: Step 5 セクション全体を削除（L374〜L401）**

`## Step 5: Key Vault 作成` の見出しから次のセクション区切り `---` までを削除。

**Step 3: RBAC セクションの Key Vault ロール付与を削除（L878〜L898付近）**

`### Key Vault ロール付与（Azureポータル） [13]` の見出しからコードブロック末尾まで削除。

**Step 4: 確認事項の Key Vault 行を削除（L1069）**

```
- [ ] Key Vault: Web Appに`Key Vault Secrets User`を付与した
```
この行を削除。

**Step 5: 付録A から Key Vault 行を削除（L2196）**

```
| Key Vault名 | |
```
この行を削除。

**Step 6: 付録B から Key Vault 行を削除 + 合計修正（L2214, L2217）**

L2214を削除:
```
| Key Vault (Standard) | ~¥500 | |
```

L2217の合計を修正:
- Before: `| **合計** | **~¥16,000/月** | |`
- After: `| **合計** | **~¥15,500/月** | |`

**Step 7: 確認**

`Key Vault` で文書内検索し、残存参照を確認。
以下は**残すべき参照**（付録Cの本番移行考慮事項で使用するため）:
- L1582 の本番移行時注記（Task 8 で修正）
- 参考文献 [13]（付録Cから参照する）

---

### Task 5: Step 8（Bot プロジェクト）修正

**Files:**
- Modify: `docs/導入手順書.md:490-650` 付近

**Step 1: LLM サービス仮値の明記（#24）**

L504の直後に注記追加:

```markdown
> **補足:** LLM サービスの設定値は仮値で可。任意の文字列（例: `dummy`）を入力する。
> 正式な値は後の手順で環境変数として設定するため、この時点ではエンドポイントのみ正式値を入力すれば十分。
```

**Step 2: AZURE_SUBSCRIPTION_ID の強調（#19）**

L644の直後に警告追加:

```markdown
> **⚠️ 必須:** `AZURE_SUBSCRIPTION_ID` は必ず実際のサブスクリプション ID に置換すること。未設定のままデプロイすると失敗する。
```

**Step 3: generateClientSecret の Tips ブロック化（#30）**

L629の括弧内注記を分離:

Before（L629の一部）:
```
（`generateClientSecret`プロパティは不要、自動で生成される）
```

After: 括弧内記述を削除し、L631の後に Tips ブロックを追加:
```markdown
> **Tips:** `generateClientSecret` プロパティは不要。Toolkit が自動でクライアントシークレットを生成する。
```

**Step 4: 確認**

修正箇所を目視確認。

---

### Task 6: Step 10〜11（RBAC データプレーン・SPO）修正

**Files:**
- Modify: `docs/導入手順書.md:900-1070` 付近

**Step 1: Cosmos DB ロール定義 ID 注釈追加（#15）**

L911:
- Before: `**Built-in Data ContributorのロールID:** `00000000-0000-0000-0000-000000000002``
- After: `**Built-in Data ContributorのロールID:** `00000000-0000-0000-0000-000000000002`（Cosmos DB 組み込みロール定義 ID）`

L913 も同様:
- After: `**Built-in Data ReaderのロールID:** `00000000-0000-0000-0000-000000000001`（Cosmos DB 組み込みロール定義 ID）`

**Step 2: 「13.5節」の参照修正（#10）**

L1246:
- Before: `13.5節で`Cosmos DB Account Reader Role`+`Built-in Data Reader`を付与済み`
- After: `Step 11 で`Cosmos DB Account Reader Role`+`Built-in Data Reader`を付与済み`

**Step 3: SPO 設定順序の注記（#23）**

L1061の直後（SPO Web App反映コマンドの後）に注記追加:

```markdown
> **補足:** SPO 関連の環境変数のみ、このステップで Web App に先行して設定する。
> その他のアプリケーション設定は Step 13 で一括設定する。
```

**Step 4: 確認**

修正箇所を目視確認。

---

### Task 7: Step 12（AI Search 設定）修正

**Files:**
- Modify: `docs/導入手順書.md:1100-1410` 付近
- Read（検証用）: `scripts/index-definition.json`, `scripts/datasource-scenarios.json`, `scripts/datasource-faqs.json`, `scripts/skillset.json`, `scripts/indexer-scenarios.json`, `scripts/indexer-faqs.json`

**Step 1: scripts/ JSON と手書き JSON の差分検証**

各 scripts/*.json ファイルを読み、手順書内の手書き JSON と比較する。
差異がある場合は scripts/ 側を正とし、手順書の注記に反映する。

**Step 2: インデックス定義を scripts/ 参照に書き換え（#25）**

L1102〜（インデックス定義 JSON ブロック全体）を以下に置換:

```markdown
JSON 定義ファイル: `scripts/index-definition.json`

```bash
curl -X PUT \
  "https://srch-maintenance-poc.search.windows.net/indexes/maintenance-search-index?api-version=2025-09-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $ADMIN_KEY" \
  -d @scripts/index-definition.json
```

> **補足:** インデックス定義の詳細は `scripts/index-definition.json` を参照。主要フィールド: `combinedContent`（検索対象テキスト）、`contentVector`（3,072次元ベクトル）、`keywords`（`ja.microsoft`アナライザー適用）。
```

**Step 3: データソース定義を scripts/ 参照に書き換え**

L1190〜（scenarios データソース JSON）と L1219〜（faqs データソース JSON）を以下に置換:

```markdown
JSON 定義ファイル: `scripts/datasource-scenarios.json` / `scripts/datasource-faqs.json`

```bash
# scenarios データソース
curl -X POST \
  "https://srch-maintenance-poc.search.windows.net/datasources?api-version=2025-09-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $ADMIN_KEY" \
  -d @scripts/datasource-scenarios.json

# faqs データソース
curl -X POST \
  "https://srch-maintenance-poc.search.windows.net/datasources?api-version=2025-09-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $ADMIN_KEY" \
  -d @scripts/datasource-faqs.json
```
```

**Step 4: Skillset 定義を scripts/ 参照に書き換え**

L1255〜（Skillset JSON）を以下に置換:

```markdown
JSON 定義ファイル: `scripts/skillset.json`

```bash
curl -X PUT \
  "https://srch-maintenance-poc.search.windows.net/skillsets/maintenance-skillset?api-version=2025-09-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $ADMIN_KEY" \
  -d @scripts/skillset.json
```
```

**Step 5: Indexer 定義を scripts/ 参照に書き換え**

L1298〜（scenarios Indexer JSON）と L1342〜（faqs Indexer JSON）を以下に置換:

```markdown
JSON 定義ファイル: `scripts/indexer-scenarios.json` / `scripts/indexer-faqs.json`

```bash
# scenarios Indexer
curl -X PUT \
  "https://srch-maintenance-poc.search.windows.net/indexers/maintenance-scenarios-indexer?api-version=2025-09-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $ADMIN_KEY" \
  -d @scripts/indexer-scenarios.json

# faqs Indexer
curl -X PUT \
  "https://srch-maintenance-poc.search.windows.net/indexers/maintenance-faqs-indexer?api-version=2025-09-01" \
  -H "Content-Type: application/json" \
  -H "api-key: $ADMIN_KEY" \
  -d @scripts/indexer-faqs.json
```

> **補足:** scenarios Indexer には `path`・`order`・`isFinalAnswer` の fieldMappings があり、
> faqs Indexer には `tags` の fieldMappings がある。これはデータ構造の違いによる意図的な設計。
> 詳細は各 JSON ファイルの `fieldMappings` セクションを参照。
```

**Step 6: Week 概念の削除（#11）**

L1289の以下を削除:
```
- 上記インデックス定義はWeek 1〜2のテキスト検索用。Week 3で画像検索を追加する際は、`imageVector`フィールド（`Collection(Edm.Single)`, 1,024次元, Azure Vision multimodal embedding用）と対応するvectorSearchプロファイルをインデックスに追加する。
```

**Step 7: maxFailedItems 注記追加（#17）**

Indexer定義の参照箇所の後に注記追加:

```markdown
> **注意:** `maxFailedItems: -1`（エラー無制限許容）は PoC 用の設定。
> 本番環境ではデータ欠落を防ぐため、上限値（例: `10`）を設定すること。
```

**Step 8: 同時 Indexer 注記追加（#18）**

L1385付近（2つのIndexerが同一インデックスに書き込む注記の後）に追加:

```markdown
> **補足:** AI Search Basic プランの同時 Indexer 実行数は最大3。本構成では2 Indexer のため問題なし。
```

**Step 9: 確認**

修正箇所を目視確認。既存の注記（データソース、Skillset関連）が適切に残っているか確認。

---

### Task 8: Step 13（Bot デプロイ）修正

**Files:**
- Modify: `docs/導入手順書.md:1500-1640` 付近

**Step 1: 「設定済み」の修正（#12）**

L1516:
- Before: `Web App には Step 6 で取得した値を設定済み`
- After: `Web App には Step 6 で取得した値をこのステップ（Web App アプリケーション設定）で設定する`

**Step 2: ignore ファイルの役割説明（#20）**

L1624（`.deployignore` の直後）に注記追加:

```markdown
> **補足:** `.deployignore` は Toolkit の `azureAppService/zipDeploy` アクションで使用するファイル除外設定。
> 別途 `.webappignore` がある場合は App Service ランタイムのファイル除外設定として機能する。
```

**Step 3: clientSecret の本番移行注記修正（#9の一部）**

L1582を以下に修正:

Before:
```
> **本番環境への移行時:** `clientSecret`（`SECRET_BOT_PASSWORD`）は現在 App Service 環境変数にプレーンテキストで保存されている。本番環境では Key Vault にシークレットを保存し、App Service の環境変数から Key Vault 参照（`@Microsoft.KeyVault(SecretUri=...)`）で読み込む構成に変更すること [10][13]。
```

After:
```markdown
> **本番環境への移行時:** `clientSecret`（`SECRET_BOT_PASSWORD`）は現在 App Service 環境変数にプレーンテキストで保存されている。本番環境でのシークレット管理については付録C「本番移行時の主要考慮事項」を参照 [10][13]。
```

**Step 4: 確認**

修正箇所を目視確認。

---

### Task 9: Step 15（テストデータ）修正

**Files:**
- Modify: `docs/導入手順書.md:1847-1880` 付近

**Step 1: Python 前提の明記（#27）**

L1851:
- Before: `**前提:** Python 3.x + openpyxl がインストール済みであること。`
- After: `**前提:** Python 3.10 以上 + openpyxl がインストール済みであること（2.3節 前提条件参照）。`

**Step 2: 確認**

修正箇所を目視確認。

---

### Task 10: 付録セクション修正

**Files:**
- Modify: `docs/導入手順書.md:2150-2239` 付近

**Step 1: Adaptive Card サイズ + ダングリング参照修正（#21, 参考文献[18]）**

L2165:
- Before: `カードサイズ25KB超過 [18]`
- After: `カードサイズ約28KB超過（Teams の Adaptive Card サイズ上限）`

※ [18] は参考文献に存在しないため削除。

**Step 2: コスト概算の修正（#31, #32）**

L2210:
- Before: `| Azure AI Search (Basic) | ~¥10,000 | Semantic Ranker含む |`
- After: `| Azure AI Search (Basic) | ~¥10,000 | Semantic Ranker は Basic 料金に含まれる（追加料金なし） |`

L2211:
- Before: `| Azure OpenAI (S0) | ~¥2,000 | Embedding利用のみ |`
- After: `| Azure OpenAI (S0) | ~¥2,000 | Embedding利用のみ（初期構築時は Indexer による生成でコスト増加） |`

**Step 3: 本番移行時の考慮事項を追加（#13, #14, #9）**

L2218（付録B の `---` の後）に新セクション追加:

```markdown
## 付録C: 本番移行時の主要考慮事項

本 PoC 環境から本番環境へ移行する際は、以下の項目を検討すること。

| 項目 | PoC 設定 | 本番推奨 |
|------|---------|----------|
| ネットワーク | 全サービスがパブリックアクセス | VNet 統合 + Private Endpoint で全サービスを閉域化 |
| 認証キー | API キー認証有効（RBAC と併用） | `--disable-local-auth true` で API キー無効化。Managed Identity 認証に完全移行 |
| シークレット管理 | Managed Identity で直接認証（Key Vault 不使用） | 外部 API キーや証明書がある場合は Key Vault を使用（下記参照） |
| Indexer エラー許容 | `maxFailedItems: -1`（無制限） | 上限値を設定（例: `10`）してデータ欠落を検知 |
| スケーリング | 各サービス最小プラン | 利用規模に応じてスケールアップ（AI Search: Standard 以上推奨） |

### Key Vault の必要性について

本 PoC では全サービス間認証に **Managed Identity** を使用しており、Key Vault にシークレットを格納する必要がない。
Managed Identity は Azure サービスに自動管理される ID を付与し、Microsoft Entra ID 経由でトークンベースの認証を行う。
これにより、接続文字列やパスワードといった資格情報をコードや設定に保持する必要がなくなる [13]。

参考: [Managed identities for Azure resources](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview)

**Key Vault が必要になるケース:**
- Microsoft Entra 認証をサポートしない外部サービスへの接続（API キー等）
- SSL/TLS 証明書の集中管理
- Bot の `clientSecret` を環境変数から Key Vault 参照（`@Microsoft.KeyVault(SecretUri=...)`）に移行する場合 [10]
```

**Step 4: 確認**

修正箇所を目視確認。

---

### Task 11: Step 番号振り直し + 全体参照修正

**Files:**
- Modify: `docs/導入手順書.md`（全体）

**Step 1: Step 番号の振り直し**

旧 Step 5（Key Vault）削除により、以下の対応で全文書を一括修正する:

| 旧番号 | 新番号 | 内容 |
|--------|--------|------|
| Step 1〜4 | Step 1〜4 | 変更なし |
| Step 5 | 削除 | Key Vault（Task 4 で削除済み） |
| Step 6 | Step 5 | Application Insights |
| Step 7 | Step 6 | Web App |
| Step 8 | Step 7 | Bot プロジェクト作成 + Provision |
| Step 9 | Step 8 | Azure Bot リソース作成 |
| Step 10 | Step 9 | サービスプリンシパル作成 |
| Step 11 | Step 10 | MI ロール付与 + Graph API + SPO |
| Step 12 | Step 11 | AI Search 設定 |
| Step 13 | Step 12 | Bot 実装 + Deploy |
| Step 14 | Step 13 | マニフェスト + Publish |
| Step 15 | Step 14 | 動作確認 + テストデータ |

**Step 2: 全体の相互参照を修正**

文書全体で「Step N」を検索し、すべて新番号に更新する。主な箇所:
- L19: `全15ステップ` → `全14ステップ`
- L25: 所要時間のStep範囲（Task 1で仮記載した値を最終値に修正）
- L47-63: 全体フロー表のStep番号
- 各Step見出し（`## Step N:`）
- 各Step本文内の「Step N で〜」参照
- L1063, L1516 等の相互参照
- トラブルシューティング表内のStep参照
- 付録C内のStep参照（もしあれば）

**Step 3: 構築対象リソース表（L34-43）の番号修正**

Key Vault行はTask 4で削除済み。残りのリソース番号を 1〜7 に振り直し。

**Step 4: 確認**

`Step ` で文書内検索し、番号の整合性を全件確認。旧番号が残っていないか確認。

**Step 5: コミット**

```bash
git add docs/導入手順書.md
git commit -m "docs: comprehensive review fixes for setup guide (32 items)

- Add CLI-first execution policy statement
- Update time estimates (3-4h env + 1-2h bot)
- Update prerequisites (Azure CLI 2.64+, VS Code 1.90+, Python 3.10+)
- Remove Key Vault step (unused in PoC, Managed Identity used instead)
- Replace inline JSON with scripts/*.json file references
- Add API key disable warning for AI Search
- Fix dangling references (13.5節, [18], Week concept)
- Add fieldMappings, maxFailedItems, concurrent indexer notes
- Add Appendix C: production migration considerations with Key Vault explanation
- Renumber steps 5-14 (was 6-15) after Key Vault removal
- Fix cost estimates and Adaptive Card size (28KB)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 12: CLAUDE.md 修正

**Files:**
- Modify: `CLAUDE.md`

**Step 1: 版数記載の修正（#1）**

「導入手順書（v1.4）概要」セクション内の版更新履歴を整理:
- `- v1.1:` / `- v1.3:` / `- v1.4:` の行を削除（版数1.0の文書に対する内部的な反復修正履歴であり、外部に出す版数ではない）

**Step 2: Key Vault 関連の RBAC 記述を修正**

「RBAC設計のポイント」セクション:
- `- Web App の MI → Key Vault: `Key Vault Secrets User`` の行を削除

**Step 3: Step 数の修正**

- `全13ステップ` → `全14ステップ`（CLAUDE.md内の記載を確認して修正）
- `Step 1〜12（環境構築）` → `Step 1〜11（環境構築）`
- `Step 13〜15` → `Step 12〜14`

**Step 4: コミット**

```bash
git add CLAUDE.md
git commit -m "docs: align CLAUDE.md with setup guide review changes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 13: 整合性レビュー

**Files:**
- Read: `docs/導入手順書.md`（全体通読）
- Read: `CLAUDE.md`

**Step 1: セクション番号の連番確認**

Step 1〜14 が連番で欠番・重複がないか確認。`## Step` で検索。

**Step 2: 変数名の統一確認**

`$ADMIN_KEY`, `$DEVELOPER_OID`, `$AZURE_SUBSCRIPTION_ID`, `$WEB_APP_PRINCIPAL_ID` 等が文書全体で統一されているか確認。

**Step 3: 相互参照の整合性確認**

「Step N」参照が全て正しい Step を指しているか再確認。特に:
- トラブルシューティング表内の参照（Step 3, 9, 10, 11, 12, 13, 14 等）
- 各Step本文の「Step N で取得/付与/設定」参照

**Step 4: Key Vault 残存参照の確認**

`Key Vault` で検索。残るべき箇所:
- 付録C（本番移行考慮事項）
- 参考文献 [13]
- L1582付近（付録Cへの誘導文）

それ以外に残っていれば削除。

**Step 5: 参考文献番号の確認**

[1]〜[17] が文書内で正しく参照されているか確認。[18] が残っていないか確認。

**Step 6: 最終コミット（不整合修正がある場合）**

```bash
git add docs/導入手順書.md CLAUDE.md
git commit -m "docs: fix review inconsistencies

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 実行見積もり

| タスク | 所要時間 | 内容 |
|--------|---------|------|
| Task 1-3 | 各3-5分 | ヘッダー・前提条件・AI Search 注記 |
| Task 4 | 10分 | Key Vault 6箇所削除 |
| Task 5-6 | 各3-5分 | Bot プロジェクト・RBAC 修正 |
| Task 7 | 15-20分 | AI Search JSON 参照化（最大のタスク） |
| Task 8-9 | 各3-5分 | Bot デプロイ・テストデータ |
| Task 10 | 10分 | 付録（本番移行考慮事項新設） |
| Task 11 | 15-20分 | Step 番号振り直し + 全参照修正 |
| Task 12 | 5分 | CLAUDE.md |
| Task 13 | 10-15分 | 整合性レビュー |
| **合計** | **約1〜1.5時間** | |
