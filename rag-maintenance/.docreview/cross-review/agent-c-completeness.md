# 完全性・参照整合性レビュー結果

**レビュー日**: 2026-03-16
**レビュー担当**: Agent C（完全性・参照整合性）
**対象文書**:
- 要件定義書.md
- 検索設計書.md
- 導入手順書.md

---

## 1. 機能カバレッジ（FR-001〜FR-009 vs 導入手順書）

全機能要件について、導入手順書で環境構築に必要なリソース・設定がカバーされているかを確認した。

| 機能ID | 機能名 | 導入手順書での対応 | 判定 |
|--------|--------|-------------------|------|
| FR-001 | 改定内容テキスト入力 | Step 7-8（Bot登録）, Step 12（Bot実装） | OK |
| FR-002 | 検索モード選択 | Step 12（Bot実装） | OK |
| FR-003 | 影響候補検索（意味検索） | Step 3（AI Search）, Step 11（インデックス・Skillset）, Step 10（ロール付与） | OK |
| FR-004 | 影響候補検索（キーワード検索） | Step 3（AI Search）, Step 11 | OK |
| FR-005 | 候補一覧表示 | Step 12（Bot実装） | OK |
| FR-006 | データマスタ同期 | Step 11（Indexer設定、10分間隔） | OK |
| FR-007 | FAQ一括削除 | Step 10（Cosmos DB Data Contributor）, Step 12 | OK |
| FR-008 | シナリオ要修正フラグ記録 | Step 4（impactAssessmentsコンテナ）, Step 10（Cosmos DB Data Contributor）, Step 12 | OK |
| FR-009 | シナリオ要修正Excel出力 | Step 10（Graph API Files.ReadWrite.All + SPOドライブID取得）, Step 12（ExcelJS, SPO環境変数） | OK |

**結論**: 全機能要件が導入手順書でカバーされている。整合性に問題なし。

---

## 2. 参照整合性

### COMPL-001: LOW 導入手順書の脚注参照 [15] がStep 2で初出だがコンテキスト不足
- **文書A**: 導入手順書 Step 2（Line 190付近）
  > `--deployment-name` ... AI SearchのSkillset/Vectorizer設定で使用 [15]
- **問題**: 脚注 [15] は参考文献セクション（Line 2211）の「AI Search Integrated Vectorization」を指しており、リンク先自体は正しい。しかし [15] の初出がStep 2であり、読者がこの時点でSkillset/Vectorizerとの関連を理解しにくい。実質的な問題ではない。
- **修正提案**: 修正不要（情報は正しく、参考文献リストで番号とURLの対応が取れている）。

全脚注参照 [1]〜[17] を確認した結果、参考文献セクション（Line 2196〜2213）の番号と一致しており、参照整合性に問題なし。

---

## 3. セクション番号

### 要件定義書
目次: 1〜13。本文: 1〜13（付録含む）。連番に問題なし。

### 検索設計書
目次: 1〜10。本文: 1〜10。連番に問題なし。

### 導入手順書
目次: 1（概要）, 2（前提条件）, Step 1〜14, 付録A〜C。連番に問題なし。

**結論**: 全文書のセクション番号は正しい連番になっている。整合性に問題なし。

---

## 4. 図表参照

### 要件定義書の図パス

| 行 | パス | 実在 |
|----|------|------|
| 78 | `drawings/system-overview.png` | OK |
| 97 | `drawings/as-is-flow.png` | OK |
| 101 | `drawings/to-be-flow.png` | OK |
| 178 | `drawings/search-flow.png` | OK |
| 568 | `drawings/azure-architecture.png` | OK |

### 検索設計書の図パス

| 行 | パス | 実在 |
|----|------|------|
| 57 | `drawings/search-architecture.png` | OK |

### 導入手順書のスクリーンショットパス

全21件のスクリーンショット参照（`screenshots/annotated/` および `screenshots/` 配下）を確認した結果、全て実在するファイルと一致している。

**結論**: 全図表パスが実在ファイルを正しく参照している。整合性に問題なし。

---

## 5. 前提条件の整合

### COMPL-002: MEDIUM 要件定義書の前提条件「既存リソースは別手順で準備済み」が導入手順書の構成と矛盾
- **文書A**: 要件定義書 11.3節（Line 921）
  > Azure AI Search、Azure OpenAI、Cosmos DB、Application Insights、Azure Bot Service は既存環境または別手順で準備済みであること
- **文書B**: 導入手順書 1.3節（Line 37〜46）
  > Step 1〜8 でこれらのリソースを新規作成する手順を記載
- **問題**: 要件定義書では「既存環境または別手順で準備済み」を前提条件としているが、導入手順書自体がその「別手順」に該当する。文言の解釈次第では矛盾ではないが、要件定義書 6.3節（Line 597）の注記「`azure.bicep` が直接作成するのは Managed Identity、App Service Plan、Azure Web App」と合わせると、元々はBicepで一部のみ作成し残りは別手順という想定だったことがわかる。導入手順書が全リソースの作成手順を網羅した結果、この前提条件の記述が実態とずれている。
- **修正提案**: 要件定義書 11.3節の該当前提条件を「Azure AI Search、Azure OpenAI、Cosmos DB、Application Insights、Azure Bot Service は導入手順書の手順に従い構築済みであること」に修正する。または「本書の対象スコープ外であり、導入手順書で構築する」と注記を追加する。

### 前提条件の対応確認

| 要件定義書 11.3節 前提条件 | 導入手順書での対応 | 判定 |
|--------------------------|-------------------|------|
| Cosmos DB（シナリオ+FAQデータ）が利用可能 | Step 4（Cosmos DB作成）+ Step 14（テストデータ投入） | OK |
| Azureサブスクリプションが利用可能 | 2.1節 | OK |
| Azure OpenAI利用申請が承認済み | 2.1節 | OK |
| 担当者がTeamsにアクセス可能 | 2.2節 | OK |
| 検索結果キャッシュの前提（メモリ保持） | 導入手順書の対象外（実装詳細） | OK（文書役割の違い） |
| 各Azureリソースが準備済み | Step 1〜8で作成 | COMPL-002参照 |

---

## 6. 欠落情報

### COMPL-003: LOW 要件定義書の「Search Index Data Contributor」ロールが導入手順書で未言及（FAQ即時削除用）
- **文書A**: 要件定義書 FR-007（Line 368）
  > 即時除外（任意）: `Documents - Index` APIで即座に該当ドキュメントを削除
- **文書B**: 導入手順書 Step 10（Line 868）
  > `Search Index Data Reader` のみ付与
- **問題**: 要件定義書 FR-007ではAI Searchのドキュメント操作API（即時除外）を「任意実装」としている。この機能を実装する場合、Web Appには `Search Index Data Reader` ではなく `Search Index Data Contributor`（書き込み権限含む）が必要になる。ただし要件定義書で「任意」「Indexerによる自動反映を基本」と明記されているため、現時点では問題にならない。
- **修正提案**: 導入手順書 Step 10 の AI Search ロール付与セクションに補足を追加する。「将来的にFAQ削除の即時インデックス反映（FR-007の任意機能）を実装する場合は、`Search Index Data Contributor` への変更が必要」と注記。

### COMPL-004: MEDIUM 要件定義書のNFR-001タイムアウト対策（proactiveActivity）が導入手順書で完全に未言及
- **文書A**: 要件定義書 5.1節 NFR-001（Line 533）
  > Teams Action.Executeのタイムアウトは10〜15秒。超過時は「処理中」カードを即時返却し、バックグラウンド処理後にproactiveActivityで結果通知する構成を採用
- **文書B**: 導入手順書 全体
- **問題**: proactiveActivity（プロアクティブメッセージング）を使用するには、BotがConversation ReferenceをCosmos DBやメモリに保存する必要があり、Teamsでのプロアクティブメッセージ送信にはService URLの信頼設定が必要になる場合がある。導入手順書はこの構成に関する環境設定を含んでいない。ただし、これは実装詳細であり「別途作成する実装ガイド」の範囲と解釈できる。
- **修正提案**: 修正不要（実装ガイドの範囲）。ただし、proactiveActivityに追加の環境設定（Cosmos DBへのConversation Reference保存用コンテナ等）が必要になる場合は、導入手順書のStep 4にコンテナ追加を検討する。

要件定義書で定義された全技術要素を確認した結果、導入手順書の環境構築範囲では主要な欠落はない。

---

## 7. 環境変数の網羅性

導入手順書 Step 12（Line 1478〜1500）の環境変数一覧と、要件定義書の機能要件から必要な環境変数を照合した。

| 要件定義書の機能 | 必要な環境変数 | 導入手順書での設定 | 判定 |
|----------------|--------------|-------------------|------|
| Bot認証（FR-001） | clientId, clientSecret, tenantId, MicrosoftApp* | Step 12 設定一覧 | OK |
| AI Search接続（FR-003/004） | AI_SEARCH_ENDPOINT, AI_SEARCH_INDEX_NAME | Step 12 設定一覧 | OK |
| Cosmos DB接続（FR-007/008） | COSMOS_DB_ENDPOINT, COSMOS_DB_DATABASE | Step 12 設定一覧 | OK |
| SPOアップロード（FR-009） | SPO_DRIVE_ID, SPO_UPLOAD_FOLDER | Step 12 設定一覧 | OK |
| 監視（NFR監視要件） | APPLICATIONINSIGHTS_CONNECTION_STRING | Step 12 設定一覧 | OK |

**結論**: 要件定義書の機能を実現するために必要な環境変数は全て導入手順書でカバーされている。整合性に問題なし。

---

## 総括

| 観点 | 結果 | 指摘数 |
|------|------|--------|
| 機能カバレッジ | 整合性に問題なし | 0 |
| 参照整合性 | 整合性に問題なし | 0（COMPL-001はLOW/情報提供） |
| セクション番号 | 整合性に問題なし | 0 |
| 図表参照 | 整合性に問題なし | 0 |
| 前提条件の整合 | MEDIUM 1件 | 1（COMPL-002） |
| 欠落情報 | LOW 1件 + MEDIUM 1件 | 2（COMPL-003, COMPL-004） |
| 環境変数の網羅性 | 整合性に問題なし | 0 |

**HIGH指摘: 0件 / MEDIUM指摘: 2件 / LOW指摘: 2件**

3文書間の完全性・参照整合性は高い水準で維持されている。MEDIUM指摘はいずれも運用上の実害は小さいが、文書としての正確性向上のため修正を推奨する。
