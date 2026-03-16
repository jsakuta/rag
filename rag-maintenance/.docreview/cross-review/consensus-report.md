# 3文書クロスドキュメントレビュー — コンセンサスレポート

**日付**: 2026-03-16
**レビュー対象**: 要件定義書.md / 検索設計書.md / 導入手順書.md
**レビュー体制**: 4エージェント独立レビュー + コンセンサス投票 + 主担当再検証

## レビュー体制

| エージェント | 担当観点 | 指摘数 |
|-------------|---------|--------|
| 総合レビュー | 全観点横断 | HIGH 2 / MED 4 / LOW 4 |
| Agent-A | 数値・パラメータ・コスト照合 | LOW 1 |
| Agent-B | 技術正確性・用語統一・Azure仕様 | MED 2 / LOW 2 |
| Agent-C | 完全性・参照整合性・機能カバレッジ | MED 2 / LOW 2 |

---

## コンセンサス結果

### 🔴 必須修正（3/4以上一致）

#### Issue-1: 要件定義書 10.1 のロール付与一覧が不完全
- **検出**: 総合(CONS-005 HIGH) + Agent-B(TECH-003 MED) = 2/4一致 + 主担当検証で確認
- **信頼度**: 🔴最高（内容が明確な事実誤認）
- **場所**: 要件定義書 10.1 認証・認可（Line 855〜866）
- **問題**: 導入手順書には6種のロール付与が記載されているが、要件定義書には2種（`Cognitive Services OpenAI User` + `Cosmos DB Built-in Data Contributor`）しか記載されていない。以下4種が欠落:
  - Web App MI → AI Search: `Search Index Data Reader`
  - AI Search MI → Cosmos DB: `Cosmos DB Account Reader Role`（管理プレーン）
  - AI Search MI → Cosmos DB: `Cosmos DB Built-in Data Reader`（データプレーン）
  - Web App MI → Microsoft Graph: `Files.ReadWrite.All`
- **修正案**: 要件定義書 10.1 にロール付与一覧表を追加し、導入手順書 Step 10 と同等の情報を記載

---

### 🔴 推奨修正（2/4一致）

#### Issue-2: 要件定義書で Azure OpenAI ロール付与先の MI が不明確
- **検出**: 総合(CONS-008 MED) + Agent-B(TECH-002 MED) = 2/4一致
- **信頼度**: 🔴高
- **場所**: 要件定義書 10.1（Line 865）
- **問題**: 「Managed Identity（Cognitive Services OpenAI Userロール付与）」とだけ記載。**AI SearchのMI**に付与するのか**Web AppのMI**に付与するのかが不明確。導入手順書では「AI SearchのManaged Identity」と明記。
- **修正案**: `AI SearchのManaged Identityに Cognitive Services OpenAI User ロールを付与` と付与先を明記

#### Issue-3: 要件定義書の bicep 記述と導入手順書の構築方式の齟齬
- **検出**: 総合(CONS-004 MED) + Agent-C(COMPL-002 MED) = 2/4一致
- **信頼度**: 🔴高
- **場所**: 要件定義書 6.3 注記（Line 597）/ 11.3 前提条件（Line 921）
- **問題**:
  - 6.3節: `azure.bicep がこの repo で直接作成するのは Managed Identity、App Service Plan、Azure Web App`（bicepがリソースを作成する前提の文言）
  - 11.3節: `Azure AI Search、Azure OpenAI、Cosmos DB 等は既存環境または別手順で準備済みであること`
  - しかし導入手順書では bicep アクションを**削除**し、全リソースを Azure CLI で個別作成。
- **修正案**:
  - 6.3節: `azure.bicep には定義が含まれるが、導入手順書では CLI で個別に作成する` に修正
  - 11.3節: `導入手順書の手順に従い構築済みであること` に修正

#### Issue-4: 要件定義書 7.2 `keywords` フィールドのアナライザー記載漏れ
- **検出**: 総合(CONS-001 MED) のみ1/4 → **主担当再検証で確認**
- **信頼度**: 🔴高（実装JSONと直接照合して事実確認済み）
- **場所**: 要件定義書 7.2 インデックス設計（Line 688）
- **問題**: `keywords` の検索可能列が `✓` のみ。`title`, `content`, `combinedContent`, `path`, `tags` は全て `✓（ja.microsoft）` と明記。実装JSON（`scripts/index-definition.json` Line 18）では `"analyzer": "ja.microsoft"` が設定されている。
- **修正案**: `✓` → `✓（ja.microsoft）` に修正

#### Issue-5: 要件定義書 7.2 `categoryName` フィールドのアナライザー記載漏れ
- **検出**: 総合(CONS-002 MED) のみ1/4 → **主担当再検証で確認**
- **信頼度**: 🔴高（実装JSONと直接照合して事実確認済み）
- **場所**: 要件定義書 7.2 インデックス設計（Line 683）
- **問題**: `categoryName` の検索可能列が `✓` のみ。実装JSON（`scripts/index-definition.json` Line 7）では `"analyzer": "ja.microsoft"` が設定されている。
- **修正案**: `✓` → `✓（ja.microsoft）` に修正

---

### 🟡 検討修正（2/4一致・LOW）

#### Issue-6: 導入手順書 Step 10 ロール名表記揺れ
- **検出**: 総合(CONS-003 LOW) + Agent-B(TECH-001 LOW) = 2/4一致
- **場所**: 導入手順書 Step 10 ロール付与一覧（Line 861）
- **問題**: テーブル内 `Cosmos DB Account Reader` ← 末尾の `Role` が欠落。CLIコマンド（Line 982）は正しく `Cosmos DB Account Reader Role`
- **修正案**: テーブル内を `Cosmos DB Account Reader Role` に統一

---

### 🟡 ユーザー判断

#### Issue-7: `combinedContent` が searchFields に含まれない設計判断の説明不足
- **検出**: 総合(CONS-009 HIGH) のみ1/4 → **主担当再検証**
- **検証結果**: 要件定義書 7.2 で `combinedContent` は `✓（ja.microsoft）` = searchable と記載。しかし検索設計書 9-B の `searchFields: ["title", "content", "keywords"]` には含まれない。これは **設計上の意図的な選択**（ベクトル化入力 + Semantic Ranker用途に限定）だが、要件定義書の表だけ見ると BM25 検索対象と誤解しうる。
- **信頼度**: 🟡中（矛盾ではなく説明不足。設計意図は正しい）
- **修正案**: 要件定義書 7.2 の `combinedContent` の説明に「BM25 searchFieldsには含めない（ベクトル化入力およびSemantic Ranker用途）」と補足。**または**検索設計書側に設計判断の理由を注記。

#### Issue-8: NFR-001 proactiveActivity 構成が導入手順書で未言及
- **検出**: Agent-C(COMPL-004 MED) のみ1/4 → **主担当再検証**
- **検証結果**: 要件定義書 NFR-001 で「超過時は処理中カードを返却しバックグラウンド処理後にproactiveActivityで結果通知」と記載。導入手順書に関連する環境設定（Conversation Reference保存等）の記載がない。ただし導入手順書は「環境構築」が主目的で、実装詳細は「別途作成する実装ガイド」の範囲。
- **信頼度**: 🟡中（導入手順書の範囲外と解釈可能）
- **修正案**: 修正不要。ただし将来的にproactiveActivity実装時、追加のCosmos DBコンテナが必要になる可能性あり。

---

### ⚪ 参考情報（修正不要）

| # | 問題 | 検出元 | 備考 |
|---|------|--------|------|
| 9 | コスト表備考テキストの微差 | 総合 + Agent-A | 金額値は一致。文書の役割差として許容 |
| 10 | rerankerScore値域「0〜4」vs「0.0〜4.0」 | Agent-B | 実用上の影響なし |
| 11 | リソース表のApp Service Plan / Log Analytics欠落 | 総合 | 実質的にカバーされており軽微 |
| 12 | FR-007即時削除用ロールの注記不足 | Agent-C | 「任意実装」のため現時点で問題なし |

---

## 問題なし確認済み項目

以下は4エージェント全てが整合性を確認した主要項目:

- リソース SKU（7リソース全一致）
- データ件数（合計 + カテゴリ別7区分全一致）
- 検索パラメータ（HNSW m=10, efSearch=1000, efConstruction=400, VECTOR_WEIGHT=4.5）
- コスト表金額（合計 ~¥20,700/月、為替 155円/USD）
- インデックスフィールド数（15フィールド全一致）
- 命名規則・環境変数名・フィールド名
- Azure サービス仕様との整合（Indexer最短5分、HNSW m上限10、Semantic Ranker Basic以上等）
- SDK選定（M365 Agents SDK）、Bot Framework SDK終了時期
- Multi-Tenant非推奨時期
- 全図表パス実在（drawings 6件 + screenshots 21件）
- 全脚注参照 [1]〜[17] の整合性
- セクション番号の連番
- FR-001〜FR-009の機能カバレッジ
- 環境変数の網羅性

---

## サマリ

| 信頼度 | 件数 | アクション |
|--------|------|-----------|
| 🔴 必須修正 | **1件** | Issue-1: ロール付与一覧の欠落 |
| 🔴 推奨修正 | **4件** | Issue-2〜5: ロール付与先不明確 / bicep齟齬 / アナライザー記載漏れ×2 |
| 🟡 検討修正 | **1件** | Issue-6: ロール名表記揺れ |
| 🟡 ユーザー判断 | **2件** | Issue-7〜8: combinedContent説明不足 / proactiveActivity未言及 |
| ⚪ 参考情報 | **4件** | 修正不要 |
| **合計** | **12件** | — |

**総評**: 3文書間の整合性は**高い水準**。数値・パラメータ・データ構成・コストは完全一致。指摘の大部分は要件定義書のセキュリティ要件セクション（10.1節）の記述粒度が導入手順書と比べて粗い点に集中しており、修正範囲は限定的。
