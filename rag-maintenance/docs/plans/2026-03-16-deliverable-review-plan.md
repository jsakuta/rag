# 要件定義書 最終レビュー修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** レビューで検出された12項目の修正 + テキスト版（参考）5箇所の削除を要件定義書に適用する

**Architecture:** 単一ファイル `docs/要件定義書.md` への Edit ツールによる逐次修正。修正は「後方→前方」の順に実施して行番号のズレを最小化する。全修正を1コミットにまとめる

**Tech Stack:** Edit ツール（テキスト置換）のみ

---

## 修正対象サマリ

| Task | カテゴリ | 修正箇所 | 重要度 |
|------|---------|---------|--------|
| 1 | 削除 | テキスト版（参考）5箇所削除（後方から） | ユーザー要望 |
| 2 | 削除 | 12章「スクロールコンテナ」用語削除 | 低 |
| 3 | コスト | Web App B1: ~¥2,000→~¥8,000(Windows) | 高 |
| 4 | コスト | AI Search備考: Semantic Ranker無料枠の明記 | 中 |
| 5 | コスト | 合計行修正: ~¥15,500→~¥22,500 | 高 |
| 6 | 仕様 | FR-009 Excel列定義: 回答行削除+注記 | 中 |
| 7 | 仕様 | 7.2 combinedContent 用途説明追記 | 中 |
| 8 | 仕様 | 7.3 データフロー図 Indexer A/B 2本構成 | 中 |
| 9 | 表現 | FR-006 Indexer間隔「Basic SKUの」削除 | 低 |
| 10 | 表現 | FR-009 ファイル名例 cat-yokin→yokin | 低 |
| 11 | 構造 | 2.3節を2.4に統合 + セクション番号繰り上げ | 低 |
| 12 | 表現 | FR-005 ページネーション記述の簡素化 | 低 |
| 13 | 表現 | 6.3リソース構成表にKey Vault/Log Analytics追記 | 低 |
| 14 | - | コミット | - |

---

### Task 1: テキスト版（参考）5箇所を後方から削除

**Files:**
- Modify: `docs/要件定義書.md`

テキスト版（参考）の `<details>...</details>` ブロックは以下の5箇所（行番号は修正前時点）。
**行番号のズレ防止のため後方から削除する。** 各ブロックの直前の空行1つは残す。

**Step 1: 6.1 アーキテクチャ図テキスト版を削除（行712-766）**

直前: `![Azure構成図](drawings/azure-architecture.png)` (行711)
削除対象: 行712の空行 + `<details>` 〜 `</details>` + 直後の空行（行712〜766、55行）

```
old:
![Azure構成図](drawings/azure-architecture.png)

<details>
<summary>テキスト版（参考）</summary>
...（中略: テキスト版Azure構成図）...
</details>

### 6.2 Azure Bot Service と Azure Web App の役割分担

new:
![Azure構成図](drawings/azure-architecture.png)

### 6.2 Azure Bot Service と Azure Web App の役割分担
```

**Step 2: FR-003 検索方式フローテキスト版を削除（行286-321）**

直前: `![原文検索方式フロー](drawings/search-flow.png)` (行285)
削除対象: 行286の空行 + `<details>` 〜 `</details>`（行286〜321、36行）

```
old:
![原文検索方式フロー](drawings/search-flow.png)

<details>
<summary>テキスト版（参考）</summary>
...（中略: テキスト版検索フロー）...
</details>

**構成上のポイント:**

new:
![原文検索方式フロー](drawings/search-flow.png)

**構成上のポイント:**
```

**Step 3: 3.2 To-Beフローテキスト版を削除（行166-208）**

直前: `![目標業務フロー](drawings/to-be-flow.png)` (行165)
削除対象: 行166の空行 + `<details>` 〜 `</details>`（行166〜208、43行）

```
old:
![目標業務フロー](drawings/to-be-flow.png)

<details>
<summary>テキスト版（参考）</summary>
...（中略: テキスト版To-Beフロー）...
</details>

### 3.3 データに対するアクション方針

new:
![目標業務フロー](drawings/to-be-flow.png)

### 3.3 データに対するアクション方針
```

**Step 4: 3.1 As-Isフローテキスト版を削除（行129-161）**

直前: `![現行業務フロー](drawings/as-is-flow.png)` (行128)
削除対象: 行129の空行 + `<details>` 〜 `</details>`（行129〜161、33行）

```
old:
![現行業務フロー](drawings/as-is-flow.png)

<details>
<summary>テキスト版（参考）</summary>
...（中略: テキスト版As-Isフロー）...
</details>

### 3.2 目標業務フロー（To-Be）

new:
![現行業務フロー](drawings/as-is-flow.png)

### 3.2 目標業務フロー（To-Be）
```

**Step 5: 2.5 システム概念図テキスト版を削除（行81-109）**

直前: `![システム概念図](drawings/system-overview.png)` (行80)
削除対象: 行81の空行 + `<details>` 〜 `</details>`（行81〜109、29行）

```
old:
![システム概念図](drawings/system-overview.png)

<details>
<summary>テキスト版（参考）</summary>
...（中略: テキスト版概念図）...
</details>

### 2.6 ローカル検証からの設計判断

new:
![システム概念図](drawings/system-overview.png)

### 2.6 ローカル検証からの設計判断
```

**Step 6: 削除行数の確認**

削除行数合計: 55 + 36 + 43 + 33 + 29 = **196行削減**。
以降のTaskは全てEditツールの文字列マッチで特定するため、行番号のズレは問題ない。

---

### Task 2: 未使用用語「スクロールコンテナ」削除

**Files:**
- Modify: `docs/要件定義書.md` — セクション12 用語定義

**Step 1: スクロールコンテナの行を削除**

```
old:
| Sequential Workflow | Action.Executeのレスポンスでカードを更新することで、確認→実行→完了のフローを実現するパターン |
| スクロールコンテナ | Adaptive CardのContainerに`maxHeight`プロパティを設定し、内容超過時に垂直スクロールバーを表示する機能。v1.5以降で利用可能 |
| impactAssessments | シナリオの要修正フラグを記録するCosmos DBコンテナ。判定保存単位ID（`searchId`）・シナリオID・検索クエリ・`rerankerScore`・判定者・日時を保持 |

new:
| Sequential Workflow | Action.Executeのレスポンスでカードを更新することで、確認→実行→完了のフローを実現するパターン |
| impactAssessments | シナリオの要修正フラグを記録するCosmos DBコンテナ。判定保存単位ID（`searchId`）・シナリオID・検索クエリ・`rerankerScore`・判定者・日時を保持 |
```

---

### Task 3: Web App B1 コスト修正（高重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション6.4

**Step 1: Web App行の修正**

bicep確認済み: `kind: 'app'`（Windowsデプロイ）+ `WEBSITE_NODE_DEFAULT_VERSION: '~22'`

```
old: | Azure Web App (B1) | ~¥2,000 | App Service Basic B1（1コア, 1.75GB RAM） |
new: | Azure Web App (B1) | ~¥8,000 | App Service Basic B1 Windows（1コア, 1.75GB RAM） |
```

---

### Task 4: AI Search 備考修正（中重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション6.4

**Step 1: AI Search行の備考修正**

```
old: | Azure AI Search (Basic) | ~¥10,000 | Semantic Ranker含む |
new: | Azure AI Search (Basic) | ~¥11,000 | Semantic Ranker: 無料枠1,000リクエスト/月。超過時は従量課金 |
```

---

### Task 5: 合計行修正（高重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション6.4

**Step 1: 合計を再計算して修正**

内訳: AI Search ¥11,000 + OpenAI ¥2,000 + Web App ¥8,000 + Cosmos ¥1,000 + AppInsights ¥500 + Bot ¥0 = **¥22,500**

```
old: | **合計** | **~¥15,500/月** | — |
new: | **合計** | **~¥22,500/月** | — |
```

---

### Task 6: FR-009 Excel列定義修正（中重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション4.2 FR-009

**Step 1: 列定義テーブルから「回答」「文字数」行を削除し注記追加**

実装確認済み: `excel.ts:70` "回答もLvカラムに統合"、`excel.ts:102` `const levels = [...pathLevels, item.content]`

```
old:
| LvN | 階層レベルNのテキスト | — |
| 文字数 | LvNの文字数 | — |
| 回答 | 回答テキスト | — |
| 文字数 | 回答の文字数 | — |

要修正行は行全体を黄色ハイライト。

new:
| LvN | 階層レベルNのテキスト | — |
| 文字数 | LvNの文字数 | — |

**注記:** 回答テキスト（content）は最終階層レベル（LvN）のカラムに統合される。独立した「回答」列は設けない。

要修正行は行全体を黄色ハイライト。
```

---

### Task 7: combinedContent フィールド用途説明追記（中重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション7.2

**Step 1: combinedContent行の説明列を修正**

根拠: `skillset.json:15` — `"source": "/document/combinedContent"`

```
old: | combinedContent | Edm.String | ✓（ja.microsoft） | — | — | title + content（結合テキスト） |
new: | combinedContent | Edm.String | ✓（ja.microsoft） | — | — | title + content（結合テキスト）。Skillsetのベクトル化入力ソース（このフィールドからcontentVectorを生成） |
```

---

### Task 8: データフロー図 Indexer A/B 2本構成に修正（中重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション7.3

**Step 1: インデクシングのフロー図を修正**

```
old:
**インデクシング（自動・定期）:**

```
[Indexer A: テキストデータ]
Cosmos DB → Indexer (10分ごと)
  → AzureOpenAIEmbeddingSkill（text-embedding-3-large でベクトル生成）
  → contentVector フィールドに格納
```

new:
**インデクシング（自動・定期）:**

```
[Indexer A: scenarios用 / Indexer B: faqs用]
Cosmos DB (scenarios / faqs) → 各Indexer (10分ごと)
  → AzureOpenAIEmbeddingSkill（text-embedding-3-large でベクトル生成）
  → 同一インデックス maintenance-search-index の contentVector フィールドに格納
```
```

---

### Task 9: Indexer間隔の記述統一（低重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション4.2 FR-006

**Step 1: 「Basic SKUの」を削除**

11.2節（行1097）では「全SKU共通の制約」と記載。Azure公式ドキュメントと一致するのは「全SKU共通」。

```
old: - Indexer実行間隔は10分（`PT10M`、Azure AI Search Basic SKUの最短間隔は5分）
new: - Indexer実行間隔は10分（`PT10M`、Azure AI Searchの最短間隔は5分）
```

---

### Task 10: ファイル名例の修正（低重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション4.2 FR-009

実装確認済み: `excel.ts:169` — `scenario_${categoryId}_${safeName}_${ts}.xlsx`（categoryIdに`cat-`プレフィックスなし）

**Step 1: FR-009処理フロー内のファイル名例を修正**

```
old:
   │ ・scenario_cat-yokin_預金_...xlsx     │
   │ ・scenario_cat-kawase_為替_...xlsx    │
new:
   │ ・scenario_yokin_預金_...xlsx         │
   │ ・scenario_kawase_為替_...xlsx        │
```

**Step 2: FR-009画面イメージ内のファイル名例を修正**

```
old:
   │   📄 scenario_cat-yokin_預金_...xlsx     │
   │   📄 scenario_cat-kawase_為替_...xlsx    │
new:
   │   📄 scenario_yokin_預金_...xlsx         │
   │   📄 scenario_kawase_為替_...xlsx        │
```

---

### Task 11: 2.3節を2.4に統合 + セクション番号繰り上げ（低重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション2.3〜2.6

**Step 1: 2.3と2.4を統合**

```
old:
### 2.3 対象範囲の位置づけ

本システムは、担当者がTeams Bot上で事務改定の内容をテキスト入力し、AI検索で影響候補を検出・判断するシステムである。

### 2.4 対象範囲

**対象範囲:**

new:
### 2.3 対象範囲

本システムは、担当者がTeams Bot上で事務改定の内容をテキスト入力し、AI検索で影響候補を検出・判断するシステムである。

**対象範囲:**
```

**Step 2: 2.5→2.4、2.6→2.5 にセクション番号を繰り上げ**

```
old: ### 2.5 システム概念図
new: ### 2.4 システム概念図

old: ### 2.6 ローカル検証からの設計判断
new: ### 2.5 ローカル検証からの設計判断
```

---

### Task 12: FR-005 ページネーション記述の簡素化（低重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション4.2 FR-005

**Step 1: 「二分探索アルゴリズム」と「UTF-8バイト計測」を削除して簡素化**

技術補足（行421付近）に二分探索の詳細は残っているため、ここでは概要のみ。

```
old: - **ページネーション方式**: ユーザーが選択した表示件数（10〜100件）を初期値とし、カードサイズ25KB（UTF-8バイト計測）を超過する場合はBotが二分探索アルゴリズムで1ページあたりの表示件数を動的に削減する。そのため、実際の表示件数はユーザー選択値以下になる場合がある。「← 前ページへ」「次ページへ →」ボタンで遷移
new: - **ページネーション方式**: ユーザーが選択した表示件数（10〜100件）を初期値とし、カードサイズ上限（25KB）を超過する場合はBotが1ページあたりの表示件数を自動調整する。そのため、実際の表示件数はユーザー選択値以下になる場合がある。「← 前ページへ」「次ページへ →」ボタンで遷移
```

---

### Task 13: 6.3リソース構成表にKey Vault / Log Analytics追記（低重要度）

**Files:**
- Modify: `docs/要件定義書.md` — セクション6.3

**Step 1: Application Insightsの後に2行追加**

アーキテクチャ図（6.1）に描かれているがリソース構成表に未掲載のリソースを追記。

```
old:
| Application Insights | — | 監視・ログ |

**注記:**

new:
| Application Insights | — | 監視・ログ |
| Log Analytics Workspace | — | Application Insightsのバックエンド |
| Key Vault | Standard | PoC段階では未使用。本番移行時に導入検討 |

**注記:**
```

---

### Task 14: コミット

**Step 1: git add + commit**

```bash
git add "docs/要件定義書.md"
git commit -m "docs: 要件定義書 最終レビュー修正（テキスト版削除+13項目修正）

- テキスト版（参考）5箇所削除（draw.io PNG画像があるため不要）
- Web App B1コスト修正: ~¥2,000→~¥8,000（Windowsデプロイ）
- AI Search備考修正: Semantic Ranker無料枠の明記
- 合計コスト修正: ~¥15,500→~¥22,500
- FR-009 Excel列定義: 回答列をLvカラム統合に修正
- combinedContent: Skillsetベクトル化入力ソースの用途追記
- データフロー図: Indexer A/B 2本構成に修正
- Indexer間隔記述統一、ファイル名例修正
- 2.3節統合+セクション番号繰り上げ
- FR-005ページネーション簡素化、未使用用語削除、6.3表補完

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## 実行順序と依存関係

```
Task 1 (テキスト版削除) — 後方から削除、行番号ズレ最小化
  ↓
Task 2-13 (個別修正) — Edit文字列マッチのため順序自由、ただし以下に注意:
  - Task 3→4→5: コスト修正は Web App → AI Search → 合計 の順
  - Task 11: セクション番号変更は Task 1 のテキスト削除後
  ↓
Task 14 (コミット)
```

全Taskは単一ファイルへの逐次修正。並列実行は不要。
