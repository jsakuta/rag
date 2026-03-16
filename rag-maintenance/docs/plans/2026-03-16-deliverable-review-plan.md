# 納品物レビュー修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 要件定義書・検索設計書を顧客引き継ぎ用の納品物品質に仕上げる（版数変更なし）

**Architecture:**
- 要件定義書: FR番号振り直し・FR-004表現修正・付録A定量化・Key Vault整理・件数確認
- 検索設計書: メタデータ追加・目次修正・件数確認
- draw.io図: ASCII図→ベクトル図化（6図）＋ PNGエクスポートスクリプト
- 両文書とも版数は v1.0 のまま変更しない

**Tech Stack:** Markdown編集 + draw.io XML + Python（エクスポートスクリプト）

**修正対象ファイル:**
- `docs/要件定義書.md`
- `docs/検索設計書.md`
- `docs/drawings/*.drawio`（新規作成 × 6）
- `scripts/export-drawings.py`（新規作成）

---

## Task 1: 件数実測値の検証

**目的:** 全文書の件数記載が実測値と合っているかを確認し、差異があれば Task 3/4 で修正する。差異がなければ Task 3 Step 4 と Task 4 Step 3 をスキップする。

**Files:**
- 参照のみ（ファイル変更なし）

**Step 1: AI Search インデックスの実件数を取得**

```bash
AI_SEARCH_KEY=$(az search admin-key show \
  -g rg-maintenance-poc \
  --service-name srch-maintenance-poc \
  --query primaryKey -o tsv)

# カテゴリ×タイプ別 facet（削除済み除外）
az rest --method POST \
  --url "https://srch-maintenance-poc.search.windows.net/indexes/maintenance-search-index/docs/search?api-version=2024-07-01" \
  --headers "api-key=$AI_SEARCH_KEY" \
  --body '{
    "search": "*",
    "top": 0,
    "count": true,
    "facets": ["categoryId,count:20", "dataType,count:5"],
    "filter": "isDeleted eq false"
  }'
```

**Step 2: 結果を以下のマッピング表と照合する**

| categoryId | dataType | 文書記載値 | 実測値 | 差分 |
|-----------|----------|-----------|--------|------|
| smile | scenario | 555 | ? | ? |
| souzoku | scenario | 269 | ? | ? |
| naibujimu | scenario | 1,384 | ? | ? |
| torikaku | scenario | 105 | ? | ? |
| smile | faq | 8,679 | ? | ? |
| sousoku | faq | 4,000 | ? | ? |
| yokin | faq | 6,055 | ? | ? |
| **合計** | | **21,047** | ? | ? |

**Step 3: 判断**

- **差分がすべて 0** → Task 3 Step 4 および Task 4 Step 3 をスキップ。現行文書値が正しい。
- **差分あり** → テスト残骸データが残っている可能性。以下を確認してから件数修正:
  - `filter` に `categoryId ne 'cat-yokin' and categoryId ne 'cat-kawase'` 等を追加して不正カテゴリを特定
  - 不正データを Cosmos DB から削除してから件数を確定する（削除手順は `scripts/upload-data.js --clean` 参照）
  - 削除後に Step 1 を再実行して最終値を確定

---

## Task 2: 要件定義書 — FR番号振り直し + FR-004修正

**Files:**
- Modify: `docs/要件定義書.md`

### 現状と振り直しマッピング

| 旧ID | 新ID | 機能名 | 対象 |
|------|------|--------|------|
| FR-001 | FR-001 | 改定内容テキスト入力 | ✓（変更なし） |
| FR-002 | FR-002 | 検索モード選択 | ✓（変更なし） |
| FR-003 | FR-003 | 影響候補検索（意味検索） | ✓（変更なし） |
| FR-004 | FR-004 | 影響候補検索（キーワード検索） | ✓（変更なし） |
| FR-005 | FR-005 | 候補一覧表示 | ✓（変更なし） |
| FR-006 | FR-006 | データマスタ同期 | ✓（変更なし） |
| FR-010 | FR-007 | レビューUI | 対象外（本格実装） |
| FR-011 | FR-008 | 判定入力・進捗管理 | 対象外（本格実装） |
| FR-012 | FR-009 | FAQ一括削除 | ✓ |
| FR-013 | FR-010 | シナリオ要修正フラグ記録 | ✓ |
| FR-014 | FR-011 | シナリオ要修正Excel出力 | ✓ |

**Step 1: 全参照箇所を洗い出す**

```bash
grep -n "FR-01[0-4]" docs/要件定義書.md
```

期待される主要参照箇所（確認のみ、全件 grep で確定）:

| 行（概算） | 箇所 | 旧 → 新 |
|-----------|------|---------|
| L228 | 4.1節テーブル | FR-010→FR-007 |
| L229 | 4.1節テーブル | FR-011→FR-008 |
| L230 | 4.1節テーブル | FR-012→FR-009 |
| L231 | 4.1節テーブル | FR-013→FR-010 |
| L232 | 4.1節テーブル | FR-014→FR-011 |
| L415 | FR-005技術補足 | FR-013→FR-010, FR-014→FR-011 |
| L417 | 4.2節見出し | FR-012→FR-009 |
| L459 | 4.2節見出し | FR-013→FR-010 |
| L481 | FR-013完了カード図 | FR-014→FR-011 |
| L485 | FR-013注記 | FR-014→FR-011 |
| L519 | 4.2節見出し | FR-014→FR-011 |
| L523-524 | FR-014概要・トリガー | FR-013→FR-010 |
| L531 | FR-014処理フロー | FR-013→FR-010 |
| L806 | 7.1節データ構成 | FR-013→FR-010 |
| L898 | 7.3節データフロー | FR-013→FR-010 |
| L910 | 7.4節保持期間 | FR-013→FR-010 |
| L962 | 8.3節Bot→CosmosDB書き込み | FR-012→FR-009, FR-013→FR-010 |
| L985 | 8.5節注記 | FR-014→FR-011 |
| L1023 | 10.1節認証 | FR-012→FR-009, FR-013→FR-010 |

**Step 2: 置換を大きい番号から実施（カスケード防止）**

置換順序: FR-014 → FR-011 → FR-013 → FR-010 → FR-012 → FR-009 → FR-011 → FR-008 → FR-010 → FR-007

```bash
# 確認用（置換前に全件確認）
grep -c "FR-014\|FR-013\|FR-012\|FR-011\|FR-010" docs/要件定義書.md
```

Editツールで各箇所を個別に置換する。`replace_all` は旧IDが新IDと衝突するリスクがあるため**使用禁止**。必ず個別のEdit呼び出しで対象行を特定してから置換する。

**Step 3: 置換後の検証**

```bash
# 旧番号が残っていないことを確認
grep -n "FR-01[0-4]" docs/要件定義書.md
# 期待: 0件
```

**Step 4: FR-004 の「全件」表現を修正**

> **前提確認:** `agent.ts:769-773` でキーワード検索も `top: topN` が適用されており、最大100件。「全件を漏れなく抽出する」は技術的に不正確。ただし、ケース⑤は正解4件すべてが候補4件内に収まっており、実態として100%検知可能。修正は表現を正確にするが、100%検知の実績は付録Aで担保。

行番号は Step 2-3 の置換後に `grep -n "全件" docs/要件定義書.md` で再確認してから編集する。

```markdown
# Before
| 概要 | ユーザーが指定したキーワードを含む全件を漏れなく抽出する |
| 入力 | 検索キーワード（1つ以上） |
| 出力 | キーワードを含む全件のリスト |

# After
| 概要 | ユーザーが指定したキーワードに関連する候補を語句ベースで抽出する |
| 入力 | 検索キーワード（1つ以上） |
| 出力 | キーワード関連候補リスト（BM25スコア順、上位N件。Nはユーザー選択の表示件数：10〜100件） |
```

> 「ローカル版のケース⑤⑥で100%検知を確認済み。」（FR-004本文末）はそのまま残す。

**Step 5: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書 FR番号を連番に振り直し（007-009欠番解消）+ FR-004全件表現修正"
```

---

## Task 3: 要件定義書 — 付録A定量化 + Key Vault整理 + 件数更新

**Files:**
- Modify: `docs/要件定義書.md`

**Step 1: 付録A「ほぼ全件検知」を定量化**

> **データの出典:** `rag-local/data/output/examples/rev_eval_batch_20260306_025442.xlsx`（サマリーシート）
> これは**ローカル版**（ChromaDB + Azure OpenAI text-embedding-3-large）での検証結果。Azure AI SearchのRRF+Semantic Rankerとは異なる構成。付録AはPhase1ローカル版の結果として明記する。

`grep -n "付録A\|Phase1\|ほぼ全件" docs/要件定義書.md` で行番号を確認してから編集。

```markdown
# Before
### 付録A: Phase1精度検証結果サマリ

| ケース | 改定タイプ | 検索方式 | Azure OpenAI結果 | 備考 |
|--------|----------|---------|-----------------|------|
| ① | 意味的関連（内容変更型） | ベクトル検索 | ほぼ全件検知 | — |
| ② | 意味的関連（内容変更型） | ベクトル検索 | ほぼ全件検知 | — |
| ③ | 意味的関連（内容変更型） | ベクトル検索 | ほぼ全件検知 | — |
| ④ | 意味的関連（内容変更型） | ベクトル検索 | ほぼ全件検知 | 1件未検知（画像内テキスト変更） |
| ⑤ | キーワード変更型 | ハイブリッド検索 | **100%検知** | AMLフィルター→ジープレックス |
| ⑥ | キーワード変更型 | ハイブリッド検索 | **100%検知** | スーパーカード(DC)→(MDC) |

# After
### 付録A: Phase1精度検証結果サマリ

> **検証構成:** ローカル版（ChromaDB + Azure OpenAI text-embedding-3-large）。Azure AI Searchのハイブリッド検索+Semantic Rankerとは別構成。クラウド版の精度については、本番環境での継続的な検索テストで検証する。

| ケース | 改定タイプ | 検索方式 | 正解数 | 検知数 | 検知率 | 備考 |
|--------|----------|---------|--------|--------|--------|------|
| ① | 意味的関連（内容変更型） | ベクトル検索 | 3 | 3 | 100% | — |
| ② | 意味的関連（内容変更型） | ベクトル検索 | 6 | 6 | 100% | — |
| ③ | 意味的関連（内容変更型） | ベクトル検索 | 16 | 15 | 93.8% | 4エリア横断（naibujimu/smile/souzoku/torikaku）。1件未検知（souzoku-bot_146：ベクトル検索の上位範囲外。VertexAI Embeddingでは100%検知） |
| ④ | 意味的関連（内容変更型） | ベクトル検索 | 1 | 1 | 100% | 別途、画像内テキスト変更1件は今回スコープ外（対象外のため計測対象に含まない） |
| ⑤ | キーワード変更型 | キーワード検索 | 4 | 4 | **100%** | AMLフィルター→ジープレックス |
| ⑥ | キーワード変更型 | キーワード検索 | 32 | 32 | **100%** | スーパーカード(DC)→(MDC) |

**合計: 62件中61件検知（検知率 98.4%）。** 画像内テキスト変更（④備考）は今回スコープ外のため計測対象外。
```

**Step 2: 「主要な知見」を更新**

```markdown
# Before（現行）
- ケース⑤⑥（名称変更型）はベクトル検索だけでは検知不可。キーワード検索との併用で100%検知を達成
- AIが人間の見落とし2件（小規模企業共済関連）を発見
- 原文検索とLLMクエリ拡張検索で精度に有意な差なし → 原文検索に統一する判断根拠
- 1件未検知（ケース④）: 画像内テキストの変更。今回スコープ外のため、別途課題として扱う

# After
- **全体検知率 98.4%**（62件中61件）。ケース①②④⑤⑥は100%検知
- ケース③の1件未検知（souzoku-bot_146）はAzure OpenAI Embeddingの上位範囲外。VertexAI Embeddingでは100%検知しており、Embeddingモデルの特性差による
- ケース⑤⑥（名称変更型）はベクトル検索では検知不可。キーワード検索との併用で100%検知を達成
- AIが人間の見落とし2件（小規模企業共済関連）を発見
- 原文検索とLLMクエリ拡張検索で精度に有意な差なし → 原文検索に統一する判断根拠
- 画像内テキスト変更（ケース④備考）は今回スコープ外。別途課題として管理
```

**Step 3: Key Vault をコスト表・リソース構成から除外**

**コスト表（6.4節）から Key Vault 行を削除:**

```markdown
# 削除対象行
| Key Vault (Standard) | ~¥500 | ※PoC段階ではManaged Identity認証を使用しKey Vaultは未使用。本番移行時に導入を検討 |

# 合計行も更新（~¥15,500 → 変わらないはずだが確認）
# Key Vault除くの注記も削除
| **合計** | **~¥15,500/月** | — |
```

**リソース構成テーブル（6.3節）から Key Vault 行を削除:**

```markdown
# 削除対象行
| Key Vault | Standard | シークレット管理（※PoC段階では未使用。本番移行時に導入検討） |
```

**6.3節注記（L755付近）の文言確認:**

```bash
grep -n "Key Vault" docs/要件定義書.md
```

注記「Key Vault、Application Insights、Azure Bot Service は既存環境または別手順で用意する前提」は、Key Vault を構成から除外した場合に残す意味がなくなる。次の通り修正:

```markdown
# Before
`maintenance-bot/infra/azure.bicep` がこの repo で直接作成するのは ... Azure AI Search、Azure OpenAI、Cosmos DB、Key Vault、Application Insights、Azure Bot Service は既存環境または別手順で用意する前提とする。

# After
`maintenance-bot/infra/azure.bicep` がこの repo で直接作成するのは ... Azure AI Search、Azure OpenAI、Cosmos DB、Application Insights、Azure Bot Service は既存環境または別手順で用意する前提とする。Key Vault はPoC段階では未使用。本番移行時に導入を検討する。
```

> 8.2節「本番移行時はAPIキー（Key Vault経由）も検討」は将来方針として残す。

**Step 4: 件数更新（Task 1 で差分ありの場合のみ実施。差分なしはスキップ）**

Task 1 で実測値と現行文書値に差分があった場合のみ以下を実施:

```bash
# 現行値の全箇所を確認
grep -n "2,313\|18,734\|21,047" docs/要件定義書.md
```

変更箇所（Task 1 の実測値に置き換え）:

| 行 | 箇所 | 変更内容 |
|----|------|---------|
| L53 | 2.1節 背景 | FAQ件数 |
| L72 | 2.4節 対象範囲 | 合計件数 |
| L211 | 3.4節 シナリオ | シナリオ検証値 |
| L212 | 3.4節 FAQ | FAQ検証値 |
| L659 | 5.3節 シナリオ現状 | シナリオ件数 |
| L660 | 5.3節 FAQ現状 | FAQ件数 |
| L772 | 6.4節 Embeddingコスト | 件数ベース試算（式ごと再計算） |
| L804 | 7.1節 シナリオ | シナリオ検証値 |
| L805 | 7.1節 FAQ | FAQ検証値 |
| L807 | 7.1節 インデックス | 合計件数 |

**Step 5: 最終検証**

```bash
grep -n "ほぼ全件" docs/要件定義書.md  # 期待: 0件
grep -n "Key Vault" docs/要件定義書.md  # 期待: 8.2節の将来方針のみ
```

**Step 6: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書 付録A定量化（98.4%）+ Key Vault整理 + 件数更新"
```

---

## Task 4: 検索設計書 — メタデータ追加 + 目次修正 + 件数更新

**Files:**
- Modify: `docs/検索設計書.md`

**Step 1: 既存の uncommitted changes を確認**

```bash
git diff docs/検索設計書.md
git status docs/検索設計書.md
```

変更があれば内容を確認してから編集する。

**Step 2: メタデータヘッダーを要件定義書と同形式に統一（L1-5）**

```markdown
# Before
# 運用保守効率化AI 検索設計書

- 版数: 1.0
- 目的: クラウド版（Azure AI Search）の検索設計を1文書で網羅し、設計意図・パラメータ・チューニング経緯を記録する

# After
# 運用保守効率化AI 検索設計書

**版数**: 1.0

**作成日**: 2026年3月3日

**作成者**: B&DX 作田

**目的**: クラウド版（Azure AI Search）の検索設計を1文書で網羅し、設計意図・パラメータ・チューニング経緯を記録する
```

**Step 3: 目次から12章を削除（L22付近）**

```bash
grep -n "12\." docs/検索設計書.md
```

```markdown
# Before
11. [チューニング経緯](#11-チューニング経緯)
12. [将来の改善候補](#12-将来の改善候補)

# After
11. [チューニング経緯](#11-チューニング経緯)
```

本文末尾に `## 12.` の見出しが残っていないことも確認:

```bash
grep -n "^## 12" docs/検索設計書.md  # 期待: 0件
```

**Step 4: 件数更新（Task 1 で差分ありの場合のみ実施。差分なしはスキップ）**

```bash
grep -n "21,047\|2,313\|8,679\|4,000\|6,055\|18,734" docs/検索設計書.md
```

変更箇所（Task 1 の実測値に置き換え）:

| 行 | 箇所 |
|----|------|
| L58-61 | 3節 シナリオ別件数（smile/souzoku/naibujimu/torikaku） |
| L68 | 3節 インデックス合計件数 |
| L72-74 | 3節 FAQ別件数（smile-faq/sousoku/yokin） |
| L268, L283 | 7-A節 preFilter/postFilter図の「全X件」 |
| L297 | 7-B節 torikaku(105件), souzoku(269件) |
| L370 | 9-A節 データ件数 |
| L436-442 | 10-A節 カテゴリ一覧の全件数 |

**Step 5: コミット**

```bash
git add docs/検索設計書.md
git commit -m "docs: 検索設計書にメタデータ追加 + 目次修正（12章削除）+ 件数更新"
```

---

## Task 5: draw.io 図の作成

**目的:** 要件定義書・検索設計書のASCII図を draw.io 形式に変換し、PNG画像をMarkdownに埋め込む

**Files:**
- Create: `docs/drawings/system-overview.drawio`
- Create: `docs/drawings/as-is-flow.drawio`
- Create: `docs/drawings/to-be-flow.drawio`
- Create: `docs/drawings/search-flow.drawio`
- Create: `docs/drawings/azure-architecture.drawio`
- Create: `docs/drawings/search-architecture.drawio`
- Create: `scripts/export-drawings.py`

### 作成する6図

| # | ファイル名 | 元ASCII図 | 種別 |
|---|-----------|----------|------|
| 1 | system-overview | 要件定義書 2.5節 L89-111 | アーキテクチャ図（入力→処理→出力） |
| 2 | as-is-flow | 要件定義書 3.1節 L130-156 | スイムレーン（管理担当者/各カテゴリ担当/デジタル戦略部） |
| 3 | to-be-flow | 要件定義書 3.2節 L160-196 | スイムレーン（担当者/システム/担当者） |
| 4 | search-flow | 要件定義書 FR-003 L283-311 | フローチャート（原文検索方式） |
| 5 | azure-architecture | 要件定義書 6.1節 L678-726 | Azure構成図（リソース図） |
| 6 | search-architecture | 検索設計書 3節 L53-110 | アーキテクチャ図（CosmosDB→Indexer→AI Search→Bot） |

**Step 1: ディレクトリ作成**

```bash
mkdir -p docs/drawings
```

**Step 2: `creating-diagrams` スキルで6図を順次作成**

> 各図の作成は `Skill: creating-diagrams` を呼び出す。以下に各図のプロンプト仕様を記載する。

---

**図1: system-overview.drawio — システム概念図**

```
スタイル: シンプルなアーキテクチャ図（左→右）
レイアウト: 入力ブロック → 処理ブロック → 出力ブロック の3列

入力ブロック（左）:
  - ラベル: "Teams Bot / テキスト入力"
  - 内容: "・改定内容（自然言語）\n・検索モード選択"

処理ブロック（中央）:
  - ラベル: "Azure AI Search (Basic)"
  - 内容: "・ハイブリッド検索（BM25+ベクトル）\n・Semantic Ranker\n・ビルトインVectorizer"

出力ブロック（右）:
  - ラベル: "Teams / Adaptive Card"
  - 内容: "・シナリオタブ\n・FAQタブ\n・スコア順表示"

下部注記:
  - "Bot バックエンド: Azure Web App (App Service)"
  - "ベクトル化: Azure OpenAI text-embedding-3-large"
```

---

**図2: as-is-flow.drawio — 現行業務フロー（As-Is）**

```
スタイル: スイムレーン図（縦）
スイムレーン（左から右）: 管理担当者 | 各カテゴリ担当 | デジタル戦略部

フロー:
管理担当者:
  1. 通達変更を手動監視（属人的）
  2. 影響範囲を手動調査（見落としリスクあり）
  3. メール/口頭で連絡 → 各カテゴリ担当へ矢印

各カテゴリ担当:
  4. FAQ更新依頼 → デジタル戦略部へ矢印

デジタル戦略部:
  5. FAQ更新

課題ボックス（スイムレーン外下部）:
  "課題: 影響調査が属人化・見落とし発生 / 複数カテゴリへの影響把握が困難 / シナリオ修正漏れ・FAQ削除漏れ"
```

---

**図3: to-be-flow.drawio — 目標業務フロー（To-Be）**

```
スタイル: スイムレーン図（縦）
スイムレーン（左から右）: 担当者 | システム | 担当者（右端）

フロー:
担当者:
  1. Teamsで改定内容をテキスト入力 → システムへ矢印

システム:
  2. AI検索実行（ハイブリッド検索 + Semantic Ranker）
  3. Adaptive Cardで候補一覧を返却 → 担当者（左）へ矢印

担当者（左）:
  4. 候補を確認
  4a. シナリオタブ → 「要修正」チェック → 保存（DB記録）→ Excelで出力
  4b. FAQタブ → 削除対象を選択 → 一括削除（DB更新）
  → システムへ矢印

システム:
  5. Cosmos DB更新（要修正フラグ / FAQ論理削除）
  6. 完了カードを返却 → 担当者（左）へ矢印

効果ボックス（下部）:
  "効果: AI支援による網羅的検出 / カテゴリ横断の影響候補を一覧 / 属人性の排除"
```

---

**図4: search-flow.drawio — 原文検索方式フロー**

```
スタイル: フローチャート（上→下）

ノード:
  1. 入力: 「入力テキスト（原文）」（例: 「本人確認書類が2点から1点に変更」）
  2. 処理ボックス: "Azure AI Search ハイブリッド検索"
     内部:
     ① ビルトインVectorizerが原文テキストを自動ベクトル化 → ベクトル検索（cosine）
     ② 同時にキーワード検索（ja.microsoftアナライザー）→ BM25スコアリング
     ③ RRF（Reciprocal Rank Fusion）でスコア統合
  3. 処理ボックス: "Semantic Ranker で最終スコアリング"
     内部: rerankerScore を優先して関連度順に整列 / 上位N件を返却（デフォルト: 30件）
  4. 出力: "Teams Adaptive Card（影響候補一覧）"

注記: "Botアプリ → AI Search REST API 直接呼び出し（LLMクエリ拡張なし）"
```

---

**図5: azure-architecture.drawio — Azure構成図**

```
スタイル: Azure構成図（Azureアイコンセット使用）
全体ラベル: "Azure Resource Group: rg-maintenance-poc"

コンポーネント（上段）:
  - [Teams（クライアント）] → [Azure Bot Service F0] → [Azure Web App B1（Botバックエンド）]

コンポーネント（中段左）:
  [Azure Web App B1] → [Azure AI Search Basic]
  [Azure AI Search Basic]:
    ・ハイブリッド検索（BM25+ベクトル+RRF）
    ・Semantic Ranker
    ・ビルトインVectorizer
    ・Indexer×2（scenarios/faqs）
    ・AzureOpenAIEmbedding Skill

コンポーネント（中段右から分岐）:
  [Azure AI Search Basic] → [Azure OpenAI S0（text-embedding-3-large）]
  [Azure AI Search Basic] ← [Cosmos DB Serverless（scenarios/faqs/impactAssessments）]
  [Azure Web App B1] → [Cosmos DB Serverless]

コンポーネント（下段）:
  [Managed Identity（サービス間認証）]
  [Application Insights（監視・ログ）]
  [Log Analytics]
```

---

**図6: search-architecture.drawio — 検索アーキテクチャ全体像**

```
スタイル: アーキテクチャ図（左→右）
大枠: "Cosmos DB" | "AI Search" | "Bot"

左列（Cosmos DB）:
  scenariosコンテナ:
    ├── smile (555件)
    ├── souzoku (269件)
    ├── naibujimu (1,384件)
    └── torikaku (105件)
  faqsコンテナ:
    ├── smile (8,679件)
    ├── sousoku (4,000件)
    └── yokin (6,055件)
  ※件数はTask 1の実測値で上書きすること

中央列（AI Search）:
  maintenance-scenarios-indexer
  ↓（AzureOpenAIEmbedding Skill経由）
  maintenance-search-index（合計21,047件）
  maintenance-faqs-indexer
  ↓（同上）
  ↑（同インデックスに書き込み）

右列（Bot）:
  searchSingle() in agent.ts
  ← maintenance-search-index

補足フロー:
  AI Search → Azure OpenAI（Skillset/Vectorizer）
```

---

**Step 3: PNGエクスポートスクリプトを作成**

draw.io デスクトップアプリの CLI 機能を使ってバッチエクスポートする。

> draw.io desktop CLI ドキュメント: https://www.drawio.com/doc/faq/export-diagram-html-cli

`scripts/export-drawings.py` を作成:

```python
#!/usr/bin/env python
"""
draw.io ファイルを PNG にバッチエクスポートするスクリプト。

前提: draw.io デスクトップアプリがインストールされていること。
     Windows: C:\Program Files\draw.io\draw.io.exe
     macOS:   /Applications/draw.io.app/Contents/MacOS/draw.io

使い方:
    python scripts/export-drawings.py
    python scripts/export-drawings.py --scale 2  # 2倍解像度
    python scripts/export-drawings.py --drawio-path "C:/path/to/draw.io.exe"
"""

import argparse
import subprocess
import sys
from pathlib import Path

# デフォルトのdraw.io実行パス（Windows/macOS対応）
DEFAULT_PATHS = [
    r"C:\Program Files\draw.io\draw.io.exe",
    r"C:\Program Files (x86)\draw.io\draw.io.exe",
    "/Applications/draw.io.app/Contents/MacOS/draw.io",
]

DRAWINGS_DIR = Path(__file__).parent.parent / "docs" / "drawings"


def find_drawio_exe(override: str | None = None) -> Path | None:
    if override:
        p = Path(override)
        return p if p.exists() else None
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists():
            return p
    return None


def export_drawio(drawio_exe: Path, src: Path, dst: Path, scale: float) -> bool:
    """1ファイルをPNGにエクスポートする。成功時True。"""
    cmd = [
        str(drawio_exe),
        "--export",
        "--format", "png",
        "--scale", str(scale),
        "--output", str(dst),
        str(src),
    ]
    print(f"Exporting: {src.name} -> {dst.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return False
    print(f"  OK: {dst}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draw.io files to PNG")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Export scale factor (default: 2.0 for Retina quality)")
    parser.add_argument("--drawio-path", default=None,
                        help="Path to draw.io executable (optional)")
    args = parser.parse_args()

    drawio_exe = find_drawio_exe(args.drawio_path)
    if drawio_exe is None:
        print("ERROR: draw.io executable not found.", file=sys.stderr)
        print("Install draw.io desktop from https://www.drawio.com/", file=sys.stderr)
        print("Or specify path with --drawio-path", file=sys.stderr)
        sys.exit(1)

    print(f"Using draw.io: {drawio_exe}")
    print(f"Scale: {args.scale}x")
    print()

    drawio_files = list(DRAWINGS_DIR.glob("*.drawio"))
    if not drawio_files:
        print(f"No .drawio files found in {DRAWINGS_DIR}")
        sys.exit(0)

    success, failed = 0, 0
    for src in sorted(drawio_files):
        dst = src.with_suffix(".png")
        if export_drawio(drawio_exe, src, dst, args.scale):
            success += 1
        else:
            failed += 1

    print()
    print(f"Done: {success} exported, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 4: スクリプトの動作確認**

```bash
python scripts/export-drawings.py --help

# draw.io がインストール済みなら実行
python scripts/export-drawings.py
```

期待出力:
```
Using draw.io: C:\Program Files\draw.io\draw.io.exe
Scale: 2.0x

Exporting: system-overview.drawio -> system-overview.png
  OK: docs/drawings/system-overview.png
...
Done: 6 exported, 0 failed
```

> draw.io がインストールされていない場合: `pip show drawio` / 手動インストール後に再実行。エクスポートは手動（VSCode draw.io拡張のExport機能）でも可。

**Step 5: Markdown へ図の参照を追加**

エクスポート済みPNGをMarkdownに埋め込む。**ASCII図は削除せずに `<details>` で折りたたむ（Word化対応のため `<details>` タグはオプション）。**

> **Word化の場合:** `<details>` タグはWordで正しく変換されない場合がある。Word化ツール（Pandoc等）を使う場合は、ASCII図を完全に削除してPNG参照のみにする。今回はMarkdown確認用として `<details>` を使うが、Word化時は除去すること。

要件定義書の各ASCII図の直前に以下を挿入（行番号はTask 2-3完了後の実際の行を `grep -n` で確認）:

```markdown
![システム概念図](drawings/system-overview.png)

<details>
<summary>テキスト版（参考）</summary>

（元のASCII図コードブロックをここに移動）

</details>
```

同様に残り5図も対応（as-is-flow / to-be-flow / search-flow / azure-architecture / search-architecture）。

検索設計書も同様に search-architecture.png を追加。

**Step 6: コミット**

```bash
git add docs/drawings/ scripts/export-drawings.py docs/要件定義書.md docs/検索設計書.md
git commit -m "docs: ASCII図をdraw.io形式に変換（6図）+ PNGエクスポートスクリプト追加"
```

---

## Task 6: 最終整合性チェック

**Files:**
- 参照のみ（差異があれば修正）
- 版数は両文書とも v1.0 のまま変更しない

**Step 1: 全文検索チェック（要件定義書）**

```bash
cd docs

# 旧FR番号の残存
grep -n "FR-01[0-4]" 要件定義書.md
# 期待: 0件

# 「全件」表現の残存
grep -n "全件" 要件定義書.md
# 期待: 0件（FR-004本文末の「100%検知を確認済み」のような文脈は別途確認）

# 「ほぼ全件」の残存
grep -n "ほぼ全件" 要件定義書.md
# 期待: 0件

# Key Vault の残存（将来方針のみ許容）
grep -n "Key Vault" 要件定義書.md
# 期待: 8.2節の1件のみ（「本番移行時はAPIキー（Key Vault経由）も検討」）

# 旧件数の残存（Task 1で変更なし確認なら旧値のまま＝正常）
grep -n "21,047\|2,313\|18,734" 要件定義書.md
```

**Step 2: 全文検索チェック（検索設計書）**

```bash
# 12章の痕跡
grep -n "12\." 検索設計書.md
# 期待: 目次の11章以降に12章の記載なし

# 版数・作成日・作成者が追加されているか
grep -n "版数\|作成日\|作成者" 検索設計書.md
# 期待: 3件
```

**Step 3: 目次とセクションの整合確認**

要件定義書:
```bash
grep -n "^## \|^### " 要件定義書.md | head -40
```
目次（1〜13節）と実際の `## X.` 見出しが1対1で対応していることを確認。

検索設計書:
```bash
grep -n "^## \|^### " 検索設計書.md
```
目次（1〜11節）と実際の `## X.` 見出しが対応していることを確認。12節の見出しがないことを確認。

**Step 4: draw.io PNG参照の確認**

```bash
grep -n "drawings/" 要件定義書.md | grep ".png"
# 期待: 5件（system-overview, as-is-flow, to-be-flow, search-flow, azure-architecture）

grep -n "drawings/" 検索設計書.md | grep ".png"
# 期待: 1件（search-architecture）
```

PNGファイルが実際に存在するか確認:
```bash
ls docs/drawings/*.png
# 期待: 6ファイル
```

**Step 5: 最終コミット**

差異があれば修正してからコミット:
```bash
git add docs/要件定義書.md docs/検索設計書.md
git commit -m "docs: 納品物最終整合性チェック（版数変更なし）"
```

---

## 実行順序と依存関係

```
Task 1 (件数検証)  ─────────────────────────────────────────┐
  ↓                                                          │ 実測値を引き渡し
Task 2 (FR振り直し)       ←────────────────── 独立実行可能  │
  ↓ FR番号確定                                               │
Task 3 (付録A+KV+件数)    ←─────────────── Task 1 完了後   ─┤
Task 4 (検索設計書)        ←─────────────── Task 1 完了後   ─┘
  ↓ 全ファイル確定
Task 5 (draw.io図)         ←─── Task 2-4 完了後（件数・FR番号確定後）
  ↓
Task 6 (最終チェック)      ←─── Task 5 完了後
```

**推定所要時間:**

| Task | 内容 | 推定時間 |
|------|------|---------|
| 1 | 件数検証 | 5分 |
| 2 | FR振り直し + FR-004修正 | 20分（慎重に） |
| 3 | 付録A定量化 + KV整理 + 件数更新 | 15分 |
| 4 | 検索設計書修正 | 10分 |
| 5 | draw.io 6図 + スクリプト | 45〜60分 |
| 6 | 最終チェック | 10分 |
| **合計** | | **約2時間** |

---

## 対象外（今回修正しないもの）

| 項目 | 理由 |
|------|------|
| 版数変更 | 両文書とも v1.0 のまま（ユーザー判断） |
| 改定履歴テーブル | ユーザー判断で不要 |
| 文書番号（DESIGN-SEARCH-001等） | ユーザー判断で不要 |
| 導入手順書の件数 | 概数表記（「約2,300件」等）のため変更不要 |
| PoC期間・完了基準 | 今回スコープ外 |
| 対象読者の更新 | 引き継ぎ先確定後に判断 |
| FR-005の検索カード/結果カードASCII図 | UI仕様図のためdraw.io化対象外 |
| FR-006の差分ベクトル化テキスト図 | テキストの方が可読性高いため対象外 |
