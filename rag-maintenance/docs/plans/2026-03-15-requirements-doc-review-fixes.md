# 要件定義書 v1.0 レビュー指摘修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 要件定義書 v1.0 のレビュー指摘16件（重大5/中6/軽微5）を修正し、v1.1 に版上げする

**Architecture:** 対象ファイルは `docs/要件定義書.md`（1,244行）と `CLAUDE.md` の2ファイル。Step 2/オート議事録の全削除、ASCII図のコード実装との整合、用語統一、数値整合を一括修正する。

**Tech Stack:** Markdown編集のみ（コード変更なし）

**修正対象ファイル:**
- `docs/要件定義書.md` — メイン修正対象
- `CLAUDE.md` — 「検索UIの設計原則」セクションの更新

**レビュー実施内容:**
- 3エージェント並列レビュー（Step2残存洗い出し / コードUI乖離分析 / 文書品質全体レビュー）
- WEB調査5項目（Action.Executeタイムアウト / Multi-Tenant非推奨 / M365 Agents SDK GA / Indexer間隔 / Graph API Upload上限）

---

## 指摘一覧（全16件）

| # | 重大度 | カテゴリ | 概要 | 対応タスク |
|---|--------|---------|------|-----------|
| 1 | 重大 | Step2削除 | Step 2 / オート議事録の残存（13箇所） | Task 1 |
| 2 | 重大 | UI乖離 | 検索カードASCII図のカテゴリ表示が実装と矛盾 | Task 2 |
| 3 | 重大 | UI乖離 | 結果カードASCII図がシナリオ+FAQ同時表示（実装は片方のみ） | Task 2 |
| 4 | 重大 | 設計原則 | CLAUDE.mdの「検索UIの設計原則」が古い設計思想のまま | Task 3 |
| 5 | 重大 | セキュリティ | Key Vault管理方針の矛盾（未使用 vs 一元管理） | Task 4 |
| 6 | 重大 | 非機能 | NFR-001「10秒以内」とAction.Executeタイムアウト競合 | Task 4 |
| 7 | 中 | 数値 | データ件数の不整合（シナリオ/FAQ/合計） | Task 5 |
| 8 | 中 | 用語 | 「コンテナ」vs「コレクション」混在 | Task 6 |
| 9 | 中 | 用語 | sousoku/souzoku混同リスク | Task 6 |
| 10 | 中 | 数値 | Embeddingコスト試算が初期想定のまま | Task 5 |
| 11 | 中 | UI乖離 | 表示件数と動的調整の関係不明 | Task 2 |
| 12 | 中 | 技術 | Indexer最短間隔「Basic SKU」→全SKU共通 | Task 4 |
| 13 | 軽微 | 用語 | Skill/Skillset/ビルトインSkill定義不足 | Task 6 |
| 14 | 軽微 | 用語 | Single-Tenant定義なし | Task 6 |
| 15 | 軽微 | 検証 | 付録BにRBAC検証・AC互換性テストなし | Task 7 |
| 16 | 軽微 | 非機能 | NFR-004 Excel出力「5秒以内」にSPO遅延未考慮 | Task 4 |

---

## WEB調査結果サマリ

| 項目 | 要件定義書の記載 | WEB調査結果 | 対応 |
|------|----------------|-----------|------|
| Action.Execute タイムアウト | （明記なし） | 10〜15秒（チャネルにより異なる）[公式](https://learn.microsoft.com/en-us/azure/bot-service/bot-builder-howto-long-operations-guidance) | NFR-001備考に注記追加 |
| Multi-Tenant非推奨 | 「2025年7月以降非推奨」(L965) | 正確（2025年7月31日以降）[公式](https://learn.microsoft.com/en-us/azure/bot-service/provision-and-publish-a-bot) | 修正不要 |
| M365 Agents SDK | CLAUDE.mdに記載 | GA v1.3.1、パッケージ名正確 | 修正不要 |
| Bot Framework SDK終了 | 「2025/12/31」(L776) | 正確 [公式](https://github.com/microsoft/botframework-sdk) | 修正不要 |
| Indexer最短間隔 | 「Basic SKUの最短間隔は5分」(L1094) | 全SKU共通で5分 [公式](https://learn.microsoft.com/en-us/azure/search/search-howto-schedule-indexers) | 「Basic SKUの」を削除 |
| Graph API Upload上限 | 「公式上限は250MB」(L1016) | 正確 [公式](https://learn.microsoft.com/en-us/graph/api/driveitem-put-content) | 修正不要 |

---

## Task 1: Step 2 / オート議事録の全削除（13箇所）

**Files:**
- Modify: `docs/要件定義書.md` — 13箇所の削除・修正

**Step 1: 1.1節のStep 1限定表記を修正**

L31 を修正:
```markdown
# Before
本書は、事務改定影響検知システム 本システム（Step 1: 手入力検索）の要件を定義する。

# After
本書は、事務改定影響検知システム（以下「本システム」）の要件を定義する。
```

**Step 2: 2.3節の段階設計を修正**

L76-83 を修正:
```markdown
# Before
### 2.3 段階設計

本システムは段階的に機能を拡充する。現段階ではStep 1を対象とする。

| 段階 | トリガー | 概要 | 状態 |
|------|---------|------|------|
| Step 1 | 担当者がTeamsで手入力 | AI検索 → 影響候補表示 → 担当者判断 | ← **対象** |
| Step 2 | オート議事録から自動抽出 | 会議での改定決定 → 自動検索 → Teams通知 | 中期目標 |

# After
### 2.3 対象範囲の位置づけ

本システムは、担当者がTeams Bot上で事務改定の内容をテキスト入力し、AI検索で影響候補を検出・判断するシステムである。
```

**Step 3: 2.4節の対象外範囲テーブルからStep 2行を削除・修正**

L100-107 を修正:
```markdown
# Before
| SharePoint PDF自動検知 | Step 2以降の機能（Event Grid連携） |
| PDF差分抽出 | Step 1では手入力のため不要 |
| レビューUI（Teams Tab） | 現段階ではAdaptive Cardでの候補表示に留める |
| 進捗管理・リマインド | 現段階では対象外 |
| FAQ/シナリオの本番マスタデータへの物理削除・修正の実行 | 本格実装フェーズ（PoCでの論理削除操作は対象） |
| カテゴリ別自動振り分け通知 | Step 2以降 |

# After
| PDF差分抽出 | 本システムでは手入力方式のため不要 |
| レビューUI（Teams Tab） | Adaptive Cardでの候補表示に留める |
| 進捗管理・リマインド | 対象外 |
| FAQ/シナリオの本番マスタデータへの物理削除・修正の実行 | 本格実装フェーズ（PoCでの論理削除操作は対象） |
```

> 削除行: L102（SharePoint PDF自動検知）、L107（カテゴリ別自動振り分け通知）
> 修正行: L103（「Step 1では」→「本システムでは」）、L104/105（「現段階では」→削除）

**Step 4: 2.5節見出しから「（Step 1）」を削除**

L109:
```markdown
# Before
### 2.5 システム概念図（Step 1）

# After
### 2.5 システム概念図
```

**Step 5: 3.2節見出しから「Step 1」を削除**

L180:（正確な行番号確認後）
```markdown
# Before
### 3.2 目標業務フロー（To-Be: Step 1）

# After
### 3.2 目標業務フロー（To-Be）
```

**Step 6: 4.1節の機能一覧テーブルからStep 2対象外行を削除**

L250-252 の3行を削除:
```markdown
# 削除
| FR-007 | PDF変更検知 | SharePoint上のPDF変更をEvent Gridで検知 | 対象外（Step 2） |
| FR-008 | PDF差分抽出 | 新旧PDFの差分をDocument Intelligence + LLMで抽出 | 対象外（Step 2） |
| FR-009 | カテゴリ別通知振り分け | 検索結果をカテゴリ別に自動振り分けてTeams通知 | 対象外（Step 2） |
```

> FR-010, FR-011 は「対象外（本格実装）」のため残置。

**Step 7: 6.1節見出しから「（Step 1）」を削除**

L711:
```markdown
# Before
### 6.1 アーキテクチャ概要（Step 1）

# After
### 6.1 アーキテクチャ概要
```

**Step 8: 付録Dの将来拡張候補テーブルからStep 2行を削除**

L1220-1230 を修正:
```markdown
# Before（7行のテーブル）
| PDF変更検知・差分抽出 | SharePoint + Event Grid + Document Intelligence | Step 2 |
| オート議事録連携 | 議事録から改定内容を自動抽出 | Step 2 |
| カテゴリ別通知振り分け | 影響候補をカテゴリ別にTeams通知 | Step 2 |
| SPOドキュメントのベクトル化 | SPO Indexer（プレビュー）で通達PDF等を直接ベクトル化 | Step 2 |
| レビューUI（Teams Tab） | 判定入力・進捗管理画面 | 本格実装 |
| シナリオ修正の実行 | Bot + Tab ハイブリッドでのシナリオ修正操作 | 本格実装 |
| Agentic Retrieval | 複雑なクエリの自動分解・統合 | GA後に検討 |

# After（3行のテーブル）
| レビューUI（Teams Tab） | 判定入力・進捗管理画面 | 本格実装 |
| シナリオ修正の実行 | Bot + Tab ハイブリッドでのシナリオ修正操作 | 本格実装 |
| Agentic Retrieval | 複雑なクエリの自動分解・統合 | GA後に検討 |
```

**Step 9: 全文を通読し、残存する「Step 1」「Step 2」表現を検索・修正**

`grep -n "Step [12]" docs/要件定義書.md` で残存箇所を確認。
1.2節のテーブル（L39「本システム Step 1の要件定義」）は削除 or 「本システムの要件定義」に修正。

**Step 10: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書からStep 2/オート議事録関連の記述を全削除（13箇所）"
```

---

## Task 2: ASCII図とUI仕様のコード実装整合（3箇所）

**Files:**
- Modify: `docs/要件定義書.md` — L388-407（検索カード図）、L410-442（結果カード図）、L375-386（表示仕様テキスト）

**Step 1: 検索カードASCII図のカテゴリ表示を修正**

L388-407 の検索カードASCII図を修正:

```markdown
# Before
│  シナリオカテゴリ                                                │
│  ☑ スマイル  ☑ 相続  ☑ 内部事務  ☑ 取引時確認                  │
│                                                                 │
│  各分野の表示件数: [30 件 ▼]                                     │

# After
│  シナリオカテゴリ: [スマイル         ▼]  ← 単一選択ドロップダウン │
│                                                                 │
│  表示件数: [30 件 ▼]                                             │
```

**Step 2: 結果カードASCII図をシナリオのみ表示に修正**

L410-442 の結果カードASCII図を修正。テキスト L381「選択したタブのみを表示対象」に合わせ、FAQセクション（❸）を削除:

```markdown
# Before（シナリオ + FAQ 混在）
│  ▼ シナリオ                                                     │
│  ❶ 預金 │ スコア: 3.42                                         │
│     ...                                                         │
│  ❷ 為替 │ スコア: 2.87                                         │
│     ...                                                         │
│                                                                 │
│  ▼ FAQ                                                          │
│  ❸ 預金 │ スコア: 2.65                                         │
│     ...                                                         │
│                                                                 │
│  [💾 要修正を保存]  [🗑️ 選択したFAQを削除]                      │

# After（シナリオのみの例）
│  ❶ スマイル │ 関連度: 3.4200                                    │
│     口座開設手続きフロー                                         │
│     「...本人確認書類を2点ご用意ください...」                    │
│     ☐ 要修正                                                   │
│                                                                 │
│  ❷ 相続 │ 関連度: 2.8700                                       │
│     相続手続きフロー                                             │
│     「...本人確認書類を提示...」                                 │
│     ☐ 要修正                                                   │
│                                                                 │
│  [💾 要修正を保存]                                               │
│  [← 前ページへ]  [次ページへ →]                                 │
```

> 注: 実装では `s.score.toFixed(4)` で4桁表示。カテゴリ名は `s.categoryName` を使用。

**Step 3: FR-005テキスト仕様の動的調整ロジック説明を補足**

L375-386 の結果カード表示仕様に補足:

```markdown
# Before
- **ページネーション方式**: 最大100件/ページで表示し、カードサイズ25KBを超過する場合は二分探索アルゴリズムで1ページあたりの表示件数を動的調整する。

# After
- **ページネーション方式**: ユーザーが選択した表示件数（10〜100件）を初期値とし、カードサイズ25KB（UTF-8バイト計測）を超過する場合はBotが二分探索アルゴリズムで1ページあたりの表示件数を動的に削減する。そのため、実際の表示件数はユーザー選択値以下になる場合がある。
```

**Step 4: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書のASCII図をコード実装に整合（カテゴリ選択/結果カード）"
```

---

## Task 3: CLAUDE.md の「検索UIの設計原則」を更新

**Files:**
- Modify: `CLAUDE.md` — L102-108（検索UIの設計原則）

**Step 1: 設計原則セクションを現行実装に合わせて修正**

```markdown
# Before
### 検索UIの設計原則
- シナリオとFAQは「同じデータカテゴリ」として統一的に扱う
- 7カテゴリ（シナリオ4 + FAQ3）をグループ表示し、1回の検索で選択カテゴリを横断検索する
- 検索ボタンは1セット（ハイブリッド検索 + キーワード一致検索）のみ。セクション別に分離しない
- 検索結果は全タイプをスコア順にマージし、統一ページネーション（1系統）で表示
- 同一ページにシナリオとFAQが混在する場合はタイプ別セクションで表示
- アクションボタン（「要修正を保存」「選択したFAQを削除」）は結果内容に応じて両方表示

# After
### 検索UIの設計原則
- シナリオタブとFAQタブを `Action.ToggleVisibility` で排他切替（初期表示はシナリオタブ）
- 各タブ内にカテゴリ単一選択（ドロップダウン）＋ 表示件数選択 ＋ 検索ボタン2つ（意味検索/キーワード検索）を配置
- 検索は `targetType`（`scenario` / `faq`）別に実行し、結果カードは選択タブのデータのみ表示
- 検索結果はスコア順にページネーション表示（二分探索で25KB上限に収まるよう動的調整）
- アクションボタン（「要修正を保存」「選択したFAQを削除」）は結果内容に応じて条件表示
```

**Step 2: コミット**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md 検索UIの設計原則を現行実装に更新"
```

---

## Task 4: セキュリティ・非機能要件の矛盾解消 + WEB調査反映

**Files:**
- Modify: `docs/要件定義書.md` — 10.1節、5.1節、11.2節、8.2節

**Step 1: Key Vault矛盾の解消（10.1節 L1056）**

```markdown
# Before
| APIキー管理 | Key Vault で一元管理 |

# After
| APIキー管理 | PoC段階ではManaged Identity認証を使用しKey Vaultは未使用。本番移行時にKey Vault導入を検討 |
```

**Step 2: 8.2節の認証記述を修正（L972）**

```markdown
# Before
| 認証 | APIキー（Key Vault経由）またはManaged Identity |

# After
| 認証 | Managed Identity（PoC段階）。本番移行時はAPIキー（Key Vault経由）も検討 |
```

**Step 3: NFR-001の備考にAction.Executeタイムアウト注記を追加（L678）**

```markdown
# Before
| NFR-001 | 影響検索応答時間 | 10秒以内 | 1回の検索 |

# After
| NFR-001 | 影響検索応答時間 | 10秒以内 | Teams Action.Executeのタイムアウトは10〜15秒（チャネル依存）。超過時は「処理中」カードを即時返却し、バックグラウンド処理後にproactiveActivityで結果通知する構成を採用 |
```

**Step 4: NFR-004の備考にSPO遅延考慮を追加（L681）**

```markdown
# Before
| NFR-004 | Excel出力応答時間 | 5秒以内 | 1回の検索結果として出力するシナリオ数百件規模。インメモリ生成 + SPOアップロード |

# After
| NFR-004 | Excel出力応答時間 | 5秒以内 | インメモリ生成 + SPOアップロード。SPOアップロード遅延時もAction.Executeタイムアウト前に処理中カードを返却し、非同期でアップロード完了後に結果通知 |
```

**Step 5: Indexer最短間隔の「Basic SKU」限定を修正（L1094）**

```markdown
# Before
| Indexer実行間隔 | 最短5分（Azure AI Search Basic SKUの制約）。現在の設定: 10分（`PT10M`） |

# After
| Indexer実行間隔 | 最短5分（全SKU共通の制約）。現在の設定: 10分（`PT10M`） |
```

**Step 6: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書のセキュリティ/非機能要件矛盾を解消 + WEB調査反映"
```

---

## Task 5: データ件数の統一 + コスト試算更新

**Files:**
- Modify: `docs/要件定義書.md` — L68, L92, L233-234, L694-695, L807, L839-842

**Step 1: 件数を検証環境最終値に統一**

修正対象箇所と修正内容:

| 箇所 | 行 | Before | After |
|------|-----|--------|-------|
| 2.1節 背景 | L68 | 「18,734件のFAQ」 | 「18,744件のFAQ」 |
| 2.4節 対象範囲 | L92 | 「検証環境実測: 21,047件」 | 「検証環境実測: 21,062件」 |
| 3.4節 シナリオ件数 | L233 | 「検証環境実測: 2,313件」 | 「検証環境実測: 2,318件」 |
| 3.4節 FAQ件数 | L234 | 「検証環境実測: 18,734件」 | 「検証環境実測: 18,744件」 |
| 5.3節 シナリオ現状 | L694 | 「2,313件（検証環境実測）」 | 「2,318件（検証環境実測）」 |
| 5.3節 FAQ現状 | L695 | 「18,734件（検証環境実測）」 | 「18,744件（検証環境実測）」 |
| 7.1節 シナリオ | L839 | 「検証環境実測: 2,313件」 | 「検証環境実測: 2,318件」 |
| 7.1節 FAQ | L840 | 「検証環境実測: 18,734件」 | 「検証環境実測: 18,744件」 |
| 7.1節 検索インデックス | L842 | 「検証環境実測: 21,047件」 | 「検証環境実測: 21,062件」 |

**Step 2: Embeddingコスト試算を実測値で更新（L807）**

```markdown
# Before
text-embedding-3-large: $0.13/1M tokens。FAQ 3,000件＋シナリオ500件 × 平均500トークン = 1.75Mトークン → 初回ベクトル化コスト約$0.23（約¥35）。月次差分更新は数円レベル。

# After
text-embedding-3-large: $0.13/1M tokens。検証環境実測: FAQ 18,744件＋シナリオ2,318件 × 平均500トークン ≈ 10.5Mトークン → 初回ベクトル化コスト約$1.37（約¥210）。月次差分更新は数十円レベル。
```

**Step 3: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書のデータ件数を検証環境最終値に統一 + コスト試算更新"
```

---

## Task 6: 用語の統一・定義追加

**Files:**
- Modify: `docs/要件定義書.md` — L532, L1138（コンテナ統一）、L861（sousoku注記強化）、12節（用語定義追加）

**Step 1: 「コレクション」→「コンテナ」に統一**

```markdown
# L532
# Before
Cosmos DB `impactAssessments`コレクションに判定レコードを作成

# After
Cosmos DB `impactAssessments`コンテナに判定レコードを作成

# L1138
# Before
impactAssessments | シナリオの要修正フラグを記録するCosmos DBコレクション。

# After
impactAssessments | シナリオの要修正フラグを記録するCosmos DBコンテナ。
```

**Step 2: sousoku/souzoku の注記強化（L861）**

```markdown
# Before
| FAQ | `sousoku` | 総則 | 総則規定（`souzoku`とは別） |

# After
| FAQ | `sousoku` | 総則 | 総則規定一般（読み: そうそく）。相続業務の `souzoku`（そうぞく）とは異なるカテゴリ |
```

**Step 3: 12節（用語定義）に不足用語を追加**

L1145（`差分ベクトル化` の行の後）に以下を追加:

```markdown
| Cosmos DBコンテナ | Azure Cosmos DB for NoSQLのデータベース内の論理的分割単位。本書では「コンテナ」に統一 |
| Skillset | Azure AI Searchのインデクシングパイプライン。複数のSkillを組み合わせてデータ変換フローを定義 |
| Single-Tenant | Botアプリ登録の構成。特定のMicrosoft Entra IDテナント内でのみ使用可能。Multi-Tenant構成は2025年7月31日以降非推奨 |
| sousoku（総則） | FAQのカテゴリID。総則規定に基づく一般的なQ&Aを格納。相続業務の `souzoku` とは異なる |
```

**Step 4: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書の用語統一（コンテナ/コレクション）+ 定義追加4件"
```

---

## Task 7: 検証計画の補強 + 版番号更新

**Files:**
- Modify: `docs/要件定義書.md` — 付録B（検証計画）、ヘッダー（版番号）

**Step 1: 付録B（検証計画）に検証項目2件を追加**

L1187（10d の後）に追加:

```markdown
| 11 | Managed Identity RBAC | AI Search・Cosmos DB・Azure OpenAIへのMI認証アクセスが正常に機能 | 各リソースへの操作（検索・読み書き・Embedding）がMIで認証される。権限不足時はApplication Insightsに記録される |
| 12 | Adaptive Card互換性 | Teamsデスクトップ・Web・Mobileでの表示・操作が正常 | ページネーション、ToggleVisibility、Action.Executeが各クライアントで動作確認 |
```

**Step 2: 版番号をv1.1に更新（L3）**

```markdown
# Before
**版数**: 1.0

# After
**版数**: 1.1
```

**Step 3: 作成日を更新（L4）**

```markdown
# Before
**作成日**: 2026年3月3日

# After
**作成日**: 2026年3月15日
```

**Step 4: 全文通読による最終整合性チェック**

以下を確認:
- [ ] 「Step 1」「Step 2」が不要な箇所に残っていないか（`grep -n "Step [12]"` で確認）
- [ ] 「オート議事録」「Event Grid」「PDF差分」が残っていないか
- [ ] 「コレクション」が残っていないか（`grep -n "コレクション"` で確認）
- [ ] セクション番号の連番が正しいか
- [ ] 目次（L9-23）が変更後のセクション名と一致しているか
- [ ] 削除したFR-007〜009が他セクションで参照されていないか

**Step 5: コミット**

```bash
git add docs/要件定義書.md
git commit -m "docs: 要件定義書 v1.1 — レビュー指摘16件修正（検証計画追加 + 版上げ）"
```

---

## 実行順序と依存関係

```
Task 1 (Step2削除)  ─┐
Task 2 (ASCII図修正)  ├── 独立して実行可能（並列可）
Task 3 (CLAUDE.md)    │
Task 4 (セキュリティ) ─┘
                      ↓
Task 5 (件数統一)     ─┐
Task 6 (用語統一)      ├── Task 1-4 完了後（行番号がずれるため）
Task 7 (検証計画+版上げ)┘
```

> **注意**: Task 1 の行削除で以降の行番号がずれる。Task 5-7 は Task 1-4 完了後に行番号を再確認して実施すること。

---
