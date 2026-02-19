# 検索精度分析: Phase 1 (ローカル) vs Phase 2 (Azure AI Search)

> 作成日: 2026-02-19
> 目的: Phase 2 の検索精度が Phase 1 より低い原因を網羅的に分析し、対策を整理する

---

## 目次

1. [両環境のアーキテクチャ比較](#1-両環境のアーキテクチャ比較)
2. [重み付けの仕組みの違い（9:1 問題）](#2-重み付けの仕組みの違い91-問題)
3. [preFilter / postFilter とは何か](#3-prefilter--postfilter-とは何か)
4. [Semantic Ranker は効果があるのか](#4-semantic-ranker-は効果があるのか)
5. [現在のインデックス構造（現状）](#5-現在のインデックス構造現状)
6. [精度低下の原因一覧](#6-精度低下の原因一覧)
7. [推奨対策（あるべき姿）](#7-推奨対策あるべき姿)

---

## 1. 両環境のアーキテクチャ比較

### 全体構成図

```
Phase 1 (ローカル)                          Phase 2 (Azure AI Search)
───────────────────                         ─────────────────────────
ChromaDB                                    Azure AI Search
├── smile_faq          (コレクション)        └── maintenance-search-index (1つ)
├── smile_scenario     (コレクション)             ├── scenarios (categoryId でフィルタ)
├── souzoku_faq        (コレクション)             └── faqs      (categoryId でフィルタ)
├── souzoku_scenario   (コレクション)
├── naibujimu_scenario (コレクション)        Cosmos DB
├── torikaku_scenario  (コレクション)        ├── scenarios コンテナ
├── yokin_faq          (コレクション)        ├── faqs コンテナ
└── ...                                     └── impactAssessments コンテナ
                                                    ↓ (Indexer × 2)
                                            AI Search Index (統合)
```

### 主要パラメータ比較

| 項目 | Phase 1 (ローカル) | Phase 2 (Azure) |
|------|-------------------|-----------------|
| **ベクトルDB** | ChromaDB (ローカル) | Azure AI Search (クラウド) |
| **データ分割** | カテゴリ × タイプ別コレクション | 1インデックス + categoryId フィルタ |
| **Embedding** | text-embedding-3-large (3072次元) | text-embedding-3-large (3072次元) |
| **HNSW m** | 16 (ChromaDB デフォルト) | **4** (Azure 最小値、上限10) |
| **検索方式** | ベクトル検索 + 独自キーワードマッチ | ハイブリッド検索 (BM25 + ベクトル + RRF) |
| **スコア統合** | スコアベース線形結合 | **順位ベース RRF** |
| **重み付け** | `0.9 × vector + 0.1 × keyword` | `weight=9.5` (RRF乗数) |
| **Reranking** | なし | Semantic Ranker あり（**現在無効**） |
| **テキスト検索** | Sudachi + Jaccard 類似度 | BM25 (ja.microsoft アナライザー) |
| **ベクトル化タイミング** | バッチスクリプト (直接API) | Indexer + Skillset (自動) |

---

## 2. 重み付けの仕組みの違い（9:1 問題）

### Phase 1: スコアベース線形結合

```
final_score = 0.9 × cosine_similarity + 0.1 × keyword_similarity
              ↑ vector_weight             ↑ keyword_weight (= 1.0 - 0.9)
```

- `cosine_similarity`: ChromaDB から返されるベクトル類似度 (0.0〜1.0)
- `keyword_similarity`: Sudachi 形態素解析 → 上位5キーワード → Jaccard 係数 (0.0〜1.0)
- 両方とも **0〜1 に正規化されたスコア** → 直接加算可能
- 設定箇所: `config/settings.yaml:27` の `vector_weight: 0.9`

**具体例:**
```
文書A: vector=0.95, keyword=0.80 → 0.9×0.95 + 0.1×0.80 = 0.935
文書B: vector=0.90, keyword=0.50 → 0.9×0.90 + 0.1×0.50 = 0.860
文書C: vector=0.85, keyword=0.95 → 0.9×0.85 + 0.1×0.95 = 0.860
  → 結果: A > B = C
```

### Phase 2: 順位ベース RRF (Reciprocal Rank Fusion)

Azure AI Search のハイブリッド検索は **RRF** でスコアを統合する。

```
RRF_score(文書) = Σ weight_i × 1/(rank_i + 60)
```

- `rank_i`: その文書の結果セット i 内での順位 (1, 2, 3, ...)
- `60`: Azure AI Search の固定パラメータ (k=60)
- `weight_i`: ベクトルクエリの weight パラメータ（デフォルト 1.0）

**現在の設定: weight=9.5**

```
RRF_score = 9.5 × 1/(vector_rank + 60) + 1.0 × 1/(BM25_rank + 60)
```

**具体例:**
```
文書A: vector 1位, BM25 10位
  → 9.5 × 1/(1+60) + 1.0 × 1/(10+60) = 9.5×0.01639 + 1.0×0.01429 = 0.1557 + 0.0143 = 0.170

文書B: vector 5位, BM25 1位
  → 9.5 × 1/(5+60) + 1.0 × 1/(1+60) = 9.5×0.01538 + 1.0×0.01639 = 0.1461 + 0.0164 = 0.163

文書C: vector 3位, BM25 3位
  → 9.5 × 1/(3+60) + 1.0 × 1/(3+60) = 9.5×0.01587 + 1.0×0.01587 = 0.1508 + 0.0159 = 0.167

  → 結果: A > C > B  (ベクトル検索結果がほぼ支配的)
```

### 核心的な違い

| 特性 | Phase 1 (Score-based) | Phase 2 (Rank-based RRF) |
|------|----------------------|--------------------------|
| **入力** | 生スコア (cosine=0.95 など) | 順位 (1位, 2位, 3位...) |
| **スコア差の保存** | あり (0.95 vs 0.90 = 0.05差) | **なし** (1位 vs 2位 のみ) |
| **正規化の必要性** | 両方0〜1なので不要 | 順位なので自動的に正規化 |
| **重みの意味** | スコア空間での割合 | RRF 逆順位スコアの乗数 |

**結論: ローカルの `9:1` をそのまま Azure の `weight=9.5` に変換しても同じ効果にはならない。**

ただし「ベクトル検索を強く重視する」という意図は同じ方向性。Azure でベクトル重視にしたい場合、weight=9.5 は妥当な方向性。問題は BM25 側の寄与がほぼゼロになる点で、日本語の専門用語（「カードローン少額払い」など）はキーワード一致が有効なケースがある。

**推奨**: weight=3.0〜5.0 程度で実験し、ベクトル重視を維持しつつ BM25 の寄与も残す。

---

## 3. preFilter / postFilter とは何か

### Azure AI Search のベクトル検索フィルタの仕組み

Azure AI Search では、ベクトル検索にフィルタ（例: `categoryId eq 'smile'`）を適用する方法が3種類ある。

### preFilter（現在のデフォルト）

```
[全21,000件のHNSWグラフ]
        │
        ▼ HNSW グラフを走査しながらフィルタを同時適用
        │  → categoryId='smile' に合致するノードだけを候補として探索
        │  → k件の候補が見つかるまでグラフを展開し続ける
        │
        ▼ 結果: フィルタ済みの k件（確実に k件返る）
```

**メリット**: 必ず k件のフィルタ済み結果が返る（再現率が高い）
**デメリット**: フィルタ後の候補が少ないと（例: torikaku=105/21,000件 = 0.5%）、
                HNSWグラフの大部分を走査する必要があり **遅くなる**。
                さらに m=4 の疎なグラフでは、フィルタ後に到達できない
                孤立ノードが発生し、**精度が低下する**。

### postFilter

```
[全21,000件のHNSWグラフ]
        │
        ▼ Step 1: フィルタなしで HNSW 探索 → 上位 k件を取得
        │          （全カテゴリ混在の k件）
        │
        ▼ Step 2: k件に対してフィルタを適用
        │          → categoryId='smile' に合致しない文書を除外
        │
        ▼ 結果: k件以下（フィルタで除外された分だけ減る）
```

**メリット**: HNSW 探索は高速（フィルタの選択性に影響されない）
**デメリット**: 上位 k件にフィルタ対象カテゴリの文書が少ないと、
                結果が k件未満になる（偽陰性リスク）

### 現在の問題

現在のコード (`agent.ts:691`) では `filter` をトップレベルで指定し、
`vectorFilterMode` は未指定 → **preFilter がデフォルト適用**。

```typescript
// 現在のコード
const options = {
  filter: `isDeleted eq false and dataType eq '${dataType}' and categoryId eq '${categoryId}'`,
  vectorSearchOptions: { queries: [...] },
  top: topN,
  // vectorFilterMode: 未指定 → preFilter
};
```

torikaku（105件/21,000件 = 0.5%）のような小カテゴリでは:
- preFilter で HNSW グラフを広範囲に走査
- m=4 の疎なグラフでは、フィルタ後に孤立ノードが発生
- 一部の関連文書が発見されない → **精度低下**

### 本プロジェクトでの最適解

**現在の「カテゴリ別に searchSingle を呼ぶ」設計では、preFilter の問題が顕著。**

対策の選択肢:

| 方式 | 説明 | 適用場面 |
|------|------|---------|
| postFilter + top 増加 | `vectorFilterMode: "postFilter"` + `top: topN*3` | 中〜大カテゴリ |
| exhaustive: true | HNSW を使わず全件探索 | 小カテゴリ（500件未満） |
| search.in() 一括検索 | 複数カテゴリを OR で一括検索 | 複数カテゴリ同時選択時 |

---

## 4. Semantic Ranker は効果があるのか

### Phase 1 で Reranking を使っていない

Phase 1 の検索パイプラインは:
1. ChromaDB でベクトル検索（top_k × 2 件）
2. Sudachi でキーワード抽出 → Jaccard 類似度計算
3. `0.9 × vector + 0.1 × keyword` で統合
4. **reranking は行っていない**

したがって「ローカルだと変化なかった」という体験は、Phase 1 に Semantic Ranker 相当の機能がなかったことと整合する。

### Azure Semantic Ranker の仕組み

```
L1 ランキング (BM25 + RRF)
  → 上位50件を取得
  → Microsoft 多言語 Transformer モデルで再スコアリング
  → @search.rerankerScore (0.0〜4.0) で並び替え
```

- Bing 検索から移植された多言語ディープラーニングモデル
- 日本語を含む複数言語で NDCG 改善が報告されている
- ただし L1 の結果が悪ければ、Semantic Ranker でも救えない

### 本プロジェクトでの効果見通し

| 状況 | Semantic Ranker の効果 |
|------|----------------------|
| L1（RRF）が良い精度で上位50件を返す | **効果あり** — 微妙な順位を改善 |
| L1 が悪い精度で関連文書が50位以内に入らない | **効果なし** — 入力が悪ければ出力も悪い |
| 現在の状態（m=4 + preFilter で精度低い） | **限定的** — まず L1 を改善すべき |

### 結論

**Semantic Ranker を有効化しても、Phase 1 との精度差の根本解決にはならない。**

Phase 1 で reranking なしに高精度だったのは:
1. カテゴリ別コレクション（フィルタ不要 → HNSW グラフが密）
2. m=16（十分な接続数）
3. スコアベース融合（微妙なスコア差を保存）

これらが Phase 2 で劣化している。Semantic Ranker は「ボーナス」であり、根本対策ではない。
ただし、L1 を改善した上で有効化すれば、さらなる精度向上が期待できる。

**推奨**: Phase 0 の他の修正（HNSW m 変更、weight 調整等）を先に実施し、
その後 Semantic Ranker を有効化して効果を実測する。

---

## 5. 現在のインデックス構造（現状）

### データフロー全体図

```
Cosmos DB                        AI Search                              Bot
─────────────                    ─────────                              ───

scenarios コンテナ               cosmos-scenarios-ds (DataSource)
 ├── smile (555件)          ──→  maintenance-scenarios-indexer
 ├── souzoku (269件)             ├── fieldMappings (13フィールド)
 ├── naibujimu (1,384件)         ├── outputFieldMappings (contentVector)
 └── torikaku (105件)            └── schedule: 1時間ごと
                                        │
                        maintenance-skillset ──→ Azure OpenAI
                        (combinedContent → 3072次元ベクトル)
                                        │
                                        ▼
                              maintenance-search-index          searchSingle()
                              (15フィールド、21,062件)    ←──  agent.ts
                                        ▲
                                        │
faqs コンテナ                cosmos-faqs-ds (DataSource)
 ├── smile (≈6,200件)       ──→  maintenance-faqs-indexer
 ├── sousoku (≈6,200件)          ├── fieldMappings (11フィールド)
 └── yokin (≈6,300件)            └── schedule: 1時間ごと
```

### インデックスフィールド構成

| # | フィールド | 型 | シナリオ | FAQ | 用途 |
|---|-----------|-----|:------:|:---:|------|
| 1 | id | String (PK) | ○ | ○ | `scenario-smile-0001`, `faq-yokin-04533` |
| 2 | dataType | String | `"scenario"` | `"faq"` | タイプ判別 |
| 3 | categoryId | String | ○ | ○ | `smile`, `souzoku`, `naibujimu`, `torikaku`, `yokin` |
| 4 | categoryName | String | ○ | ○ | 日本語名 |
| 5 | title | String (検索可) | ○ | ○ | ja.microsoft アナライザー |
| 6 | content | String (検索可) | ○ | ○ | ja.microsoft アナライザー |
| 7 | combinedContent | String (検索可) | ○ | ○ | `分類: ... \| 質問: ... \| 回答: ...` |
| 8 | **contentVector** | Collection(Single) | ○ | ○ | **3072次元、combinedContent から生成** |
| 9 | keywords | Collection(String) | ○ | ○ | 検索用キーワード配列 |
| 10 | updatedAt | DateTimeOffset | ○ | ○ | HWM 用 |
| 11 | isDeleted | Boolean | ○ | ○ | SoftDelete 用 |
| 12 | path | String | ○ | null | シナリオの階層パス |
| 13 | order | Int32 | ○ | null | 表示順 |
| 14 | isFinalAnswer | Boolean | ○ | null | 最終回答フラグ |
| 15 | tags | String | null | ○ | FAQ 用タグ |

### HNSW 設定

```json
{
  "m": 4,              // ← Azure 最小値（上限: 10、ChromaDB デフォルト: 16）
  "efConstruction": 400,
  "efSearch": 500,
  "metric": "cosine"
}
```

### 検索フロー

```
ユーザーが「相続 カードローン」で検索
    │
    ▼ extractCategorySelections()
    │  ユーザーが選択したカテゴリを抽出
    │  例: scenarios=[souzoku], faqs=[sousoku]
    │
    ▼ searchByCategories()
    │  選択カテゴリ分の searchSingle を並列実行
    │  ├── searchSingle("相続 カードローン", "semantic", "scenario", "souzoku", 30)
    │  └── searchSingle("相続 カードローン", "semantic", "faq",      "sousoku", 30)
    │
    │  各 searchSingle 内:
    │  ├── filter = "isDeleted eq false and dataType eq 'scenario' and categoryId eq 'souzoku'"
    │  ├── BM25: title, content, keywords を検索
    │  ├── ベクトル: "相続 カードローン" → Azure OpenAI で 3072次元化 → contentVector と比較
    │  ├── weight=9.5 の RRF でスコア統合
    │  └── preFilter（デフォルト）でフィルタ適用
    │
    ▼ deduplicateById + scoreSort
    │
    ▼ buildResultCard (ページネーション付き Adaptive Card)
```

---

## 6. 精度低下の原因一覧

### 確定要因（コード・設定で確認済み）

| # | 要因 | Phase 1 | Phase 2 | 影響度 | 確信度 |
|---|------|---------|---------|--------|--------|
| **1** | HNSW m | 16 (ChromaDB デフォルト) | **4** (Azure 最小値) | **大** | 確定 |
| **2** | データ分割 | カテゴリ別コレクション | 1インデックス + フィルタ | **大** | 確定 |
| **3** | スコア統合方式 | Score-based 線形結合 | Rank-based RRF | **中** | 確定 |
| **4** | フィルタ方式 | フィルタ不要（コレクション分離） | preFilter（HNSW断片化リスク） | **大** (小カテゴリ) | 確定 |
| **5** | テキスト検索 | Sudachi + Jaccard | BM25 (ja.microsoft) | **中** | 確定 |
| **6** | キーワード抽出 | Sudachi 形態素解析で上位5語 | なし（BM25 に依存） | **小** | 確定 |

### 設定ミスの可能性

| # | 要因 | 現状 | あるべき姿 | 影響度 |
|---|------|------|-----------|--------|
| **7** | Semantic Ranker | 無効（queryType="simple"） | 有効化可能だが Phase 1 にもないため比較には無関係 | — |
| **8** | weight 値 | 9.5（ローカル 9:1 の模倣） | RRF では別物、3.0〜5.0 が妥当か | **中** |

### アーキテクチャ上の制約

| # | 要因 | 説明 | 影響度 |
|---|------|------|--------|
| **9** | Azure m 上限 | m=10 が上限（ChromaDB の m=16 に届かない） | **中** |
| **10** | カテゴリ別検索のスコア非互換 | カテゴリ別に独立 RRF → スコア直接比較が不正確 | **中** |
| **11** | RRF のスコア差損失 | 順位のみ使用、cosine 0.95 vs 0.90 の差が消失 | **小〜中** |

---

## 7. 推奨対策（あるべき姿）

### Phase 0: コード修正のみ（即日実施可、インデックス変更なし）

#### 対策 0-1: vectorFilterMode の明示設定

**現状**: 未指定 → preFilter（小カテゴリで精度低下）
**対策**: 小カテゴリには exhaustive、それ以外は preFilter 維持

```typescript
// searchSingle() の修正案
const isSmallCategory = /* カテゴリ別の件数で判定 */;

const vectorQuery = {
  kind: "text" as const,
  text: query,
  fields: ["contentVector"],
  weight: 4.0,  // 9.5 → 4.0 に調整（後述）
  exhaustive: isSmallCategory,  // 500件未満は全件探索
};
```

#### 対策 0-2: weight 値の調整

**現状**: weight=9.5（RRF でベクトル側を 9.5 倍に増幅）
**問題**: BM25 の寄与がほぼゼロ。日本語専門用語のキーワードマッチが効かない。
**対策**: weight=3.0〜5.0 で実験

```
weight=1.0: ベクトルと BM25 が対等
weight=3.0: ベクトルが 3 倍優位（推奨開始値）
weight=5.0: ベクトルが 5 倍優位
weight=9.5: ベクトルがほぼ支配的（現状）
```

**注意**: ローカルの 9:1 は score-based だったが、Azure の RRF は rank-based。
同じ比率にしても同じ効果にはならない。実験的チューニングが必要。

#### 対策 0-3: 複数カテゴリ選択時の一括検索

**現状**: カテゴリ別に searchSingle → スコア直接比較でマージ
**問題**: 異なる RRF 計算のスコアを直接比較（不正確）
**対策**: 複数カテゴリ選択時は `search.in()` で一括検索

```typescript
// 複数カテゴリ一括検索
const filter = `isDeleted eq false and search.in(categoryId, '${categoryIds.join(",")}', ',')`;
// → Azure 内部で統一 RRF → スコアが比較可能
```

#### 対策 0-4: Semantic Ranker の有効化（オプション）

Phase 1 にはない機能なので、比較には無関係。ただし精度向上のボーナスとして有効化可能。

```typescript
// 有効化する場合
const options = {
  queryType: "semantic" as const,
  semanticSearchOptions: {
    configurationName: "semantic-config",
  },
  // ... 他のオプション
};
```

**効果**: L1 (RRF) の結果を Microsoft 多言語モデルで rerank。日本語でも効果報告あり。
**注意**: L1 の精度が低いと Semantic Ranker でも改善は限定的。まず L1 を改善すべき。

### Phase 1: インデックス再作成（ゼロダウンタイム切替可能）

#### 対策 1-1: HNSW m=10 に変更

**現状**: m=4（Azure 最小値）
**推奨**: m=10（Azure 上限）
**注意**: m=16（ChromaDB デフォルト）は Azure では設定不可

```json
{
  "hnswParameters": {
    "m": 10,
    "efConstruction": 400,
    "efSearch": 1000,
    "metric": "cosine"
  }
}
```

**手順（ゼロダウンタイム）**:
1. 新インデックス `maintenance-search-index-v2` を m=10 で作成
2. エイリアス `maintenance-search-index` を v1 に向ける
3. v2 に Indexer を向けて全件再インデックス
4. 完了後、エイリアスを v2 に切替（アトミック操作）
5. 確認後 v1 を削除

#### 対策 1-2: スカラー量子化の追加

50万件スケール対応の布石として、int8 スカラー量子化を追加。
精度劣化はほぼなし（Azure 公式: "minimal impact on recall"）でメモリ 4倍圧縮。

### Phase 2: 長期対策（必要に応じて）

#### 対策 2-1: 小カテゴリのインデックス分割

torikaku（105件）や souzoku（269件）のような小カテゴリは、
1インデックス + フィルタ方式では HNSW の構造的限界がある。

**Phase 1 のようにカテゴリ別インデックスに分割する選択肢:**
- メリット: フィルタ不要 → HNSW グラフが密 → 高精度
- デメリット: インデックス × カテゴリ数、管理コスト増
- Basic tier のインデックス数上限に注意

#### 対策 2-2: 次元削減

text-embedding-3-large は MRL（Matryoshka Representation Learning）対応。
3072 → 1536 次元に削減可能（API パラメータで指定）。

- メモリ半減、検索速度向上
- 精度影響は軽微（Azure 公式ベンチマークで確認済み）

---

## 付録: 現在の設定ファイル一覧

### Azure AI Search 設定 (`scripts/` 配下)

| ファイル | 内容 |
|---------|------|
| `index-definition.json` | 15フィールド定義 + HNSW + Semantic Config |
| `datasource-scenarios.json` | Cosmos DB scenarios コンテナ → DataSource |
| `datasource-faqs.json` | Cosmos DB faqs コンテナ → DataSource |
| `skillset.json` | AzureOpenAIEmbeddingSkill (combinedContent → 3072次元) |
| `indexer-scenarios.json` | scenarios Indexer (13 fieldMappings) |
| `indexer-faqs.json` | faqs Indexer (11 fieldMappings) |

### Bot 検索コード (`maintenance-bot/src/agent.ts`)

| 関数 | 行 | 役割 |
|------|-----|------|
| `searchSemantic` handler | 117-158 | 意味検索のエントリポイント（非同期 + プロアクティブ） |
| `searchKeyword` handler | 161-202 | キーワード検索のエントリポイント |
| `searchByCategories()` | 615-660 | カテゴリ別並列検索 + マージ |
| `searchSingle()` | 665-719 | 単一カテゴリ検索（ハイブリッド or キーワード） |
| `fetchAllScenariosForCategory()` | 725-750 | Excel出力用全量取得 |

---

## 付録: 用語集

| 用語 | 説明 |
|------|------|
| **HNSW** | Hierarchical Navigable Small World — 近似最近傍探索のグラフベースアルゴリズム |
| **m** | HNSW の各ノードからの双方向リンク数。大きいほど精度向上するがメモリ増加 |
| **efSearch** | HNSW 探索時の候補リストサイズ。大きいほど精度向上するが検索遅延増加 |
| **efConstruction** | HNSW グラフ構築時の候補リストサイズ。大きいほど構築品質向上 |
| **RRF** | Reciprocal Rank Fusion — 順位ベースのスコア統合。`score = 1/(rank + k)` |
| **BM25** | Best Matching 25 — TF-IDF ベースの全文検索スコアリング |
| **preFilter** | HNSW 探索中にフィルタを適用。再現率高いが小規模フィルタで遅い |
| **postFilter** | HNSW 探索後にフィルタを適用。高速だが結果が k 件未満になりうる |
| **Semantic Ranker** | Microsoft 多言語モデルによる L2 リランキング。L1 結果の上位50件を再評価 |
| **Score-based fusion** | 生スコアの線形結合。Phase 1 方式 |
| **Rank-based fusion** | 順位の逆数で統合。Azure RRF 方式 |
