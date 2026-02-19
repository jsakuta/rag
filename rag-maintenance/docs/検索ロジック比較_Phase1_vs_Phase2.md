# 検索ロジック比較: Phase 1 (rag-gemini) vs Phase 2 (maintenance-bot)

- 文書番号: COMP-SEARCH-001
- 最終更新: 2026-02-18（ベクトル重み制御 `weight` パラメータ調査 + RRF チューニング手法を追記）
- 目的: Phase 1→2 の検索精度差分を網羅的に記録し、Phase 2 改善の結果を反映する
- 前提: Phase 2 は「業務分野選択 + 表示件数選択 UI」を実装済み。Semantic Ranker は精度向上が確認されなかったため廃止

---

## 差分サマリー

| 観点 | Phase 1 (rag-gemini) | Phase 2 (maintenance-bot) | 評価 |
|------|---------------------|--------------------------|------|
| **ベクトル化** | combined_text / 3,072次元 | combinedContent / 3,072次元 | **同等**（同一フォーマット） |
| **テキスト検索** | Sudachi + Jaccard（1フィールド） | ja.microsoft + BM25（5フィールド横断） | **同等〜改善**（再現率向上） |
| **スコア統合** | 加重平均 (0.9v + 0.1k) → チューニング自在 | RRF (k=60固定) + **`weight` パラメータでベクトル重み制御可能** | **方式変更**（RRFは順位ベースで異スケールに強い。`weight` で Phase 1 相当の重み調整が可能） |
| **キーワード抽出** | Sudachi 品詞重み + 位置重み（クエリ主導） | BM25 IDF + keywords フィールド検索（ドキュメント主導） | **間接カバー**（銀行固有名詞の明示重み付けは消失） |
| **再ランキング** | oversampling + keyword boost（2段階） | BM25 + HNSW cosine の RRF 統合（2段階） | **方式変更**（Semantic Ranker 廃止、RRF のみ） |
| **スコア表示** | cosine 0〜1 + keyword 0〜1 → 加重平均 | RRF @search.score（順位ベースの統合スコア） | **方式変更** |
| **業務分野スコープ** | 分野別DB（自動分離・高精度） | filter で分野絞り込み（実装済み） | **同等** |
| **LLM クエリ拡張** | 3戦略あり | なし（精度向上未確認のため非採用） | **影響なし** |
| **結果件数** | 固定 top_k=4 | ユーザー選択 (10〜150、11段階) | **改善** |

> **総合:** Phase 2 は BM25 + ベクトル cosine の RRF 統合（2段階ランキング）。Semantic Ranker は PoC 評価で精度向上が確認されなかったため廃止。業務分野 filter + 件数選択 UI の実装により柔軟性を実現。さらに `weight` パラメータ（安定版 API `2025-09-01` で利用可能）でベクトル検索の重み制御が可能であり、Phase 1 の加重平均（0.9v + 0.1k）に近い挙動を再現できる。

---

## 1. アーキテクチャ全体像

### Phase 1 (rag-gemini)

```
ユーザー → Excel バッチ入力
  ↓
Python (Searcher)
  ├─ Sudachi 形態素解析 → キーワード抽出
  ├─ Embedding API (Gemini or Azure OpenAI) → クエリベクトル生成
  ├─ ChromaDB (ローカル) → ベクトル検索 (cosine)
  ├─ KeywordSearchEngine → Jaccard 類似度計算
  ├─ 加重スコア統合 (0.9×vector + 0.1×keyword)
  └─ Excel 出力 (top_k=4)
```

### Phase 2 (maintenance-bot) — 実装済み

```
ユーザー → Teams チャット入力
  ↓
モード選択カード: 業務分野選択 + 表示件数選択（10〜150件）
  ↓
Bot (TypeScript) → Azure AI Search API 1回コール
  ↓
AI Search 内部:
  ├─ BM25 テキスト検索 (ja.microsoft アナライザ)
  ├─ Vectorizer → クエリベクトル生成 (text-embedding-3-large)
  ├─ HNSW ベクトル検索 (cosine)
  └─ RRF (Reciprocal Rank Fusion) でスコア統合
  ↓
Adaptive Card (top N件, RRFスコア順, 10件/ページ)
```

> **注:** Semantic Ranker（3段階目の再ランキング）は PoC 評価で精度向上が確認されなかったため廃止。
> 上位50件制約が解除され、ユーザーが選択した件数（最大150件）をそのまま使用可能になった。

---

## 2. ベクトル化対象の比較

### 2-A. 何をベクトル化しているか

| 項目 | Phase 1 | Phase 2 | 差分 |
|------|---------|---------|------|
| 対象テキスト | `combined_text` | `combinedContent` | **同一フォーマット** |
| Embedding モデル | Gemini embedding-001 or Azure OpenAI text-embedding-3-large | Azure OpenAI text-embedding-3-large のみ | Phase 1 は選択可能 |
| 次元数 | 3,072 | 3,072 | 同一 |
| ベクトル化タイミング | Python スクリプト実行時（クライアント側） | AI Search Indexer 実行時（サーバー側） | Phase 2 は自動 |
| ベクトル DB | ChromaDB (ローカル HNSW) | AI Search (HNSW) | 同一アルゴリズム |

### 2-B. combinedContent のフォーマット

**シナリオ:**
```
"分類: {Lv1 > Lv2 > ... > Lv(n-2)} | 質問: {Lv(n-1)} | 回答: {Lv(n)}"
```

**FAQ:**
```
"質問: {question} | 回答: {answer}"
```

**Phase 1 と Phase 2 で同一フォーマット。** Phase 2 の `convert-excel-to-json.py` は Phase 1 の `TextCombiner._build_combined_text()` と同じ生成ルールを実装している。

### 2-C. combinedContent の構成要素決定ロジック

#### シナリオの右→左スキャン（Phase 1 / Phase 2 共通）

```
Excel行: [Lv1="預金関連", Lv2="普通預金", Lv3="新規口座開設", Lv4="本人確認", Lv5="運転免許証の場合"]

右→左スキャン:
  Lv5 "運転免許証の場合" → 最右非空 → answer (content)
  Lv4 "本人確認"         → answerの1つ左 → question
  Lv1〜Lv3              → questionより左 → hierarchy (分類)

結果:
  content = "運転免許証の場合"
  title   = "/預金関連/普通預金/新規口座開設/本人確認"
  combined = "分類: 預金関連 > 普通預金 > 新規口座開設 | 質問: 本人確認 | 回答: 運転免許証の場合"
```

#### FAQ の列マッピング

| ファイル | title (質問) | content (回答) | 補足 |
|---------|-------------|---------------|------|
| スマイル_履歴データ | `問い合わせ` 列 | `回答` 列 + `\n\n` + `補足回答` 列 | タグなし |
| 総則_履歴データ | `質問` 列 | `回答` 列 | `タグ付け` 列あり |
| 預金_履歴データ | `質問` 列 | `回答` 列 | `タグ付け` 列あり |

**差分:** Phase 1 は `dynamic_db_manager.py` で `\n` 区切り、Phase 2 は `convert-excel-to-json.py` で `\n\n` 区切り。実質的影響は軽微（Embedding の類似度にほぼ影響なし）。

---

## 3. テキスト検索の比較

### 3-A. トークン化

| 項目 | Phase 1 (Sudachi) | Phase 2 (ja.microsoft) |
|------|-------------------|----------------------|
| エンジン | Sudachi (SplitMode.C) | Microsoft 日本語アナライザ (MeCab ベース) |
| 辞書 | sudachi-dictionary-full | Microsoft 独自辞書 |
| 分かち書き精度 | 高（銀行用語に強い） | 高（一般用語に強い） |
| カスタム辞書 | Sudachi ユーザー辞書追加可能 | 不可（Azure 管理） |

**精度影響:** 両者とも日本語形態素解析としては十分な品質。Sudachi は「取引時確認」「預り物件」等の複合語を1トークンとして認識する精度がやや高いが、BM25 は部分一致でもスコアが付くため実用差は小さい。

### 3-B. スコア計算

| 項目 | Phase 1 (Jaccard) | Phase 2 (BM25) |
|------|-------------------|----------------|
| アルゴリズム | Jaccard 類似度（集合の共通率） | BM25 (TF-IDF 拡張 + 文書長正規化) |
| 計算式 | `weighted_intersection / union_size` | `Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1-b+b×dl/avgdl))` |
| 文書長正規化 | なし（集合ベースなので文書長に鈍感） | あり（b=0.75 で長い文書のスコアを抑制） |
| 値域 | 0.0〜1.0 | 0.0〜∞（上限なし、相対スコア） |
| 検索対象フィールド | combined_text（1フィールド） | title, content, combinedContent, categoryName, keywords（**5フィールド**） |

**精度影響:**
- BM25 は IDF（逆文書頻度）を考慮するため、「口座」のような高頻度語よりも「AMLフィルター」のような低頻度語を重視する。Jaccard にはこの概念がない
- Phase 2 は 5 フィールド横断検索なので、title に含まれなくても keywords や content で拾える。**再現率は Phase 2 の方が高い**
- BM25 の文書長正規化により、短い FAQ（"解決済み"）は長いシナリオに対して不当に高スコアにならない

### 3-C. キーワード抽出の詳細比較

#### Phase 1: Sudachi ベースの明示的キーワード抽出

```python
# keyword_search_engine.py
def extract_keywords(self, text, top_k=5):
    morphemes = tokenizer.tokenize(text, SplitMode.C)
    keywords = []
    for m in morphemes:
        if m.part_of_speech()[0] == '名詞':           # 名詞のみ
            important = ['固有名詞', '一般']
            weight = 2 if m.part_of_speech()[1] in important else 1  # 固有名詞・一般名詞は2倍
            word = m.dictionary_form()
            if len(word) > 1:                          # 1文字は除外
                keywords.extend([word] * weight)
    # ストップワード除去
    filtered = {w: c for w, c in Counter(keywords).items() if w not in STOP_WORDS}
    return [w for w, _ in Counter(filtered).most_common(top_k)]
```

**ストップワード:** `['こと', 'もの', 'これ', 'それ', 'ところ', '方', 'する', 'ある', 'いる', 'れる', 'られる', 'なる', 'その']`

**品詞フィルタ:**
- 名詞のみ抽出（動詞・形容詞・助詞等は除外）
- 固有名詞・一般名詞: 重み 2
- その他名詞（数詞、接尾辞等）: 重み 1

**位置重み付け:**
```python
# keyword_search_engine.py (Line 120-124)
weighted_score = sum(
    position_weight if reference_text.find(kw) < len(reference_text) // 2 else 1.0
    for kw in intersection
)
# position_weight = 1.2（テキスト前半に出現するキーワードを1.2倍）
```

#### Phase 2: 明示的なキーワード抽出なし

- **クエリ側:** ユーザー入力をそのまま AI Search に送信。トークン化は ja.microsoft が内部で実行
- **ドキュメント側:** `keywords` フィールドに事前格納済み（シナリオ: 階層テキスト、FAQ: タグから抽出）
- **マッチング:** BM25 が keywords フィールドを searchable として検索（キーワード検索時は searchFields に明示指定）

**Phase 2 の keywords フィールドの中身:**

| データ種別 | keywords の決定ロジック | 例 |
|-----------|----------------------|-----|
| シナリオ | 階層パスの各レベルテキスト | `["預金関連", "普通預金", "新規口座開設", "本人確認"]` |
| FAQ (タグあり) | タグ文字列から `Lv\d+:\s*` でパース | `["営業事務", "預り物件管理システム"]` |
| FAQ (タグなし) | 空配列 | `[]` |

**精度影響:**
- Phase 1 は **クエリ側** からキーワードを抽出して Jaccard で比較（クエリ主導）
- Phase 2 は **ドキュメント側** に事前格納したキーワードを BM25 で検索（ドキュメント主導）
- Phase 1 の品詞フィルタ（名詞のみ）と位置重み付けは Phase 2 にはない → **銀行固有名詞の重み付けが弱化**
- BM25 の IDF 重み付けが部分的に補完するが、銀行固有名詞に対する明示的な重み付けは Phase 1 の方が優れている

---

## 4. スコア統合の比較

### 4-A. Phase 1: 明示的加重平均

```
combined_score = 0.9 × vector_similarity + 0.1 × keyword_similarity
```

| パラメータ | デフォルト値 | 調整可能 |
|-----------|------------|---------|
| vector_weight | 0.9 | settings.yaml で変更可 |
| keyword_weight | 0.1 (= 1.0 - vector_weight) | 自動計算 |
| VECTOR_SEARCH_MULTIPLIER | 2 | oversampling 倍率 |
| top_k | 4 | 最終結果件数 |

**特徴:**
- スコア内訳（vector_sim, keyword_sim）が個別に見える
- weight を変えて A/B テスト可能
- 値域: 0.0〜1.0（コサイン類似度ベース）

### 4-B. Phase 2: RRF + ベクトル重み制御

**RRF (Reciprocal Rank Fusion) の基本式** [^1]

```
RRF_score(doc) = 1/(k + rank_BM25) + weight × 1/(k + rank_vector)
k = 60（RRF アルゴリズム内の定数、変更不可。ベクトルクエリの k とは別）
weight = ベクトルクエリの weight パラメータ（デフォルト 1.0）
```

- BM25 での順位と HNSW ベクトル検索での順位を **順位ベース** で統合
- スコア値の絶対値ではなく相対順位を使うため、スケールの異なるスコアを安全に統合できる
- AI Search の `@search.score` として返却される
- Bot では `result.score` として取得し、小数点以下4桁で表示
- RRF スコアは 0.03 程度でも高い関連性を示す場合がある（順位ベースのため絶対値は小さい）[^2]

**Semantic Ranker 廃止の経緯:**

| 項目 | 内容 |
|------|------|
| 廃止理由 | PoC 評価で rerankerScore による精度向上が確認されなかった |
| 上位50件制約 | Semantic Ranker は RRF 上位50件のみ再ランキング（固定制約）→ 大量取得時にスコア混在の問題 |
| 廃止による利点 | top の50件制約が解除され、ユーザー選択値（最大150件）をそのまま使用可能 |
| 変更内容 | `queryType: "semantic"` → `queryType: "simple"` + `vectorSearchOptions`（ハイブリッド検索維持） |

### 4-C. Phase 2 のランキング構成（現行）

Phase 2 は **2段階のランキング** を行っている:

1. BM25（テキスト一致）← ja.microsoft アナライザ
2. HNSW cosine（ベクトル類似度）← Phase 1 と同等

上記2つの結果を RRF で順位ベース統合 → `@search.score` として返却

### 4-D. Azure AI Search で利用可能な重み制御パラメータ（2026-02-18 調査）

Phase 1 の加重平均（`0.9v + 0.1k`）に近い挙動を RRF 上で再現するために、Azure AI Search は以下の3つのチューニングパラメータを提供している。

#### (1) `weight` パラメータ — ベクトルクエリの重み乗数

`vectorQueries` の各クエリに設定可能。RRF サブスコアに対する乗数として機能する。[^3]

| 項目 | 内容 |
|------|------|
| パラメータ位置 | `vectorQueries[].weight` |
| 型 | `number`（正の数、0より大きい値） |
| デフォルト | 1.0（重み付けなし） |
| API バージョン | **`2025-09-01`（安定版）** 以降 |
| JS SDK | `@azure/search-documents` v12.2.0 以降の `VectorizableTextQuery.weight` [^4] |

**計算式:**

```
重み付き RRF_score(doc) = 1/(60 + rank_BM25) + weight × 1/(60 + rank_vector)
```

**weight 値によるベクトル貢献比率**（あるドキュメントが BM25/ベクトル両方で同一順位にある場合の理論値）:

| weight | ベクトル貢献比率 | Phase 1 近似 | 備考 |
|--------|-----------------|-------------|------|
| 1.0 | 50% | 0.5:0.5 | デフォルト（現行） |
| 2.0 | 67% | 0.67:0.33 | — |
| 3.0 | 75% | 0.75:0.25 | — |
| 4.0 | 80% | 0.8:0.2 | Phase 1 の 0.8 相当 |
| **5.0** | **83%** | **0.83:0.17** | **推奨初期値** |
| 7.0 | 88% | 0.88:0.12 | — |
| **9.0** | **90%** | **0.9:0.1** | **Phase 1 の 0.9 に最も近い** |

> **注意:** RRF は順位ベース、Phase 1 は実スコアベースのため、貢献比率は厳密には同一ではない。
> ベクトル検索で上位・BM25 で下位のドキュメントは weight 増加で大幅にブーストされる。
> 逆にベクトル検索で下位のドキュメントは weight を上げても順位改善の効果が限定的。

**TypeScript 実装例:**

```typescript
// agent.ts — searchSingle() の semantic モード
vectorSearchOptions: {
  queries: [{
    kind: "text" as const,
    text: query,
    fields: ["contentVector"],
    weight: 5.0,  // ← ベクトルの重みを強化（デフォルト1.0）
  }],
},
```

#### (2) `maxTextRecallSize` — BM25 側の結果プールサイズ制限

RRF に渡す BM25 側の結果数を制御する。小さくするとキーワード側の影響力が低下する。[^5]

| 項目 | 内容 |
|------|------|
| パラメータ位置 | `hybridSearch.maxTextRecallSize` |
| デフォルト | 1,000 |
| 最大値 | 10,000 |
| API バージョン | **`2024-05-01-preview` 以降（プレビュー版のみ）** |
| 関連パラメータ | `hybridSearch.countAndFacetMode`（`"countAllResults"` or `"countRetrievableResults"`） |

**使い分け:**

| シナリオ | 推奨値 | 理由 |
|---------|--------|------|
| ベクトル検索が優勢 | 50〜200 に縮小 | BM25 のノイズを抑制 |
| キーワード一致が重要 | 2,000〜5,000 に拡大 | 長尾のキーワードマッチを拾う |
| デフォルト運用 | 1,000（変更不要） | ほとんどのケースで十分 |

**REST API 例:**

```json
POST .../docs/search?api-version=2025-11-01-preview
{
  "search": "<クエリ>",
  "hybridSearch": {
    "maxTextRecallSize": 100,
    "countAndFacetMode": "countRetrievableResults"
  },
  "vectorQueries": [{
    "kind": "text",
    "text": "<クエリ>",
    "fields": "contentVector",
    "weight": 5.0
  }],
  "top": 30
}
```

> **注:** 現時点の JS SDK 安定版（v12.2.0）では `hybridSearch` パラメータは未サポート。
> REST API で直接呼び出すか、SDK のプレビュー版を使用する必要がある。

#### (3) `threshold` — 低スコア結果の除外（プレビュー）

ベクトル検索結果から低スコアのマッチを RRF 統合前に除外する。[^6]

```typescript
// 2024-05-01-preview 以降
vectorQueries: [{
  kind: "text",
  text: query,
  fields: ["contentVector"],
  threshold: { kind: "vectorSimilarity", value: 0.75 },
}],
```

k=30 でも類似度 0.75 未満は除外されるため、ノイズの少ない結果セットを RRF に渡せる。

### 4-E. Phase 1 との方式差異の本質

| 観点 | Phase 1（加重平均） | Phase 2（RRF + weight） |
|------|---------------------|------------------------|
| **統合対象** | スコアの **絶対値**（cosine 0〜1, Jaccard 0〜1） | **順位**（rank 1, 2, 3, ...） |
| **重み付けの効果** | スコアの大きさに比例して線形に影響 | 順位のRRFサブスコアに乗数として影響 |
| **利点** | 直感的。重みの意味が明確 | スケール不一致に強い。異なるスコア体系を安全に統合 |
| **欠点** | スコア体系が異なると重みの意味が変わる | 順位の差が大きいと weight の効果が減衰 |
| **チューニング** | `settings.yaml` で `vector_weight` を変更 | `vectorQueries[].weight` を変更 |

**具体例 — weight=5.0 の場合:**

あるドキュメントがベクトル検索で1位、BM25 で50位のとき:
```
RRF = 1/(60+50) + 5.0 × 1/(60+1) = 0.0091 + 0.0820 = 0.0911
→ ベクトル貢献 90%（Phase 1 の 0.9 × vector_sim と近い効果）
```

逆にベクトル検索で50位、BM25 で1位のとき:
```
RRF = 1/(60+1) + 5.0 × 1/(60+50) = 0.0164 + 0.0455 = 0.0619
→ ベクトル貢献 73%（順位が低いと weight の効果が減衰する）
```

---

## 5. 検索スコープの比較

### 5-A. 業務分野の扱い

| 項目 | Phase 1 | Phase 2 (現行) | Phase 2 (改善後) |
|------|---------|---------------|-----------------|
| DB/インデックス構成 | 業務分野ごとに ChromaDB コレクション分離 | 全分野が 1 インデックスに同居 | 同左（filter で分離） |
| 分野選択 | ファイル名から自動判定 | なし（全分野横断検索） | **ユーザーが UI で選択** |
| 精度影響 | 同一分野内のみ比較 → **高精度** | 全 21,000 件から検索 → ノイズ混入リスク | filter で分野絞り込み → Phase 1 相当 |

**全分野統合の問題点:**
- 「口座開設」で検索すると、スマイル・預金・総則の全カテゴリから結果が返る
- 異なる業務分野の文書が混在し、ユーザーが求める分野の結果が埋もれる
- Phase 1 は分野別 DB なのでこの問題がない

**改善策:** `filter: "categoryId eq 'smile'"` で Phase 1 と同等の分野絞り込みを実現

### 5-B. 結果件数

| 項目 | Phase 1 | Phase 2 (現行) | Phase 2 (改善後) |
|------|---------|---------------|-----------------|
| 件数 | top_k=4（固定） | — | **ユーザーが選択**（10〜150、11段階） |
| oversampling | 2×top_k=8 件取得 → 4件に絞り込み | — | AI Search 内部で自動 |

---

## 6. LLM クエリ拡張の比較

| 項目 | Phase 1 | Phase 2 |
|------|---------|---------|
| クエリ拡張 | あり（3戦略: Original / LLMEnhanced / MultiStage） | **なし** |
| 拡張方法 | Gemini/Azure OpenAI で要約・言い換え | — |
| MultiStage OR マージ | Original結果 ∪ LLM結果 → 3分類（Both/Original_Only/LLM_Only） | — |
| 閾値 | multi_stage_threshold = 0.45 | — |

**Phase 2 で非採用の理由:** Phase 1 での評価結果、クエリ拡張による精度向上が確認されなかったため。

---

## 7. 差分サマリー（改善計画込み）

### 精度に影響する差分

| # | 差分項目 | Phase 1 | Phase 2 (改善後) | 精度影響 |
|---|---------|---------|-----------------|---------|
| 1 | ベクトル化対象 | combined_text | combinedContent | **同等**（同一フォーマット） |
| 2 | Embedding モデル | Gemini or Azure OpenAI | Azure OpenAI のみ | **同等**（同一モデル使用時） |
| 3 | テキスト検索 | Sudachi + Jaccard（1フィールド） | BM25 + ja.microsoft（5フィールド） | **同等〜改善** |
| 4 | スコア統合 | 加重平均 (0.9v+0.1k) | RRF + `weight` パラメータ（Semantic Ranker 廃止） | **方式変更**（`weight` でベクトル重み制御可能。セクション 4-D 参照） |
| 5 | キーワード重み付け | Sudachi 品詞重み + 位置重み | BM25 IDF + keywords フィールド | **やや変化**（方式が異なるが同等効果） |
| 6 | 業務分野スコープ | 分野別 DB（自動） | **分野別 filter（ユーザー選択）** | **同等**（実装済み） |
| 7 | LLM クエリ拡張 | あり | なし | **影響なし**（精度向上未確認のため非採用） |
| 8 | 再ランキング | oversampling + keyword boost | RRF（Semantic Ranker は精度未向上のため廃止） | **方式変更** |
| 9 | 結果件数 | 固定 (top_k=4) | **ユーザー選択 (10〜150、11段階)** | **改善**（柔軟性向上） |

### 運用面の差分

| # | 項目 | Phase 1 | Phase 2 |
|---|------|---------|---------|
| 1 | 実行形式 | Python バッチ（Excel入出力） | Teams Bot（リアルタイム） |
| 2 | インフラ | ローカル PC | Azure (AI Search + Cosmos DB) |
| 3 | Embedding 更新 | スクリプト手動実行 | Indexer 自動実行（1時間ごと） |
| 4 | チューニング | settings.yaml で weight/threshold 変更可 | `vectorQueries[].weight` でベクトル重み制御可（安定版API）。`maxTextRecallSize` でBM25側も制御可（プレビューAPI） |
| 5 | 分野追加 | DB パスとコレクション追加 | Cosmos DB にデータ追加 → Indexer 自動反映 |

---

## 8. 実装済みの検索パラメータ

### ハイブリッド検索（searchSemantic）— ベクトル重み制御対応

```typescript
const results = await searchClient.search(query, {
  queryType: "simple",                                // ← "semantic" から変更（Semantic Ranker 廃止）
  vectorSearchOptions: {
    queries: [{
      kind: "text",
      text: query,
      fields: ["contentVector"],
      weight: 5.0,                                    // ← ベクトル重み（セクション4-D参照）
    }],
  },
  select: ["id", "dataType", "categoryName", "title", "content"],
  top: safeTopN,                                      // ← ユーザー選択 (10〜150)
  filter,                                             // ← "isDeleted eq false" + categoryId filter
});
// RRFスコア（@search.score）で自然に順位付け。weight でベクトル側を増幅。
```

### キーワード検索（searchKeyword）

```typescript
const results = await searchClient.search(query, {
  queryType: "full",
  searchFields: ["title", "content", "keywords"],     // ← keywords 追加
  select: ["id", "dataType", "categoryName", "title", "content"],
  top: safeTopN,                                      // ← ユーザー選択 (10〜150)
  filter,                                             // ← "isDeleted eq false" + categoryId filter
});
```

### バリデーション（共通）

```typescript
const validCategories = CATEGORIES.map((c) => c.id);  // ホワイトリスト
const safeCategoryId = validCategories.includes(categoryId) ? categoryId : "all";
const safeTopN = Number.isNaN(rawTopN) ? 30 : Math.min(Math.max(rawTopN, 10), 150);
```

### カテゴリ一覧（UI 選択肢）

| categoryId | 表示名 | dataType | 件数 |
|-----------|--------|----------|------|
| smile | スマイル | scenario + faq | 555 + 8,679 = 9,234 |
| souzoku | 相続 | scenario | 269 |
| naibujimu | 内部事務 | scenario | 1,384 |
| torikaku | 取引時確認 | scenario | 105 |
| sousoku | 総則 | faq | 4,000 |
| yokin | 預金 | faq | 6,055 |
| (全分野) | すべて | scenario + faq | 21,047 |

### 表示件数選択肢（TOP_N_OPTIONS）

`[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]` — 11項目（Teams ChoiceSet 15件制限に収まる）

デフォルト: 30件

---

## 9. 精度改善計画: ベクトル重み制御の段階的チューニング（2026-02-18 策定）

### 背景

Phase 1 では `vector_weight = 0.9` が最も精度が高く、`0.8` では精度が低下した。
ベクトルオンリー（`1.0`）よりもキーワードを少量混ぜた方が良い結果が得られた。

Phase 2（RRF）のデフォルト（`weight = 1.0`）はベクトルとキーワードが 50:50 であり、Phase 1 の 0.9:0.1 とは大きく異なる。これが精度差の主要因と考えられる。

### 改善ステップ

| ステップ | 内容 | `weight` 値 | 対応比率 | 必要なAPI変更 |
|---------|------|------------|---------|-------------|
| **Step 1** | `weight` パラメータ追加（初期値） | 5.0 | 83% | `agent.ts` の `searchSingle()` に `weight` 追加（安定版APIで対応） |
| Step 2 | テスト結果を見て weight を調整 | 7.0 or 9.0 | 88% or 90% | 同上（値の変更のみ） |
| Step 3（任意） | `maxTextRecallSize` で BM25 をさらに抑制 | — | — | プレビュー API に変更が必要。Step 1-2 で十分な場合は不要 |
| Step 4（任意） | `threshold` で低スコア除外 | — | — | プレビュー API。ノイズが多い場合に検討 |

### 変更箇所

`maintenance-bot/src/agent.ts` の `searchSingle()` 関数（1箇所のみ）:

```diff
  const options = mode === "semantic"
    ? {
        queryType: "simple" as const,
        vectorSearchOptions: {
          queries: [
            {
              kind: "text" as const,
              text: query,
              fields: ["contentVector"],
+             weight: 5.0,
            },
          ],
        },
        ...
      }
```

### 評価方法

1. Phase 1 で精度が確認されているクエリセット（事務改定関連）で A/B テスト
2. `weight` = 1.0（現行）、5.0、7.0、9.0 の4パターンで上位10件の一致率を比較
3. Phase 1 の結果と Phase 2 の結果を突合し、順位の一致度（Top-k Overlap）を計測

---

## 参考文献

[^1]: [Hybrid Search Scoring (RRF) - Azure AI Search | Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) — RRF アルゴリズムの詳細、k=60 定数、重み付きスコア計算の解説
[^2]: [Create a hybrid query - Azure AI Search | Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-how-to-query) — ハイブリッドクエリの構築方法、`maxTextRecallSize` / `countAndFacetMode` の設定
[^3]: [Create a Vector Query - Azure AI Search | Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query) — `vectorQueries[].weight` パラメータ、`threshold` の設定、安定版/プレビュー版 API の差異
[^4]: [VectorizableTextQuery interface - @azure/search-documents | Microsoft Learn](https://learn.microsoft.com/en-us/javascript/api/@azure/search-documents/vectorizabletextquery?view=azure-node-latest) — JS SDK の `weight` プロパティ定義（`BaseVectorQuery.weight`）
[^5]: [Hybrid Search Overview - Azure AI Search | Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview) — ハイブリッド検索の全体像
[^6]: [Azure AI Search's New Hybrid and Vector Search Updates | Microsoft Tech Community](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-s-new-hybrid-and-vector-search-updates-to-boost/ba-p/4141467) — ベクトル重み制御・`maxTextRecallSize` の発表記事
