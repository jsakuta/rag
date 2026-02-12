# 検索ロジック比較: Phase 1 (rag-gemini) vs Phase 2 (maintenance-bot)

- 文書番号: COMP-SEARCH-001
- 最終更新: 2026-02-12
- 目的: Phase 1→2 の検索精度差分を網羅的に記録し、Phase 2 改善計画の基礎資料とする
- 前提: Phase 2 は「業務分野選択 + 表示件数選択 UI」を追加実装する予定

---

## 差分サマリー

| 観点 | Phase 1 (rag-gemini) | Phase 2 (maintenance-bot) | 評価 |
|------|---------------------|--------------------------|------|
| **ベクトル化** | combined_text / 3,072次元 | combinedContent / 3,072次元 | **同等**（同一フォーマット） |
| **テキスト検索** | Sudachi + Jaccard（1フィールド） | ja.microsoft + BM25（5フィールド横断） | **同等〜改善**（再現率向上） |
| **スコア統合** | 加重平均 (0.9v + 0.1k) → チューニング自在 | RRF (k=60固定) → チューニング自由度は低下 | **方式変更**（RRFは順位ベースで異スケールに強い） |
| **キーワード抽出** | Sudachi 品詞重み + 位置重み（クエリ主導） | BM25 IDF + Semantic Ranker keywords参照（ドキュメント主導） | **間接カバー**（銀行固有名詞の明示重み付けは消失） |
| **再ランキング** | oversampling + keyword boost（2段階） | BM25 + HNSW cosine + **Semantic Ranker**（3段階） | **改善**（言語モデル再評価が追加） |
| **rerankerScore** | — | 0〜4（言語モデルが生テキストを読解して付与） | cosine類似度(0〜1)とは**全く別の概念** |
| **業務分野スコープ** | 分野別DB（自動分離・高精度） | 全21,000件横断（改善後: filter で分野絞り込み） | **改善後に同等** |
| **LLM クエリ拡張** | 3戦略あり | なし（精度向上未確認 → Semantic Rankerで代替） | **影響なし** |
| **結果件数** | 固定 top_k=4 | ユーザー選択 (5/10/20) | **改善** |

> **総合:** Phase 2 は BM25 + ベクトル cosine + Semantic Ranker の3段階ランキングで Phase 1 の2段階より層が厚い。チューニング自由度は低下するが、業務分野 filter 追加後は Phase 1 と同等以上の検索精度を実現可能。

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

### Phase 2 (maintenance-bot) — 現行

```
ユーザー → Teams チャット入力
  ↓
Bot (TypeScript) → Azure AI Search API 1回コール
  ↓
AI Search 内部:
  ├─ BM25 テキスト検索 (ja.microsoft アナライザ)
  ├─ Vectorizer → クエリベクトル生成 (text-embedding-3-large)
  ├─ HNSW ベクトル検索 (cosine)
  ├─ RRF (Reciprocal Rank Fusion) でスコア統合
  └─ Semantic Ranker で再ランキング
  ↓
Adaptive Card (top 20, rerankerScore >= 1.5)
```

### Phase 2 — 改善後（予定）

```
ユーザー → Teams チャット入力
  ↓
モード選択カード: 業務分野選択 + 表示件数選択
  ↓
Bot → AI Search API (filter: categoryId eq '{選択分野}')
  ↓
AI Search 内部: （同上）
  ↓
Adaptive Card (top N件, 類似度スコア順)
```

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
- **マッチング:** BM25 が keywords フィールドを searchable として検索 + Semantic Ranker が prioritizedKeywordsFields として利用

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
- ただし Semantic Ranker が文脈理解で補完するため、全体的な精度低下は限定的

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

### 4-B. Phase 2: RRF + Semantic Ranker（2段階）

**第1段階: RRF (Reciprocal Rank Fusion)**

```
RRF_score(doc) = 1/(k + rank_BM25) + 1/(k + rank_vector)
k = 60（固定、変更不可）
```

- BM25 での順位と HNSW ベクトル検索での順位を **順位ベース** で統合
- スコア値の絶対値ではなく相対順位を使うため、スケールの異なるスコアを安全に統合できる
- AI Search の `@search.score` として返却される

**第2段階: Semantic Ranker (再ランキング)**

```
入力:  RRF 上位 50件のドキュメント
処理:  Microsoft の多言語言語モデルが以下を読解
       - titleField: title
       - prioritizedContentFields: content, combinedContent
       - prioritizedKeywordsFields: keywords
出力:  rerankerScore (0.0〜4.0)
```

| 項目 | 値 |
|------|-----|
| 値域 | 0.0〜4.0 |
| 0.0〜1.0 | ほぼ無関係 |
| 1.0〜2.0 | 部分的に関連 |
| 2.0〜3.0 | 強い関連 |
| 3.0〜4.0 | ほぼ完全一致（稀） |
| 現在のフィルタ閾値 | >= 1.5 |

### 4-C. rerankerScore とコサイン類似度の概念的違い

| 観点 | cosine similarity (Phase 1) | rerankerScore (Phase 2) |
|------|---------------------------|------------------------|
| 何を計算しているか | 2つのベクトル間の角度の余弦 | 言語モデルによるクエリ−ドキュメント間の意味的関連度 |
| 入力 | Embedding ベクトル 2本 | クエリ文字列 + ドキュメントの生テキスト（title/content/keywords） |
| 表現力 | ベクトル空間上の幾何学的距離 | 自然言語理解に基づく関連性判定 |
| 値域 | 0.0〜1.0 | 0.0〜4.0 |
| 同義語理解 | Embedding 学習に依存 | 言語モデルが動的に判断 |
| 語順・文脈 | Embedding で暗黙的に考慮 | 言語モデルが明示的に考慮 |

**重要:** Semantic Ranker は Embedding とは **別の言語モデル** を使用している。Phase 2 は実質 **3段階のランキング** を行っている:

1. BM25（テキスト一致）
2. HNSW cosine（ベクトル類似度）← Phase 1 と同等
3. Semantic Ranker（言語モデル再評価）← **Phase 1 にはない**

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
| 件数 | top_k=4（固定） | top 20（セマンティック）/ 50（キーワード） | **ユーザーが選択**（5/10/20） |
| oversampling | 2×top_k=8 件取得 → 4件に絞り込み | AI Search 内部で自動 | 同左 |

---

## 6. LLM クエリ拡張の比較

| 項目 | Phase 1 | Phase 2 |
|------|---------|---------|
| クエリ拡張 | あり（3戦略: Original / LLMEnhanced / MultiStage） | **なし** |
| 拡張方法 | Gemini/Azure OpenAI で要約・言い換え | — |
| MultiStage OR マージ | Original結果 ∪ LLM結果 → 3分類（Both/Original_Only/LLM_Only） | — |
| 閾値 | multi_stage_threshold = 0.45 | — |

**Phase 2 で非採用の理由:** Phase 1 での評価結果、クエリ拡張による精度向上が確認されなかったため。Semantic Ranker が文脈理解で同等の効果を提供する。

---

## 7. 差分サマリー（改善計画込み）

### 精度に影響する差分

| # | 差分項目 | Phase 1 | Phase 2 (改善後) | 精度影響 |
|---|---------|---------|-----------------|---------|
| 1 | ベクトル化対象 | combined_text | combinedContent | **同等**（同一フォーマット） |
| 2 | Embedding モデル | Gemini or Azure OpenAI | Azure OpenAI のみ | **同等**（同一モデル使用時） |
| 3 | テキスト検索 | Sudachi + Jaccard（1フィールド） | BM25 + ja.microsoft（5フィールド） | **同等〜改善** |
| 4 | スコア統合 | 加重平均 (0.9v+0.1k) | RRF + Semantic Ranker | **改善**（3段階ランキング） |
| 5 | キーワード重み付け | Sudachi 品詞重み + 位置重み | BM25 IDF + Semantic Ranker | **やや変化**（方式が異なるが同等効果） |
| 6 | 業務分野スコープ | 分野別 DB（自動） | **分野別 filter（ユーザー選択）** | **同等**（改善後） |
| 7 | LLM クエリ拡張 | あり | なし | **影響なし**（精度向上未確認のため非採用） |
| 8 | 再ランキング | oversampling + keyword boost | Semantic Ranker | **改善** |
| 9 | 結果件数 | 固定 (top_k=4) | **ユーザー選択 (5/10/20)** | **改善**（柔軟性向上） |

### 運用面の差分

| # | 項目 | Phase 1 | Phase 2 |
|---|------|---------|---------|
| 1 | 実行形式 | Python バッチ（Excel入出力） | Teams Bot（リアルタイム） |
| 2 | インフラ | ローカル PC | Azure (AI Search + Cosmos DB) |
| 3 | Embedding 更新 | スクリプト手動実行 | Indexer 自動実行（1時間ごと） |
| 4 | チューニング | settings.yaml で weight/threshold 変更可 | RRF k=60 固定、rerankerScore 閾値のみ変更可 |
| 5 | 分野追加 | DB パスとコレクション追加 | Cosmos DB にデータ追加 → Indexer 自動反映 |

---

## 8. 改善後の検索パラメータ設計

### セマンティック検索（改善版）

```typescript
const results = await searchClient.search(query, {
  queryType: "semantic",
  semanticSearchOptions: { configurationName: "semantic-config" },
  vectorSearchOptions: {
    queries: [{ kind: "text", text: query, fields: ["contentVector"] }],
  },
  select: ["id", "dataType", "categoryName", "title", "content"],
  top: userSelectedTopN,                              // ← ユーザー選択 (5/10/20)
  filter: `isDeleted eq false and categoryId eq '${selectedCategory}'`,  // ← 分野フィルタ
});
```

### キーワード検索（改善版）

```typescript
const results = await searchClient.search(query, {
  queryType: "full",
  searchFields: ["title", "content", "keywords"],     // ← keywords 追加
  select: ["id", "dataType", "categoryName", "title", "content"],
  top: userSelectedTopN,                              // ← ユーザー選択
  filter: `isDeleted eq false and categoryId eq '${selectedCategory}'`,  // ← 分野フィルタ
});
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
