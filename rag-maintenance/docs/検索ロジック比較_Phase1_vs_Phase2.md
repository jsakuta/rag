# 検索ロジチE��比輁E Phase 1 (rag-local) vs Phase 2 (maintenance-bot)

- 斁E��番号: COMP-SEARCH-001
- 最終更新: 2026-02-12�E�Eemantic Ranker 廁E�� + topN 選択肢拡張を反映�E�E
- 目皁E Phase 1ↁE の検索精度差刁E��網羁E��に記録し、Phase 2 改喁E�E結果を反映する
- 前提: Phase 2 は「業務�E野選抁E+ 表示件数選抁EUI」を実裁E��み。Semantic Ranker は精度向上が確認されなかったため廁E��

---

## 差刁E��マリー

| 観点 | Phase 1 (rag-local) | Phase 2 (maintenance-bot) | 評価 |
|------|---------------------|--------------------------|------|
| **ベクトル匁E* | combined_text / 3,072次允E| combinedContent / 3,072次允E| **同筁E*�E�同一フォーマット！E|
| **チE��スト検索** | Sudachi + Jaccard�E�Eフィールド！E| ja.microsoft + BM25�E�Eフィールド横断�E�E| **同等〜改喁E*�E��E現玁E��上！E|
| **スコア統吁E* | 加重平坁E(0.9v + 0.1k) ↁEチューニング自在 | RRF (k=60固宁E ↁEチューニング自由度は低丁E| **方式変更**�E�ERFは頁E���Eースで異スケールに強ぁE��E|
| **キーワード抽出** | Sudachi 品詞重み + 位置重み�E�クエリ主導！E| BM25 IDF + keywords フィールド検索�E�ドキュメント主導！E| **間接カバ�E**�E�銀行固有名詞�E明示重み付けは消失�E�E|
| **再ランキング** | oversampling + keyword boost�E�E段階！E| BM25 + HNSW cosine の RRF 統合！E段階！E| **方式変更**�E�Eemantic Ranker 廁E��、RRF のみ�E�E|
| **スコア表示** | cosine 0、E + keyword 0、E ↁE加重平坁E| RRF @search.score�E�頁E���Eースの統合スコア�E�E| **方式変更** |
| **業務�E野スコーチE* | 刁E��別DB�E��E動�E離・高精度�E�E| filter で刁E��絞り込み�E�実裁E��み�E�E| **同筁E* |
| **LLM クエリ拡張** | 3戦略あり | なし（精度向上未確認�Eため非採用�E�E| **影響なぁE* |
| **結果件数** | 固宁Etop_k=4 | ユーザー選抁E(10、E50、E1段隁E | **改喁E* |

> **総合:** Phase 2 は BM25 + ベクトル cosine の RRF 統合！E段階ランキング�E�。Semantic Ranker は PoC 評価で精度向上が確認されなかったため廁E��。業務�E釁Efilter + 件数選抁EUI の実裁E��より、Phase 1 と同等以上�E検索精度と柔軟性を実現、E

---

## 1. アーキチE��チャ全体像

### Phase 1 (rag-local)

```
ユーザー ↁEExcel バッチ�E劁E
  ↁE
Python (Searcher)
  ├─ Sudachi 形態素解极EↁEキーワード抽出
  ├─ Embedding API (Gemini or Azure OpenAI) ↁEクエリベクトル生�E
  ├─ ChromaDB (ローカル) ↁEベクトル検索 (cosine)
  ├─ KeywordSearchEngine ↁEJaccard 類似度計箁E
  ├─ 加重スコア統吁E(0.9×vector + 0.1×keyword)
  └─ Excel 出劁E(top_k=4)
```

### Phase 2 (maintenance-bot)  E実裁E��み

```
ユーザー ↁETeams チャチE��入劁E
  ↁE
モード選択カーチE 業務�E野選抁E+ 表示件数選択！E0、E50件�E�E
  ↁE
Bot (TypeScript) ↁEAzure AI Search API 1回コール
  ↁE
AI Search 冁E��:
  ├─ BM25 チE��スト検索 (ja.microsoft アナライザ)
  ├─ Vectorizer ↁEクエリベクトル生�E (text-embedding-3-large)
  ├─ HNSW ベクトル検索 (cosine)
  └─ RRF (Reciprocal Rank Fusion) でスコア統吁E
  ↁE
Adaptive Card (top N件, RRFスコア頁E 10件/ペ�Eジ)
```

> **注:** Semantic Ranker�E�E段階目の再ランキング�E��E PoC 評価で精度向上が確認されなかったため廁E��、E
> 上佁E0件制紁E��解除され、ユーザーが選択した件数�E�最大150件�E�をそ�Eまま使用可能になった、E

---

## 2. ベクトル化対象の比輁E

### 2-A. 何をベクトル化してぁE��ぁE

| 頁E�� | Phase 1 | Phase 2 | 差刁E|
|------|---------|---------|------|
| 対象チE��スチE| `combined_text` | `combinedContent` | **同一フォーマッチE* |
| Embedding モチE�� | Gemini embedding-001 or Azure OpenAI text-embedding-3-large | Azure OpenAI text-embedding-3-large のみ | Phase 1 は選択可能 |
| 次允E�� | 3,072 | 3,072 | 同一 |
| ベクトル化タイミング | Python スクリプト実行時�E�クライアント�E�E�E| AI Search Indexer 実行時�E�サーバ�E側�E�E| Phase 2 は自勁E|
| ベクトル DB | ChromaDB (ローカル HNSW) | AI Search (HNSW) | 同一アルゴリズム |

### 2-B. combinedContent のフォーマッチE

**シナリオ:**
```
"刁E��E {Lv1 > Lv2 > ... > Lv(n-2)} | 質啁E {Lv(n-1)} | 回筁E {Lv(n)}"
```

**FAQ:**
```
"質啁E {question} | 回筁E {answer}"
```

**Phase 1 と Phase 2 で同一フォーマット、E* Phase 2 の `convert-excel-to-json.py` は Phase 1 の `TextCombiner._build_combined_text()` と同じ生�Eルールを実裁E��てぁE��、E

### 2-C. combinedContent の構�E要素決定ロジチE��

#### シナリオの右→左スキャン�E�Ehase 1 / Phase 2 共通！E

```
Excel衁E [Lv1="預��関連", Lv2="普通預��", Lv3="新規口座開設", Lv4="本人確誁E, Lv5="運転免許証の場吁E]

右→左スキャン:
  Lv5 "運転免許証の場吁E ↁE最右非空 ↁEanswer (content)
  Lv4 "本人確誁E         ↁEanswerの1つ左 ↁEquestion
  Lv1〜Lv3              ↁEquestionより左 ↁEhierarchy (刁E��E

結果:
  content = "運転免許証の場吁E
  title   = "/預��関連/普通預��/新規口座開設/本人確誁E
  combined = "刁E��E 預��関連 > 普通預�� > 新規口座開設 | 質啁E 本人確誁E| 回筁E 運転免許証の場吁E
```

#### FAQ の列�EチE��ング

| ファイル | title (質啁E | content (回筁E | 補足 |
|---------|-------------|---------------|------|
| スマイル_履歴チE�Eタ | `問い合わせ` 刁E| `回答` 刁E+ `\n\n` + `補足回答` 刁E| タグなぁE|
| 総則_履歴チE�Eタ | `質問` 刁E| `回答` 刁E| `タグ付け` 列あめE|
| 預��_履歴チE�Eタ | `質問` 刁E| `回答` 刁E| `タグ付け` 列あめE|

**差刁E** Phase 1 は `dynamic_db_manager.py` で `\n` 区刁E��、Phase 2 は `convert-excel-to-json.py` で `\n\n` 区刁E��。実質皁E��響は軽微�E�Embedding の類似度にほぼ影響なし）、E

---

## 3. チE��スト検索の比輁E

### 3-A. ト�Eクン匁E

| 頁E�� | Phase 1 (Sudachi) | Phase 2 (ja.microsoft) |
|------|-------------------|----------------------|
| エンジン | Sudachi (SplitMode.C) | Microsoft 日本語アナライザ (MeCab ベ�Eス) |
| 辞書 | sudachi-dictionary-full | Microsoft 独自辞書 |
| 刁E��ち書き精度 | 高（銀行用語に強ぁE��E| 高（一般用語に強ぁE��E|
| カスタム辞書 | Sudachi ユーザー辞書追加可能 | 不可�E�Ezure 管琁E��E|

**精度影響:** 両老E��も日本語形態素解析としては十�Eな品質。Sudachi は「取引時確認」「預り物件」等�E褁E��語を1ト�Eクンとして認識する精度がやめE��いが、BM25 は部刁E��致でもスコアが付くため実用差は小さぁE��E

### 3-B. スコア計箁E

| 頁E�� | Phase 1 (Jaccard) | Phase 2 (BM25) |
|------|-------------------|----------------|
| アルゴリズム | Jaccard 類似度�E�集合�E共通率�E�E| BM25 (TF-IDF 拡張 + 斁E��長正規化) |
| 計算弁E| `weighted_intersection / union_size` | `Σ IDF(t) ÁE(tf ÁE(k1+1)) / (tf + k1 ÁE(1-b+b×dl/avgdl))` |
| 斁E��長正規化 | なし（集合�Eースなので斁E��長に鈍感�E�E| あり�E�E=0.75 で長ぁE��書のスコアを抑制�E�E|
| 値埁E| 0.0、E.0 | 0.0〜�E�E�上限なし、相対スコア�E�E|
| 検索対象フィールチE| combined_text�E�Eフィールド！E| title, content, combinedContent, categoryName, keywords�E�E*5フィールチE*�E�E|

**精度影響:**
- BM25 は IDF�E�送E��書頻度�E�を老E�Eするため、「口座」�Eような高頻度語よりも「AMLフィルター」�Eような低頻度語を重視する、Eaccard にはこ�E概念がなぁE
- Phase 2 は 5 フィールド横断検索なので、title に含まれなくてめEkeywords めEcontent で拾える、E*再現玁E�E Phase 2 の方が高い**
- BM25 の斁E��長正規化により、短ぁEFAQ�E�E解決済み"�E��E長ぁE��ナリオに対して不当に高スコアにならなぁE

### 3-C. キーワード抽出の詳細比輁E

#### Phase 1: Sudachi ベ�Eスの明示皁E��ーワード抽出

```python
# keyword_search_engine.py
def extract_keywords(self, text, top_k=5):
    morphemes = tokenizer.tokenize(text, SplitMode.C)
    keywords = []
    for m in morphemes:
        if m.part_of_speech()[0] == '名詁E:           # 名詞�Eみ
            important = ['固有名詁E, '一般']
            weight = 2 if m.part_of_speech()[1] in important else 1  # 固有名詞�E一般名詞�E2倁E
            word = m.dictionary_form()
            if len(word) > 1:                          # 1斁E���E除夁E
                keywords.extend([word] * weight)
    # ストップワード除去
    filtered = {w: c for w, c in Counter(keywords).items() if w not in STOP_WORDS}
    return [w for w, _ in Counter(filtered).most_common(top_k)]
```

**ストップワーチE** `['こと', 'も�E', 'これ', 'それ', 'ところ', '方', 'する', 'ある', 'ぁE��', 'れる', 'られめE, 'なめE, 'そ�E']`

**品詞フィルタ:**
- 名詞�Eみ抽出�E�動詞�E形容詞�E助詞等�E除外！E
- 固有名詞�E一般名詁E 重み 2
- そ�E他名詞（数詞、接尾辞等！E 重み 1

**位置重み付け:**
```python
# keyword_search_engine.py (Line 120-124)
weighted_score = sum(
    position_weight if reference_text.find(kw) < len(reference_text) // 2 else 1.0
    for kw in intersection
)
# position_weight = 1.2�E�テキスト前半に出現するキーワードを1.2倍！E
```

#### Phase 2: 明示皁E��キーワード抽出なぁE

- **クエリ側:** ユーザー入力をそ�Eまま AI Search に送信。トークン化�E ja.microsoft が�E部で実衁E
- **ドキュメント�E:** `keywords` フィールドに事前格納済み�E�シナリオ: 階層チE��スト、FAQ: タグから抽出�E�E
- **マッチング:** BM25 ぁEkeywords フィールドを searchable として検索�E�キーワード検索時�E searchFields に明示持E��！E

**Phase 2 の keywords フィールド�E中身:**

| チE�Eタ種別 | keywords の決定ロジチE�� | 侁E|
|-----------|----------------------|-----|
| シナリオ | 階層パスの吁E��ベルチE��スチE| `["預��関連", "普通預��", "新規口座開設", "本人確誁E]` |
| FAQ (タグあり) | タグ斁E���Eから `Lv\d+:\s*` でパ�Eス | `["営業事務", "預り物件管琁E��スチE��"]` |
| FAQ (タグなぁE | 空配�E | `[]` |

**精度影響:**
- Phase 1 は **クエリ側** からキーワードを抽出して Jaccard で比輁E��クエリ主導！E
- Phase 2 は **ドキュメント�E** に事前格納したキーワードを BM25 で検索�E�ドキュメント主導！E
- Phase 1 の品詞フィルタ�E�名詞�Eみ�E�と位置重み付けは Phase 2 にはなぁEↁE**銀行固有名詞�E重み付けが弱匁E*
- BM25 の IDF 重み付けが部刁E��に補完するが、E��行固有名詞に対する明示皁E��重み付けは Phase 1 の方が優れてぁE��

---

## 4. スコア統合�E比輁E

### 4-A. Phase 1: 明示皁E��重平坁E

```
combined_score = 0.9 ÁEvector_similarity + 0.1 ÁEkeyword_similarity
```

| パラメータ | チE��ォルト値 | 調整可能 |
|-----------|------------|---------|
| vector_weight | 0.9 | settings.yaml で変更可 |
| keyword_weight | 0.1 (= 1.0 - vector_weight) | 自動計箁E|
| VECTOR_SEARCH_MULTIPLIER | 2 | oversampling 倍率 |
| top_k | 4 | 最終結果件数 |

**特徴:**
- スコア冁E���E�Eector_sim, keyword_sim�E�が個別に見えめE
- weight を変えて A/B チE��ト可能
- 値埁E 0.0、E.0�E�コサイン類似度ベ�Eス�E�E

### 4-B. Phase 2: RRF のみ�E�Eemantic Ranker 廁E���E�E

**RRF (Reciprocal Rank Fusion)**

```
RRF_score(doc) = 1/(k + rank_BM25) + 1/(k + rank_vector)
k = 60�E�固定、変更不可�E�E
```

- BM25 での頁E��と HNSW ベクトル検索での頁E��を **頁E���Eース** で統吁E
- スコア値の絶対値ではなく相対頁E��を使ぁE��め、スケールの異なるスコアを安�Eに統合できる
- AI Search の `@search.score` として返却されめE
- Bot では `result.score` として取得し、小数点以丁E桁で表示

**Semantic Ranker 廁E��の経緯:**

| 頁E�� | 冁E�� |
|------|------|
| 廁E��琁E�� | PoC 評価で rerankerScore による精度向上が確認されなかっぁE|
| 上佁E0件制紁E| Semantic Ranker は RRF 上佁E0件のみ再ランキング�E�固定制紁E���E 大量取得時にスコア混在の問顁E|
| 廁E��による利点 | top の50件制紁E��解除され、ユーザー選択値�E�最大150件�E�をそ�Eまま使用可能 |
| 変更冁E�� | `queryType: "semantic"` ↁE`queryType: "simple"` + `vectorSearchOptions`�E�ハイブリチE��検索維持E��E|

### 4-C. Phase 2 のランキング構�E�E�現行！E

Phase 2 は **2段階�Eランキング** を行ってぁE��:

1. BM25�E�テキスト一致�E��E ja.microsoft アナライザ
2. HNSW cosine�E��Eクトル類似度�E��E Phase 1 と同筁E

上訁Eつの結果めERRF で頁E���Eース統吁EↁE`@search.score` として返却

---

## 5. 検索スコープ�E比輁E

### 5-A. 業務�E野�E扱ぁE

| 頁E�� | Phase 1 | Phase 2 (現衁E | Phase 2 (改喁E��E |
|------|---------|---------------|-----------------|
| DB/インチE��クス構�E | 業務�E野ごとに ChromaDB コレクション刁E�� | 全刁E��ぁE1 インチE��クスに同屁E| 同左�E�Eilter で刁E���E�E|
| 刁E��選抁E| ファイル名から�E動判宁E| なし（�E刁E��横断検索�E�E| **ユーザーぁEUI で選抁E* |
| 精度影響 | 同一刁E��冁E�Eみ比輁EↁE**高精度** | 全 21,000 件から検索 ↁEノイズ混入リスク | filter で刁E��絞り込み ↁEPhase 1 相彁E|

**全刁E��統合�E問題点:**
- 「口座開設」で検索すると、スマイル・預��・総則の全カチE��リから結果が返る
- 異なる業務�E野�E斁E��が混在し、ユーザーが求める�E野�E結果が埋もれめE
- Phase 1 は刁E��別 DB なのでこ�E問題がなぁE

**改喁E��E** `filter: "categoryId eq 'smile'"` で Phase 1 と同等�E刁E��絞り込みを実現

### 5-B. 結果件数

| 頁E�� | Phase 1 | Phase 2 (現衁E | Phase 2 (改喁E��E |
|------|---------|---------------|-----------------|
| 件数 | top_k=4�E�固定！E|  E| **ユーザーが選抁E*�E�E0、E50、E1段階！E|
| oversampling | 2×top_k=8 件取征EↁE4件に絞り込み |  E| AI Search 冁E��で自勁E|

---

## 6. LLM クエリ拡張の比輁E

| 頁E�� | Phase 1 | Phase 2 |
|------|---------|---------|
| クエリ拡張 | あり�E�E戦略: Original / LLMEnhanced / MultiStage�E�E| **なぁE* |
| 拡張方況E| Gemini/Azure OpenAI で要紁E�E言ぁE��ぁE|  E|
| MultiStage OR マ�Eジ | Original結果 ∪ LLM結果 ↁE3刁E��！Eoth/Original_Only/LLM_Only�E�E|  E|
| 閾値 | multi_stage_threshold = 0.45 |  E|

**Phase 2 で非採用の琁E��:** Phase 1 での評価結果、クエリ拡張による精度向上が確認されなかったため、E

---

## 7. 差刁E��マリー�E�改喁E��画込み�E�E

### 精度に影響する差刁E

| # | 差刁E��E�� | Phase 1 | Phase 2 (改喁E��E | 精度影響 |
|---|---------|---------|-----------------|---------|
| 1 | ベクトル化対象 | combined_text | combinedContent | **同筁E*�E�同一フォーマット！E|
| 2 | Embedding モチE�� | Gemini or Azure OpenAI | Azure OpenAI のみ | **同筁E*�E�同一モチE��使用時！E|
| 3 | チE��スト検索 | Sudachi + Jaccard�E�Eフィールド！E| BM25 + ja.microsoft�E�Eフィールド！E| **同等〜改喁E* |
| 4 | スコア統吁E| 加重平坁E(0.9v+0.1k) | RRF のみ�E�Eemantic Ranker 廁E���E�E| **方式変更**�E�E段階ランキング�E�E|
| 5 | キーワード重み付け | Sudachi 品詞重み + 位置重み | BM25 IDF + keywords フィールチE| **めE��変化**�E�方式が異なるが同等効果！E|
| 6 | 業務�E野スコーチE| 刁E��別 DB�E��E動！E| **刁E��別 filter�E�ユーザー選択！E* | **同筁E*�E�実裁E��み�E�E|
| 7 | LLM クエリ拡張 | あり | なぁE| **影響なぁE*�E�精度向上未確認�Eため非採用�E�E|
| 8 | 再ランキング | oversampling + keyword boost | RRF�E�Eemantic Ranker は精度未向上�Eため廁E���E�E| **方式変更** |
| 9 | 結果件数 | 固宁E(top_k=4) | **ユーザー選抁E(10、E50、E1段隁E** | **改喁E*�E�柔軟性向上！E|

### 運用面の差刁E

| # | 頁E�� | Phase 1 | Phase 2 |
|---|------|---------|---------|
| 1 | 実行形弁E| Python バッチE��Excel入出力！E| Teams Bot�E�リアルタイム�E�E|
| 2 | インフラ | ローカル PC | Azure (AI Search + Cosmos DB) |
| 3 | Embedding 更新 | スクリプト手動実衁E| Indexer 自動実行！E時間ごと�E�E|
| 4 | チューニング | settings.yaml で weight/threshold 変更可 | RRF k=60 固定（チューニング不可�E�E|
| 5 | 刁E��追加 | DB パスとコレクション追加 | Cosmos DB にチE�Eタ追加 ↁEIndexer 自動反映 |

---

## 8. 実裁E��みの検索パラメータ

### ハイブリチE��検索�E�EearchHybrid�E� ESemantic Ranker 廁E��済み

```typescript
const results = await searchClient.search(query, {
  queryType: "simple",                                // ↁE"semantic" から変更
  // semanticSearchOptions 削除�E�Eemantic Ranker 廁E���E�E
  vectorSearchOptions: {
    queries: [{ kind: "text", text: query, fields: ["contentVector"] }],
  },
  select: ["id", "dataType", "categoryName", "title", "content"],
  top: safeTopN,                                      // ↁEユーザー選抁E(10、E50)
  filter,                                             // ↁE"isDeleted eq false" + categoryId filter
});
// rerankerScore フィルタ削除 ↁERRFスコア�E�Esearch.score�E�で自然に頁E��付け
```

### キーワード検索�E�EearchKeyword�E�E

```typescript
const results = await searchClient.search(query, {
  queryType: "full",
  searchFields: ["title", "content", "keywords"],     // ↁEkeywords 追加
  select: ["id", "dataType", "categoryName", "title", "content"],
  top: safeTopN,                                      // ↁEユーザー選抁E(10、E50)
  filter,                                             // ↁE"isDeleted eq false" + categoryId filter
});
```

### バリチE�Eション�E��E通！E

```typescript
const validCategories = CATEGORIES.map((c) => c.id);  // ホワイトリスチE
const safeCategoryId = validCategories.includes(categoryId) ? categoryId : "all";
const safeTopN = Number.isNaN(rawTopN) ? 30 : Math.min(Math.max(rawTopN, 10), 150);
```

### カチE��リ一覧�E�EI 選択肢�E�E

| categoryId | 表示吁E| dataType | 件数 |
|-----------|--------|----------|------|
| smile | スマイル | scenario + faq | 555 + 8,679 = 9,234 |
| souzoku | 相綁E| scenario | 269 |
| naibujimu | 冁E��事務 | scenario | 1,384 |
| torikaku | 取引時確誁E| scenario | 105 |
| sousoku | 総則 | faq | 4,000 |
| yokin | 預�� | faq | 6,055 |
| (全刁E��) | すべて | scenario + faq | 21,047 |

### 表示件数選択肢�E�EOP_N_OPTIONS�E�E

`[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150]`  E11頁E���E�Eeams ChoiceSet 15件制限に収まる！E

チE��ォルチE 30件
