# 運用保守効率化AI パフォーマンス改善 設計書

## 概要

revision-eval（事務改定評価システム）の検索パフォーマンスが低い問題について、
バッチ版（evaluate_revisions.py）、UI版（eval_ui.py）、検索エンジン層（src/core/search/）を
網羅的に調査し、ボトルネックの特定と改善アプローチを設計する。

併せて、顧客引き継ぎのためのターミナルログ可読性改善も設計する。

---

## 調査結果サマリー

4つの独立調査エージェントにより、合計30件超のボトルネックを検出。
重複を統合し15件のユニークな問題に整理した。

### 根本原因の構造

1クエリ実行時のコスト構造（UI版 hybrid、providers=both、area=1）:

```
process_query()
└─ execute_dual_provider_search()
   ├─ _search_with_provider("azure_openai")      ~30-60秒
   │   ├─ create_embedding_model()                P6: 毎回再構築
   │   ├─ MetadataVectorDB()                      P6: 毎回再構築
   │   ├─ collection.get(全11,439件)              P1: キャッシュなし
   │   ├─ text_combiner.parse() x 11,439件        P1
   │   ├─ build_cache() -> Sudachi x 1,384件      P2: キャッシュ破棄
   │   ├─ create_llm() + QueryEnhancer()          P6: 毎回再構築
   │   └─ orchestrator.execute()
   │       ├─ extract_keywords(Sudachi)
   │       ├─ encode_query(embedding API)          P10: 1回目
   │       ├─ vector_search + keyword_match
   │       ├─ query_enhancer.enhance(LLM API)      P3: area重複
   │       ├─ encode_query(embedding API)          P10: 2回目
   │       └─ vector_search + keyword_match + merge P8: O(N^2)
   │
   └─ _search_with_provider("vertex_ai")          P4: 直列で+30-60秒
       └─ (上記と同じ処理を完全に繰り返す)
```

1クエリあたりのAPI/重処理の回数:
- ChromaDB全件取得: 2回（P1）
- Sudachi全件解析: 2回 x ~1,384件（P2）
- LLM API: 2回 同一テキストで重複（P3）
- Embedding API: 4回（P10）
- エンジン初期化: 2セット（P6）

---

## ボトルネック一覧

### 深刻度: 高

#### P1: ChromaDB全件取得の繰り返し（キャッシュなし）

- バッチ: `evaluate_revisions.py` `_get_reference_queries` L251
- UI: `eval_ui.py` `_search_with_provider` L348
- 影響規模: 改定6件 x area2 x provider2 = 24回の全件取得（各最大11,439件）
- `MetadataVectorDB` の `_client_cache` はクライアントのみキャッシュし、ドキュメントデータはキャッシュしない

#### P2: Sudachi全件形態素解析の繰り返し（キャッシュ破棄）

- `keyword_search_engine.py` `build_cache` L154-171
- `evaluate_revisions.py` `_create_orchestrator` L220-224
- `KeywordSearchEngine` が毎回 new されるため `_keyword_cache` が破棄される
- 影響規模: 24回 x ~1,384件 = 約33,000回のSudachi呼び出し

#### P3: LLMクエリ拡張の重複呼び出し

- `multi_stage_orchestrator.py` `execute` L115-123
- `query_enhancer.enhance()` が area x provider 回呼ばれる（最大4回、同一テキスト）
- 1回あたり1-5秒 + リトライ時最大16秒の wait

#### P4: Azure/VertexAI検索の完全直列実行

- バッチ: `evaluate_revisions.py` L621-647
- UI: `eval_ui.py` `execute_dual_provider_search` L180-190
- 2つのプロバイダーは完全に独立（異なるDB、異なるAPIエンドポイント）
- 並列化で実行時間が半減可能

#### P5: LLM分析（judgment）の直列実行

- `evaluate_revisions.py` `_run_llm_analysis` L512-545
- MAX_RESULTS=100 の場合、Azure 100件 + VertexAI 100件 = 最大200回のLLM直列呼び出し
- 200回 x 1-3秒 = 3-10分

#### P6: UI版 エンジン群の毎回再構築（Streamlitキャッシュ未使用）

- `eval_ui.py` `_search_with_provider` L325-393
- `create_embedding_model` + `create_llm` + `MetadataVectorDB` が毎クエリ x area x provider 回生成
- `@st.cache_resource` が一切使われていない
- `ChromaDBKeywordSearcher` も毎回 new（`_collection_cache` が破棄される）

#### P7: UI版 source_filterがDB側でなくPython側フィルタ + バグ

- `eval_ui.py` `_search_with_provider` L347-364
- `collection.get()` に `where={"source": source_filter}` を渡していない（全件取得後にPython側フィルタ）
- `orchestrator.execute()` に `filter_metadata` を渡していない（ベクトル検索もフィルタなし）
- **バグ**: hybrid検索結果の dict に `_source` キーが存在しない（L401-411で追加されていない）
  → `"_source" not in r` が常に True → **フィルタが実質無効**

### 深刻度: 中

#### P8: `_merge_results` の O(N x M) 線形探索

- `multi_stage_orchestrator.py` L286-300
- `both_ids` の各要素に対して `original_results` と `llm_results` を線形探索
- max_results=100 では軽微だが、拡大時に顕在化

#### P9: `_fetch_scenario_content` が未発見シナリオごとにChromaDBクエリ

- `evaluate_revisions.py` L129-159
- `_get_reference_queries` で既に全ドキュメントを取得済みなのに、同じDBに再度クエリ
- 正解発見率が低い改定初期で顕在化

#### P10: embedding API が Stage1/Stage2 で2回直列呼び出し

- `vector_search_engine.py` `encode_query` L52
- Stage 1（原文）と Stage 2（LLMクエリ）でそれぞれ1回ずつ呼び出し
- Azure OpenAI の場合1リクエストあたり50-200ms

#### P11: `ChromaDBKeywordSearcher._collection_cache` にLRU上限なし

- `chromadb_keyword_search.py` L65
- naibujimu全件ドキュメント（11,439件 x 平均200-400文字）がメモリに永久保持
- Streamlitプロセスが長時間稼働するとメモリ増大

#### P12: `text_combiner.parse()` の同一ドキュメント2回パース

- `multi_stage_orchestrator.py` `_build_result_data` L221
- Stage 1 と Stage 2 の `_execute_hybrid_search` で同一 doc_id が2回パースされる

### 深刻度: 低

#### P13: `text_combiner.parse()` のf-string定数未キャッシュ

- `text_combiner.py` L56-65
- `f"{self.LABEL_HIERARCHY}: "` が毎呼び出しで生成

#### P14: Excel書き込みのセル単位 `worksheet.write`

- `evaluate_revisions.py` L1035-1087
- 数千回のPython関数呼び出し（xlsxwriter内部バッファリングで実I/Oは少ない）

#### P15: `iterrows()` による低速ループ（DB構築時のみ）

- `dynamic_db_manager.py` L1058 / `evaluate_revisions.py` L711-714

---

## 改善アプローチ

### フェーズ1: キャッシュ導入（最大の効果、低リスク）

対象: P1, P2, P6

#### アプローチA: バッチ版 — インスタンスレベルメモ化

`RevisionEvaluator` に以下のキャッシュを導入:

```python
class RevisionEvaluator:
    def __init__(self, ...):
        # 既存のコード...
        self._reference_queries_cache: Dict[Tuple[str, str], List[str]] = {}
        self._keyword_engine_cache: Dict[Tuple[str, str], KeywordSearchEngine] = {}
        self._orchestrator_cache: Dict[Tuple[str, str, float], MultiStageOrchestrator] = {}
```

- `_get_reference_queries`: `(area, provider)` をキーにキャッシュ。改定ループ全体で再利用
- `_create_orchestrator`: `KeywordSearchEngine` と `build_cache` 結果をキャッシュ
- 改定6件 x area2 x provider2 = 24回 → 最大4回（area x provider の組み合わせ数）に削減

#### アプローチB: UI版 — `@st.cache_resource` 導入

```python
@st.cache_resource
def get_cached_engine(provider: str, area: str, db_path: str):
    """エンジン群をセッション横断でキャッシュ"""
    embedding_model = create_embedding_model(...)
    vector_db = MetadataVectorDB(...)
    keyword_engine = KeywordSearchEngine(...)
    # build_cache はここで1回だけ実行
    return embedding_model, vector_db, keyword_engine

@st.cache_resource
def get_cached_keyword_searcher(base_db_path: str):
    """ChromaDBKeywordSearcherをセッション横断でシングルトン化"""
    keyword_engine = KeywordSearchEngine(...)
    return ChromaDBKeywordSearcher(base_db_path=base_db_path, keyword_engine=keyword_engine, ...)
```

**推定効果**: 2回目以降のクエリで ChromaDB全件取得 + Sudachi解析をスキップ → 数十秒 → 数秒

### フェーズ2: LLMクエリ拡張のキャッシュ（中程度の効果、低リスク）

対象: P3

#### アプローチ: `evaluate_revision` 先頭で1回だけ拡張

```python
def evaluate_revision(self, revision, revision_content, correct_ids, ...):
    # revision_content に対するLLM拡張を1回だけ実行
    llm_enhanced_query = self._get_or_create_llm_query(revision_content)

    # search_revision_multi_stage に llm_query を渡す
    azure_results = self.search_revision_multi_stage(
        ..., pre_enhanced_query=llm_enhanced_query
    )
```

- `MultiStageOrchestrator.execute` に `pre_enhanced_query` パラメータを追加
- 指定時は `query_enhancer.enhance` をスキップして渡されたクエリを使用
- area x provider 回 → 1回に削減

**推定効果**: LLM API呼び出し 4回 → 1回（1改定あたり3-15秒短縮）

### フェーズ3: 並列化（大きな効果、中リスク）

対象: P4, P5

#### アプローチA: プロバイダー並列化

```python
def evaluate_revision(self, ...):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if providers in ("both", "azure"):
            futures["azure"] = executor.submit(
                self.search_revision_multi_stage, ..., "azure_openai"
            )
        if providers in ("both", "vertex"):
            futures["vertex"] = executor.submit(
                self.search_revision_multi_stage, ..., "vertex_ai"
            )
        # 両方の完了を待つ
        for key, future in futures.items():
            results[key] = future.result()
```

**推定効果**: 実行時間が約半分（Azure 30秒 + Vertex 30秒 → 並列で30秒）

#### アプローチB: LLM分析並列化

```python
def _run_llm_analysis(self, results, revision_content):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(self._evaluate_single_result, r, revision_content): r
            for r in results
        }
        for future in concurrent.futures.as_completed(futures):
            result = futures[future]
            # result に LLM 分析結果を反映
```

- レート制限を考慮して max_workers=5
- **推定効果**: 200件 x 2秒 = 400秒 → 200件/5並列 x 2秒 = 80秒（5倍速）

### フェーズ4: DB側フィルタの適用（正確性修正 + パフォーマンス改善）

対象: P7

#### アプローチ

1. `collection.get(where={"source": source_filter})` でDB側フィルタ
2. `orchestrator.execute(filter_metadata={"source": source_filter})` でベクトル検索もDB側フィルタ
3. hybrid検索結果の dict に `_source` キーを追加（バグ修正）
4. Python側の後フィルタを削除

```python
# 変更前（eval_ui.py L347-352）
if source_filter:
    result = vector_db.collection.get(include=["documents", "metadatas"])
else:
    result = vector_db.collection.get(include=["documents"])

# 変更後
get_kwargs = {"include": ["documents"]}
if source_filter:
    get_kwargs["include"].append("metadatas")
    get_kwargs["where"] = {"source": source_filter}
result = vector_db.collection.get(**get_kwargs)
```

**推定効果**: 全11,439件 → シナリオのみ1,384件（88%削減）

### フェーズ5: 細かい最適化

対象: P8, P9, P10, P12

- P8: `_merge_results` のdict化（O(N^2) → O(N)）
- P9: `_get_reference_queries` 結果を `row_index` キーの dict に保持し `_fetch_scenario_content` で再利用
- P12: `_build_result_data` の parse 結果キャッシュ

---

## ターミナルログ改善設計

### 方針

1. 顧客向けの言葉に置き換え（内部パラメータは `--verbose` 時のみ）
2. `logger.info` と `print_*` の役割分離（進捗は `print_*` に統一）
3. 処理中のフィードバック強化（スピナー / ステップ表示）
4. エリア名の日本語表示（`rev03_naibujimu` → 「内部事務」）

### 改善箇所一覧

#### L1: 評価設定の簡素化

変更前:
```
─────────── 評価設定 ───────────
│ LLM分析: 無効
│ 最大検索結果数: 30
│ デフォルトベクトル重み: 0.7
│ キーワード必須検索: rev02_souzoku
│ 類似検索(hybrid): rev01_smile, rev03_...
│ フィルタモード: threshold
│ 閾値 (Azure): 0.4
│ 閾値 (VertexAI): 0.35
│ プロバイダー: 両方
```

変更後:
```
─────────── 実行設定 ───────────
│ 検索プロバイダー: Azure + VertexAI
│ LLM判定: 無効
│ 評価対象: 6件の改定
```

技術パラメータは `--verbose` フラグ追加時のみ出力。

#### L2: `logger.info` と `print_*` の二重出力を排除

```python
# 削除: logger.info(f"  {area}: {len(results)}件取得")   # evaluate_revisions.py L423
# 削除: logger.info(f"\n結果を保存しました: {output_file}")  # L744
# 残す: print_search_result / print_completion のみ
```

#### L3: 検索中スピナーの追加

```python
# 変更後のイメージ
with console.status("[bold]Azure で検索中...", spinner="dots"):
    azure_results = self.search_revision_multi_stage(...)
print_search_result("azure", total_azure, azure_areas, azure_correct, len(correct_ids))
```

#### L4: エリア名の日本語変換

`AREA_DISPLAY_NAMES` 辞書を `evaluate_revisions.py` または `logger.py` に追加:

```python
AREA_DISPLAY_NAMES = {
    "naibujimu": "内部事務",
    "smile": "スマイル",
    "souzoku": "相続",
    "torikaku": "取引時確認",
}

def get_display_name(area: str) -> str:
    """rev03_naibujimu -> 内部事務 のように変換"""
    for key, name in AREA_DISPLAY_NAMES.items():
        if key in area:
            return name
    return area
```

`print_search_result` と `print_revision_header` で使用。

#### L5: DB MISSING時のガイド表示

```python
# 変更前
azure_status = "[green]OK[/green]" if azure_path.exists() else "[red]MISSING[/red]"

# 変更後
if missing_dbs:
    print_status(
        "以下のDBが見つかりません。build_db.py --revisions-only を実行してください:",
        "warning"
    )
    for db in missing_dbs:
        print_status(f"  - {get_display_name(db)}", "warning")
```

#### L6: traceback の Rich Console 経由化

```python
# 変更前
traceback.print_exc()

# 変更後
console = get_console()
if console:
    console.print_exception(show_locals=False)
else:
    traceback.print_exc()
```

#### L7: 改定ヘッダーの可読性改善

変更前:
```
╭─────────────────────────────────────╮
│  rev02_souzoku  正解ID: 8件  1/6   │
│  相続少額払い拡大に伴い、...           │
╰─────────────────────────────────────╯
```

変更後:
```
╭─────────────────────────────────────╮
│  [1/6] 改定② — 相続 (rev02)        │
│  正解ID: 8件                        │
│  相続少額払い拡大に伴い、...           │
╰─────────────────────────────────────╯
```

---

## 改善の優先順位

| 優先度 | フェーズ | 対象 | 推定効果 | リスク | 工数 |
|--------|---------|------|---------|--------|------|
| 1 | フェーズ1 | P1+P2+P6: キャッシュ導入 | 数十秒→数秒（UI）、24回→4回（バッチ） | 低 | 中 |
| 2 | フェーズ4 | P7: DB側フィルタ + バグ修正 | 11,439件→1,384件（88%削減）+ 正確性修正 | 低 | 小 |
| 3 | フェーズ2 | P3: LLMクエリキャッシュ | LLM呼び出し4回→1回 | 低 | 小 |
| 4 | フェーズ3A | P4: プロバイダー並列化 | 実行時間半減 | 中 | 中 |
| 5 | フェーズ3B | P5: LLM分析並列化 | 400秒→80秒 | 中 | 中 |
| 6 | ログ改善 | L1-L7: ターミナルログ可読性 | 顧客体験向上 | 低 | 中 |
| 7 | フェーズ5 | P8-P12: 細かい最適化 | 軽微 | 低 | 小 |

---

## 推定総合効果

### バッチ版（evaluate_revisions.py）

- 現状: 改定6件 x providers=both で推定 10-30分
- フェーズ1+2後: ChromaDB/Sudachi/LLMの重複排除 → 推定 5-15分
- フェーズ3後: 並列化 → 推定 3-8分

### UI版（eval_ui.py）

- 現状: 1クエリ hybrid providers=both で推定 30-120秒
- フェーズ1後: キャッシュ導入で2回目以降 → 推定 5-15秒
- フェーズ4後: source_filter DB側適用 → 推定 3-10秒
