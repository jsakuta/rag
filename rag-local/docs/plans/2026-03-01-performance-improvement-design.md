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
その後、3つの独立レビューエージェントにより設計の妥当性と既存機能への影響を検証した。

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
  - 注: Sudachi辞書は `_shared_tokenizer` でシングルトン共有済み。コストは辞書初期化ではなく `tokenize()` の形態素解析呼び出し
- LLM API: 2回 同一テキストで重複（P3）
- Embedding API: 4回（P10）
- エンジン初期化: 2セット（P6）

---

## ボトルネック一覧

### 深刻度: 高

#### P1: ChromaDB全件取得の繰り返し（キャッシュなし）

- バッチ: `evaluate_revisions.py` `_get_reference_queries` L251
- UI: `eval_ui.py` `_search_with_provider` L348
- 影響規模: 改定ごとにarea数が異なる（rev02は1area, rev03は複数area等）。providers=both の場合、全件取得は `Σ(各改定のarea数) x 2` 回発生
- `MetadataVectorDB` の `_client_cache` はクライアントのみキャッシュし、ドキュメントデータはキャッシュしない
- ただし ChromaDB `PersistentClient` 自体がファイルI/Oをバッファリングするため、2回目以降の `collection.get()` のコスト削減幅は実測が必要

#### P2: Sudachi全件形態素解析の繰り返し（キャッシュ破棄）

- `keyword_search_engine.py` `build_cache` L154-171
- `evaluate_revisions.py` `_create_orchestrator` L220-224
- `KeywordSearchEngine` が毎回 new されるため `_keyword_cache` が破棄される
  - さらに `build_cache` 内で `self._keyword_cache = {}` リセットもあり二重リセット
- 影響規模: `Σ(各改定のarea数) x provider数` 回 x 各コレクションのドキュメント数のSudachi呼び出し

#### P3: LLMクエリ拡張の重複呼び出し

- `multi_stage_orchestrator.py` `execute` L115-123
- `query_enhancer.enhance()` が area x provider 回呼ばれる（最大4回、同一テキスト）
- 1回あたり1-5秒 + リトライ時最大16秒の wait
- 前提: `query_enhancer.enhance` は改定内容テキストの言い換えを行うだけで、area情報は使用しない。従って同一 `revision_content` に対する結果は共有可能

#### P4: Azure/VertexAI検索の完全直列実行

- バッチ: `evaluate_revisions.py` L621-647
- UI: `eval_ui.py` `execute_dual_provider_search` L180-190
- 2つのプロバイダーは完全に独立（異なるDB、異なるAPIエンドポイント）
- 並列化で実行時間が半減可能
- 注意: バッチ版の `providers` 引数は `"azure"` / `"vertex"` / `"both"` を使用。UI版の `selected_providers` は `"azure_openai"` / `"vertex_ai"` / `"both"` を使用。**文字列値が異なる**

#### P5: LLM分析（judgment）の直列実行

- `evaluate_revisions.py` `_run_llm_analysis` L512-545
- MAX_RESULTS=100 の場合、Azure 100件 + VertexAI 100件 = 最大200回のLLM直列呼び出し
- 200回 x 1-3秒 = 3-10分
- `_evaluate_single_result` は `result` dict を in-place で書き換える（各スレッドが別 result を担当するため並列化は安全）

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
- **注意**: 改定別コレクション (`rev0X_XXX`) はシナリオのみで構成されるため `source_filter` が意味を持つのは影響調査モード（naibujimu/smile コレクション）のみ。評価モードでは効果なし

### 深刻度: 中

#### P8: `_merge_results` の O(N x M) 線形探索

- `multi_stage_orchestrator.py` L286-300
- `both_ids` の各要素に対して `original_results` と `llm_results` を線形探索
- max_results=100 では `|both_ids| * 100 = 最大10,000操作` と軽微
- `Original_Only` / `LLM_Only` の処理は既に O(N) で問題なし

#### P9: `_fetch_scenario_content` が未発見シナリオごとにChromaDBクエリ

- `evaluate_revisions.py` L129-159
- `_get_reference_queries` で既に全ドキュメントを取得済みなのに、同じDBに再度クエリ
- ただし `MetadataVectorDB._client_cache` でクライアントはキャッシュ済みのため、実際のコストは SQLite 1クエリ分（軽微）
- 正解発見率が低い改定初期で顕在化するが、効果は限定的

#### P10: embedding API が Stage1/Stage2 で2回直列呼び出し

- `vector_search_engine.py` `encode_query` L52
- Stage 1（原文）と Stage 2（LLMクエリ）でそれぞれ1回ずつ呼び出し
- Azure OpenAI の場合1リクエストあたり50-200ms
- LLMエラー時は `llm_query == query_text` となり同一テキストを2回エンコードする無駄が追加発生

#### P11: `ChromaDBKeywordSearcher._collection_cache` にLRU上限なし

- `chromadb_keyword_search.py` L65
- naibujimu全件ドキュメント（11,439件 x 平均200-400文字）がメモリに永久保持
- **ただし現状は毎回インスタンスが破棄されるため実害なし**
- **フェーズ1（`@st.cache_resource`）実施後に初めてこの問題が顕在化する**（依存関係あり）

#### P12: `text_combiner.parse()` の同一ドキュメント2回パース

- `multi_stage_orchestrator.py` `_build_result_data` L221
- Stage 1 と Stage 2 の `_execute_hybrid_search` で同一 doc_id が2回パースされる
- `text_combiner.parse()` は正規表現ベースのテキスト分割で軽量。深刻度は「低」寄り

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
  - **前提条件**: area 引数には `rev02_souzoku` 等の改定番号を含むフルパス名が入るため、
    (area, provider) だけで改定を区別できる。ただし将来 area 名が改定番号を含まない形式に
    変更された場合は silent corruption を起こす。安全策として `(revision, area, provider)` を
    キーにすることを推奨
- `_create_orchestrator`: `KeywordSearchEngine` と `build_cache` 結果をキャッシュ
  - キャッシュが有効な理由: keyword_cache は reference_queries のキャッシュであり、
    area が同じなら reference_queries も同一。改定が異なっても問題ない
  - 注意: オーケストレーターをキャッシュすると `query_enhancer` の内部状態（LLMオブジェクト）も
    共有される。`QueryEnhancer` は `__init__` でプロンプトを読むだけでステートレスなため問題ないが、
    将来のLLMクライアント変更時に注意

#### アプローチB: UI版 — `@st.cache_resource` 導入

```python
@st.cache_resource(ttl=3600)  # 1時間TTL: DB更新時のstaleデータ防止
def get_cached_engine(provider: str, area: str, db_path: str):
    """エンジン群をセッション横断でキャッシュ"""
    embedding_model = create_embedding_model(...)
    vector_db = MetadataVectorDB(...)
    keyword_engine = KeywordSearchEngine(...)
    # build_cache はここで1回だけ実行
    return embedding_model, vector_db, keyword_engine

@st.cache_resource(ttl=3600)
def get_cached_keyword_searcher(base_db_path: str):
    """ChromaDBKeywordSearcherをセッション横断でシングルトン化"""
    keyword_engine = KeywordSearchEngine(...)
    return ChromaDBKeywordSearcher(base_db_path=base_db_path, keyword_engine=keyword_engine, ...)
```

**キャッシュ無効化戦略**:
- `ttl=3600`（1時間）で自動無効化。`build_db.py` 実行後にStreamlit再起動せずとも1時間で更新
- サイドバーに「キャッシュクリア」ボタンを追加し `st.cache_resource.clear()` を呼ぶことで即座に更新可能
- 注意: 評価モード（area=`rev0X_XXX`）と影響調査モード（area=`naibujimu`）は `db_path` が異なるため自然に分離される
- 注意: `@st.cache_resource` は `@st.cache_data` と異なりオブジェクトをシリアライズせず参照を返すため、pickle不可オブジェクトでも安全

**推定効果**: 2回目以降のクエリで ChromaDB全件取得 + Sudachi解析をスキップ → 数十秒 → 数秒

#### 実装時の注意: source_filter パディングとキーワードキャッシュの整合性

UI版の `_search_with_provider` では source_filter 非マッチ文書を空文字でパディングして
`reference_queries` のインデックスを維持している。`build_cache` はこの空文字に対して
`set()` を返す。ベクトル検索結果の `doc_id` の数値部分が `reference_queries` のインデックスと
一致する前提に依存している。

**この前提はフェーズ4（DB側フィルタ）と組み合わせると崩れる可能性がある**:
- `collection.get(where={"source": ...})` を使うと返却ドキュメントの順序/件数が変わり、
  パディングのインデックスがベクトル検索の `doc_id` と対応しなくなる
- **対策**: フェーズ1とフェーズ4は独立して実装し、フェーズ4ではパディング方式を廃止して
  DB側フィルタに一本化する

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
- **前提条件**: `query_enhancer.enhance` は改定内容テキストの言い換えのみを行い、
  area 情報は入力に含まれない。プロンプト（`prompt/summarize_v1.0.txt`）を確認の上、
  area非依存であることを実装前に検証すること

**推定効果**: LLM API呼び出し 4回 → 1回（1改定あたり3-15秒短縮）

### フェーズ3: 並列化（大きな効果、中リスク）

対象: P4, P5

#### アプローチA: プロバイダー並列化（バッチ版のみ）

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
        # 片方がエラーでも他方の結果を保持する
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except Exception as e:
                logger.error(f"{key} 検索でエラー: {e}")
                results[key] = ({}, "", [], [])  # 空の結果で継続
```

**スレッドセーフティの確認事項**:
- ChromaDB `PersistentClient`: SQLite WALモードで読み取りはスレッドセーフ。評価時は読み取りのみ
- `MetadataVectorDB._client_cache`: `LRUCache` は `threading.Lock()` でガード済み（vector_db.py L58-59）
- Azure/VertexAI Embedding SDK: 各スレッドが異なるプロバイダーを使うため競合なし
- `KeywordSearchEngine._shared_tokenizer`: クラス変数だが `threading.Lock` でガード済み（L30-41）

**UI版への適用**: UI版は `selected_providers` の文字列値が異なる（`"azure_openai"` / `"vertex_ai"`）。
並列化を適用する場合は文字列値の変換が必要。ただしUI版では単一プロバイダー選択が主流のため優先度低

**推定効果**: 実行時間が約半分（Azure 30秒 + Vertex 30秒 → 並列で30秒）

#### アプローチB: LLM分析並列化

```python
def _run_llm_analysis(self, results, revision_content):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(self._evaluate_single_result, r, revision_content): i
            for i, r in enumerate(results)
        }
        # 全futureの完了を待つ（結果順序はリストのインデックスで保証）
        concurrent.futures.wait(futures)
        # 例外チェック
        for future in futures:
            if future.exception():
                logger.error(f"LLM分析エラー: {future.exception()}")
    return results  # in-place書き換え済み
```

- `_evaluate_single_result` は各 `result` dict を in-place で書き換え。各スレッドが別の result を
  担当するため dict への並行書き込みは発生しない
- `concurrent.futures.wait()` を使用し、`as_completed()` の順序非決定性を回避
- **レート制限の考慮**: max_workers=5 は Azure OpenAI のデフォルト RPM 制限（モデル・デプロイ依存）に
  対して安全側だが、プロジェクト固有のデプロイメント制限を事前確認すること
- **P4+P5同時適用時の注意**: プロバイダー並列化（2スレッド）+ LLM並列化（5スレッド）で最大10スレッドが
  同時にLLM APIを呼ぶ。ただし現行コードでは `_run_llm_analysis` は検索完了後に呼ばれるため、
  P4のスレッドプール完了後にP5のスレッドプールが起動する構造になり、同時10スレッドにはならない
- **推定効果**: 200件 x 2秒 = 400秒 → 200件/5並列 x 2秒 = 80秒（5倍速）

### フェーズ4: DB側フィルタの適用（正確性修正 + パフォーマンス改善）

対象: P7

#### 適用範囲の明確化

- **影響調査モード（naibujimu/smile コレクション）**: `source` フィールドが `scenario` / `history_data` で
  格納されており、DB側フィルタが有効。主な効果はここで発生（11,439件 → 1,384件）
- **評価モード（rev0X_XXX コレクション）**: シナリオのみで構成されるため `source_filter` は不要。
  DB側フィルタを適用しても効果なし
- **実装前確認**: `check_db_content.py` で改定別コレクションのメタデータに `source` フィールドが
  存在するかを確認すること

#### アプローチ

1. **バグ修正（最優先）**: `_search_with_provider` の結果 dict に `_source` キーを追加
   ```python
   all_results.append({
       ...,
       "_source": meta.get("source", "unknown"),  # 追加
       "_area": area,
   })
   ```
2. **影響調査モードのみ**: `collection.get(where={"source": source_filter})` でDB側フィルタ
   - ChromaDB の `get(where=...)` は内部的にSQLiteフルスキャンを行う。小規模コレクション（rev0X_XXX:
     数百件）では Python 側フィルタと大差ない可能性がある。naibujimu (11,439件) では効果大
3. **ベクトル検索の filter_metadata**: `orchestrator.execute(filter_metadata={"source": source_filter})`
   - `vector_db.py` の `ALLOWED_METADATA_KEYS` に `source` は含まれておりホワイトリスト通過
   - **重要な副作用**: `filter_metadata` 適用時、ベクトル検索結果の `doc_id` と
     `reference_queries` のインデックスの対応が崩れる可能性がある。
     `_execute_hybrid_search` の `original_idx = int(doc_id.split('_')[1])` が
     `keyword_cache` のインデックスと一致する前提の検証が必要
   - **安全策**: フェーズ4ではまず「バグ修正 + collection.get の where フィルタ」のみ実装し、
     ベクトル検索の filter_metadata は十分なテスト後に適用
4. **Python側フォールバックは残す**: DB側フィルタが万一機能しない場合（`source` フィールド欠落等）の
   二重防御として、Python側フィルタも残す

**推定効果**: 影響調査モードで全11,439件 → シナリオのみ1,384件（88%削減）

### フェーズ5: 細かい最適化

対象: P8, P9, P10, P12

- P8: `_merge_results` のdict化（`both_ids` 処理のみ O(N^2) → O(N)。Original_Only/LLM_Only は既に O(N)）
- P9: 効果が限定的（SQLite 1クエリ分のコスト）のため優先度最低
- P12: `text_combiner.parse()` は正規表現ベースで軽量。コスト対効果が低い

---

## ターミナルログ改善設計

### 方針

1. 顧客向けの言葉に追加・変換（既存ログ情報は削除しない）
2. `logger.info` と `print_*` は役割が異なる（ログファイル記録 vs 画面表示）ため**両方残す**
3. 処理中のフィードバック強化（ステップ表示。スピナーはログリダイレクト問題あり）
4. エリア名の日本語表示（既存 `BusinessAreaTranslator` を拡張）
5. 処理時間の表示（`print_completion` の `elapsed_time` パラメータを活用）

### 改善箇所一覧

#### L1: 評価設定の整理

**方針変更**: 設定情報を削除するのではなく、顧客向けの情報を上部に、技術詳細を下部に分離する。
改定ごとの `search_type` 切り分け（keyword_filter vs hybrid）は常に表示する（デバッグに必須）。

変更後:
```
─────────── 実行設定 ───────────
│ 検索プロバイダー: Azure + VertexAI
│ LLM判定: 無効
│ 評価対象: 6件の改定
│
│ 【検索タイプ】
│   キーワード検索: 相続 (rev02)
│   類似検索(hybrid): スマイル (rev01), 内部事務 (rev03), ...
│
│ 【詳細設定】 (--verbose で非表示化可能)
│   デフォルトベクトル重み: 0.7
│   フィルタモード: threshold
│   閾値 (Azure): 0.4 / (VertexAI): 0.35
```

`--verbose` フラグはバッチ版（argparse）のみ適用。UI版は影響なし（UI版は設定がサイドバーで可視）。

#### L2: logger.info の扱い（削除しない）

**レビュー結果**: `logger.info` と `print_*` は二重出力ではなく補完的な情報。
- `logger.info(f"  {area}: {len(results)}件取得")` → エリア単位の件数（ログファイルに記録）
- `print_search_result` → プロバイダー全体の合計（画面に表示）

**対応**: 両方残す。ただし画面表示の冗長感を減らすため、`logger.info` のログレベルを
`logger.debug` に変更し、通常実行時は画面に表示されないようにする。ログファイルには記録される。

#### L3: 検索中のフィードバック強化

**レビュー結果**: `console.status` スピナーは `force_terminal=True` の設定下でログリダイレクト時に
大量のANSIエスケープコードを生成する問題がある。

**対応**: スピナーではなく `print_status` でステップ表示に変更:
```python
print_status("Azure で検索中... (内部事務)", "info")
azure_results = self.search_revision_multi_stage(...)
print_search_result("azure", total_azure, azure_areas, azure_correct, len(correct_ids))
```

#### L4: エリア名の日本語変換

**レビュー結果**: `business_area_translator.py` + `business_areas.yaml` に既存の変換インフラがある（日本語→英語方向）。
新規に `AREA_DISPLAY_NAMES` 辞書を作ると二重管理になる。

**対応**: `BusinessAreaTranslator` に逆引きメソッド `display_name(area: str) -> str` を追加する。
```python
# business_area_translator.py に追加
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

既存の `print_table` によるDB状態一覧表示は維持し、MISSINGがある場合に**追加で**
ガイドメッセージを表示する（テーブルを置き換えない）:
```python
# 既存のテーブル表示はそのまま
print_table("ベクトルDB状態", ...)

# MISSINGがある場合のみ追加表示
if missing_dbs:
    print_status(
        "build_db.py --revisions-only を実行してDBを構築してください",
        "warning"
    )
```

#### L6: traceback の Rich Console 経由化

**対象箇所（全3箇所）**:
- `evaluate_revisions.py` L241（`_create_orchestrator` 内）
- `evaluate_revisions.py` L426（`search_revision_multi_stage` 内）
- `eval_ui.py` L416

```python
# 変更後（全箇所共通）
console = get_console()
if console:
    console.print_exception(show_locals=False)
else:
    traceback.print_exc()
```

Rich は既存依存（`logger.py` L15-25 で `try/except ImportError` ガード済み）のため追加リスクなし。

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
│  [1/6] 相続 (rev02_souzoku)        │
│  正解ID: 8件                        │
│  相続少額払い拡大に伴い、...           │
╰─────────────────────────────────────╯
```

- `get_display_name()` でエリア名を日本語化
- `rev02` → 「改定②」の丸数字変換は、台帳番号との対応が複雑（rev03 が複数 area に跨がる等）
  なため見送り。内部名 `rev02_souzoku` はそのまま括弧内に表示

#### L8: 処理時間の表示（新規追加）

**レビューで発見された既存欠落**: `print_completion` は `elapsed_time` パラメータを持つ（`logger.py` L315）が、
`main()` L1173 では渡されていない。

```python
# main() に追加
import time
start_time = time.time()
# ... 評価処理 ...
elapsed = time.time() - start_time
print_completion(str(output_file), elapsed_time=elapsed)
```

---

## 改善の優先順位

| 優先度 | フェーズ | 対象 | 推定効果 | リスク | 工数 |
|--------|---------|------|---------|--------|------|
| 1 | フェーズ1 | P1+P2+P6: キャッシュ導入 | UI: 2回目以降 数秒、バッチ: 大幅削減 | 低（TTL/キー設計に注意） | 中 |
| 2 | フェーズ4 | P7: バグ修正 + DB側フィルタ | 影響調査モードで88%削減 + 正確性修正 | 低（段階的適用） | 小 |
| 3 | フェーズ2 | P3: LLMクエリキャッシュ | LLM呼び出し 4回→1回 | 低（プロンプト確認要） | 小 |
| 4 | フェーズ3A | P4: プロバイダー並列化 | 実行時間半減 | 中（スレッドセーフ確認要） | 中 |
| 5 | フェーズ3B | P5: LLM分析並列化 | 400秒→80秒 | 中（レート制限確認要） | 中 |
| 6 | ログ改善 | L1-L8: ターミナルログ可読性 | 顧客体験向上 | 低 | 中 |
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
- フェーズ4後: source_filter DB側適用（影響調査モード） → 推定 3-10秒

---

## レビュー指摘事項の対応履歴

### レビュー1: キャッシュ導入（フェーズ1-2）

| 指摘 | 対応 |
|------|------|
| キャッシュキー `(area, provider)` に revision 未含の理由が不明 | area にrev番号が含まれる前提を明記。安全策として `(revision, area, provider)` を推奨 |
| `@st.cache_resource` の TTL なし → staleデータリスク | `ttl=3600` を追加。キャッシュクリアボタンも追加 |
| 「24回の全件取得」の数値が不正確 | 改定ごとにarea数が異なることを明記。具体的数値を式に変更 |
| P3 LLMクエリのarea非依存前提が未記載 | 前提条件として明記。実装前にプロンプト確認を指示 |
| source_filter パディングと keyword_cache インデックスの整合性未検証 | フェーズ1と4の依存関係として注意事項を追記 |
| P2 Sudachi辞書初期化と形態素解析の区別が曖昧 | 注記を追加（辞書はシングルトン、コストは tokenize() 呼び出し） |

### レビュー2: 並列化 + フィルタ（フェーズ3-5）

| 指摘 | 対応 |
|------|------|
| バッチ/UIで providers 文字列値が異なる | P4 に注意事項として明記。並列化はバッチ版のみに限定 |
| ChromaDB スレッドセーフティ未検証 | 検証結果（WALモード読み取り安全、_client_cache はLock済み）を記載 |
| P5 in-place書き換えの説明不足 | 各スレッドが別resultを担当する旨を明記 |
| 改定別コレクションに source フィールドがあるか未確認 | 適用範囲を明確化（影響調査モードのみ効果あり） |
| filter_metadata とキーワードキャッシュの整合性リスク | 段階的適用を推奨（まずバグ修正 + collection.get where のみ） |
| 並列化のエラーハンドリングなし | try/except を追加。片方エラーでも他方の結果を保持 |
| `as_completed()` の順序非決定性 | `concurrent.futures.wait()` に変更 |
| P4+P5同時適用時の最大同時スレッド数 | 構造上同時にはならないことを説明 |
| P9 の効果が過大評価 | 「軽微」に修正 |
| P11 がフェーズ1実施後に初めて顕在化する依存関係 | 依存関係を明記 |
| P12 の深刻度が過大評価 | 「低寄り」に修正 |

### レビュー3: ターミナルログ（L1-L7）

| 指摘 | 対応 |
|------|------|
| L2「二重出力」診断が誤り。logger.info と print_* は補完的 | 方針変更: 両方残す。logger.info を logger.debug に変更 |
| L5の設計が既存 print_table フローを置き換えている | テーブルは維持し、MISSINGガイドを追加表示に変更 |
| L4 AREA_DISPLAY_NAMES が BusinessAreaTranslator と重複 | BusinessAreaTranslator に逆引きメソッドを追加する方針に変更 |
| L7「改定②」丸数字変換のロジック未記述 | 複雑なため見送り。内部名をそのまま括弧内に表示 |
| L3 スピナーが force_terminal=True でログリダイレクト問題 | スピナーではなく print_status でステップ表示に変更 |
| L1 --verbose がUI版に適用できない | バッチ版のみに適用。UI版はサイドバーで制御 |
| traceback.print_exc() の対象が3箇所（1箇所のみ記載） | 全3箇所を明記 |
| print_completion の elapsed_time が main() で渡されていない | L8 として新規追加 |
