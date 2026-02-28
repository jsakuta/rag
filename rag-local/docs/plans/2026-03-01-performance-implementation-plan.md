# 運用保守効率化AI パフォーマンス改善 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** revision-eval の検索パフォーマンスを改善し、ターミナルログの可読性を向上させる

**Architecture:** キャッシュ導入（フェーズ1-2）→ 並列化（フェーズ3）→ DB側フィルタ（フェーズ4）→ ログ改善の順で段階的に実装。各フェーズは独立してコミット可能。既存テストがない（evaluate_revisions.py / eval_ui.py）ため、変更箇所にユニットテストを追加しつつ進める。

**Tech Stack:** Python 3.11, Streamlit, ChromaDB, LangChain (ChatVertexAI/ChatOpenAI), Sudachi, Rich, concurrent.futures

**設計書:** `docs/plans/2026-03-01-performance-improvement-design.md`

---

### Task 1: バッチ版キャッシュ — `_get_reference_queries` のメモ化

**対象:** P1（ChromaDB全件取得の繰り返し）

**Files:**
- Modify: `apps/revision-eval/evaluate_revisions.py:106-117` (`__init__`)
- Modify: `apps/revision-eval/evaluate_revisions.py:244-264` (`_get_reference_queries`)
- Test: `tests/unit/test_evaluate_revisions_cache.py` (新規作成)

**Step 1: テストファイルを作成**

```python
# tests/unit/test_evaluate_revisions_cache.py
"""evaluate_revisions.py のキャッシュ機構テスト"""
import pytest
from unittest.mock import MagicMock, patch


class TestReferenceQueriesCache:
    """_get_reference_queries のメモ化テスト"""

    def _make_evaluator(self):
        """テスト用の RevisionEvaluator を最小構成で作成"""
        with patch("apps.revision_eval.evaluate_revisions.create_llm") as mock_llm, \
             patch("apps.revision_eval.evaluate_revisions.JudgmentSupport"):
            mock_llm.return_value = MagicMock()
            from apps.revision_eval.evaluate_revisions import RevisionEvaluator
            config = MagicMock()
            config.STOP_WORDS = ("の", "は", "が")
            config.POSITION_WEIGHT = 1.2
            evaluator = RevisionEvaluator(config, enable_llm_analysis=False)
            return evaluator

    def test_cache_hit_returns_same_result(self):
        """同じ (area, provider) の2回目呼び出しはキャッシュを返す"""
        evaluator = self._make_evaluator()
        # キャッシュに直接データを入れる
        evaluator._reference_queries_cache[("rev02_souzoku", "azure_openai")] = ["q1", "q2"]

        result = evaluator._get_reference_queries("rev02_souzoku", "azure_openai")
        assert result == ["q1", "q2"]

    def test_cache_miss_calls_chromadb(self):
        """キャッシュにない場合はChromaDBを呼ぶ"""
        evaluator = self._make_evaluator()
        assert ("rev02_souzoku", "azure_openai") not in evaluator._reference_queries_cache
```

**Step 2: テストが失敗することを確認**

Run: `cd rag-local && python -m pytest tests/unit/test_evaluate_revisions_cache.py -v`
Expected: FAIL (`_reference_queries_cache` 属性が存在しない)

**Step 3: `__init__` にキャッシュ変数を追加**

`evaluate_revisions.py` L117 の直後に追加:

```python
        # パフォーマンス改善: キャッシュ
        self._reference_queries_cache: Dict[Tuple[str, str], List[str]] = {}
```

**Step 4: `_get_reference_queries` にキャッシュロジックを追加**

`evaluate_revisions.py` L244-264 を変更:

```python
    def _get_reference_queries(self, area: str, provider: str) -> List[str]:
        # キャッシュチェック
        cache_key = (area, provider)
        if cache_key in self._reference_queries_cache:
            return self._reference_queries_cache[cache_key]

        # 既存のロジック（L246-264 はそのまま維持）
        queries = []
        db_path = VECTOR_DB_BASE / area / provider
        # ... 既存コード ...

        # キャッシュに保存
        self._reference_queries_cache[cache_key] = queries
        return queries
```

**Step 5: テストが通ることを確認**

Run: `cd rag-local && python -m pytest tests/unit/test_evaluate_revisions_cache.py -v`
Expected: PASS

**Step 6: コミット**

```bash
git add tests/unit/test_evaluate_revisions_cache.py apps/revision-eval/evaluate_revisions.py
git commit -m "perf: _get_reference_queries にメモ化キャッシュ導入 (P1)"
```

---

### Task 2: バッチ版キャッシュ — `_create_orchestrator` のメモ化

**対象:** P2（Sudachi全件形態素解析の繰り返し）

**Files:**
- Modify: `apps/revision-eval/evaluate_revisions.py:106-117` (`__init__`)
- Modify: `apps/revision-eval/evaluate_revisions.py:195-242` (`_create_orchestrator`)

**Step 1: `__init__` にオーケストレーターキャッシュを追加**

L117 直後に追加（Task 1 で追加した行の後）:

```python
        self._orchestrator_cache: Dict[Tuple[str, str, float], MultiStageOrchestrator] = {}
```

**Step 2: `_create_orchestrator` にキャッシュロジックを追加**

`evaluate_revisions.py` L195 の関数先頭:

```python
    def _create_orchestrator(
        self, provider: str, area: str, reference_queries: List[str], vector_weight: float,
    ) -> Optional[MultiStageOrchestrator]:
        # キャッシュチェック（area にrev番号が含まれるため provider+area+weight で一意）
        cache_key = (area, provider, vector_weight)
        if cache_key in self._orchestrator_cache:
            return self._orchestrator_cache[cache_key]

        # 既存のロジック（L198-240 はそのまま維持）
        try:
            # ... 既存コード ...
            orchestrator = MultiStageOrchestrator(...)
            # キャッシュに保存
            self._orchestrator_cache[cache_key] = orchestrator
            return orchestrator
        except Exception as e:
            # ... 既存のエラー処理 ...
```

**Step 3: テストを追加して確認**

`test_evaluate_revisions_cache.py` に追加:

```python
    def test_orchestrator_cache_hit(self):
        """同じ (area, provider, weight) の2回目はキャッシュを返す"""
        evaluator = self._make_evaluator()
        mock_orch = MagicMock()
        evaluator._orchestrator_cache[("rev02_souzoku", "azure_openai", 0.9)] = mock_orch

        result = evaluator._create_orchestrator("azure_openai", "rev02_souzoku", ["q1"], 0.9)
        assert result is mock_orch
```

**Step 4: テスト実行**

Run: `cd rag-local && python -m pytest tests/unit/test_evaluate_revisions_cache.py -v`
Expected: PASS

**Step 5: コミット**

```bash
git add apps/revision-eval/evaluate_revisions.py tests/unit/test_evaluate_revisions_cache.py
git commit -m "perf: _create_orchestrator にメモ化キャッシュ導入 (P2)"
```

---

### Task 3: UI版キャッシュ — `@st.cache_resource` 導入

**対象:** P6（エンジン群の毎回再構築）

**Files:**
- Modify: `apps/revision-eval/ui/eval_ui.py:311-419` (`_search_with_provider`)
- Modify: `apps/revision-eval/ui/eval_ui.py:231-308` (`_execute_keyword_filter_search`, `_execute_impact_keyword_search`)

**Step 1: キャッシュ関数を `eval_ui.py` に追加**

ファイル上部（`initialize_session_state` の前あたり、L140付近）に追加:

```python
@st.cache_resource(ttl=3600)
def _get_cached_hybrid_engine(provider: str, db_path: str, source_filter_key: str):
    """hybrid検索用エンジン群をキャッシュ（TTL=1時間）"""
    from src.core.search.vector_search_engine import VectorSearchEngine
    from src.core.search.keyword_search_engine import KeywordSearchEngine
    from src.core.search.text_combiner import get_text_combiner

    provider_config = copy.copy(st.session_state.config)
    if provider == "azure_openai":
        provider_config.embedding_provider = "azure_openai"
        provider_config.embedding_model = "text-embedding-3-large"
    else:
        provider_config.embedding_provider = "vertex_ai"
        provider_config.embedding_model = "text-multilingual-embedding-002"

    embedding_model = create_embedding_model(provider_config)
    vector_db = MetadataVectorDB(db_path=db_path, collection_name="default")
    text_combiner = get_text_combiner()

    # reference_queries を構築
    result = vector_db.collection.get(include=["documents"])
    documents = result.get("documents", [])
    reference_queries = []
    for doc in documents:
        if doc:
            parsed = text_combiner.parse(doc)
            reference_queries.append(parsed.query if parsed.query else doc[:100])
        else:
            reference_queries.append("")

    keyword_engine = KeywordSearchEngine(
        stop_words=st.session_state.config.STOP_WORDS,
        position_weight=st.session_state.config.POSITION_WEIGHT,
    )
    keyword_engine.build_cache(reference_queries)

    vector_engine = VectorSearchEngine(
        embedding_model=embedding_model,
        vector_db=vector_db,
        reference_queries=reference_queries,
    )

    return {
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "keyword_engine": keyword_engine,
        "vector_engine": vector_engine,
        "reference_queries": reference_queries,
    }


@st.cache_resource(ttl=3600)
def _get_cached_keyword_searcher():
    """ChromaDBKeywordSearcher をキャッシュ（TTL=1時間）"""
    from src.core.search.keyword_search_engine import KeywordSearchEngine
    from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher

    keyword_engine = KeywordSearchEngine(
        stop_words=st.session_state.config.STOP_WORDS,
        position_weight=st.session_state.config.POSITION_WEIGHT,
    )
    return ChromaDBKeywordSearcher(
        base_db_path=str(VECTOR_DB_BASE),
        keyword_engine=keyword_engine,
        area_to_bot=AREA_TO_BOT,
        area_to_category=AREA_TO_CATEGORY,
    )
```

**Step 2: `_search_with_provider` をキャッシュ利用に書き換え**

L325-393 のエンジン構築部分を、キャッシュ関数の呼び出しに置換する。
`source_filter` がある場合のパディングロジックは**そのまま維持**（フェーズ4で対応）。

**Step 3: `_execute_keyword_filter_search` と `_execute_impact_keyword_search` をキャッシュ利用に書き換え**

毎回 `ChromaDBKeywordSearcher()` を new している箇所を `_get_cached_keyword_searcher()` に置換。

**Step 4: サイドバーにキャッシュクリアボタンを追加**

`run_streamlit_ui` のサイドバー末尾に追加:

```python
if st.button("キャッシュクリア", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()
```

**Step 5: 手動テスト**

Streamlit UIを起動し、以下を確認:
1. 1回目のクエリ: 従来通りの速度（キャッシュ構築）
2. 2回目のクエリ: 大幅に高速化（キャッシュヒット）
3. キャッシュクリアボタン: 押下後に再構築される

**Step 6: コミット**

```bash
git add apps/revision-eval/ui/eval_ui.py
git commit -m "perf: UI版に @st.cache_resource キャッシュ導入 (P6)"
```

---

### Task 4: LLMクエリ拡張のキャッシュ

**対象:** P3（同一テキストで area x provider 回の LLM 重複呼び出し）

**Files:**
- Modify: `src/core/search/multi_stage_orchestrator.py:82-139` (`execute`)
- Modify: `apps/revision-eval/evaluate_revisions.py:547-650` (`evaluate_revision`)
- Test: `tests/unit/test_evaluate_revisions_cache.py` (追記)

**Step 1: `MultiStageOrchestrator.execute` に `pre_enhanced_query` パラメータを追加**

`multi_stage_orchestrator.py` L82:

```python
    def execute(
        self,
        input_number: str,
        query_text: str,
        original_answer: str,
        filter_metadata: Dict[str, str] = None,
        pre_enhanced_query: Optional[str] = None,  # 追加
    ) -> List[MultiStageSearchResultDict]:
```

L114-123 の LLM 呼び出し部分を条件分岐:

```python
        # Stage 2: LLMクエリ検索
        if pre_enhanced_query:
            llm_query = pre_enhanced_query
        else:
            try:
                llm_query = self.query_enhancer.enhance(query_text)
            except Exception as e:
                logger.warning(f"LLMクエリ拡張失敗、原文で続行: {e}")
                llm_query = query_text
```

**Step 2: `evaluate_revision` で事前に1回だけ LLM クエリを生成**

`evaluate_revisions.py` の `evaluate_revision` メソッド内、hybrid検索の前（L615付近）に追加:

```python
        # LLMクエリ拡張を1回だけ実行（area/provider間で共有）
        pre_enhanced_query = None
        if search_type != "keyword_filter":
            try:
                pre_enhanced_query = self.llm_query_enhancer_cache.get(revision_content)
                if pre_enhanced_query is None:
                    from src.core.search.query_enhancer import QueryEnhancer
                    temp_enhancer = QueryEnhancer(llm=self.llm, prompt_path=str(PROMPT_PATH / "summarize_v1.0.txt"))
                    pre_enhanced_query = temp_enhancer.enhance(revision_content)
                    self.llm_query_enhancer_cache[revision_content] = pre_enhanced_query
            except Exception as e:
                logger.warning(f"LLMクエリ事前拡張失敗: {e}")
```

`__init__` にキャッシュ変数を追加:

```python
        self.llm_query_enhancer_cache: Dict[str, str] = {}
```

**Step 3: `search_revision_multi_stage` に `pre_enhanced_query` を渡す**

`evaluate_revisions.py` のAzure/VertexAI呼び出し箇所（L622-646）と `orchestrator.execute()` 呼び出し（L403-408）に `pre_enhanced_query` を渡す。

`search_revision_multi_stage` のシグネチャを拡張:

```python
    def search_revision_multi_stage(
        self, revision: str, query: str, correct_ids: List[str], provider: str,
        pre_enhanced_query: Optional[str] = None,  # 追加
    ) -> Tuple[...]:
```

L403-408 の `orchestrator.execute()` 呼び出し:

```python
            results = orchestrator.execute(
                input_number=revision,
                query_text=query,
                original_answer="",
                filter_metadata=None,
                pre_enhanced_query=pre_enhanced_query,  # 追加
            )
```

**Step 4: テスト実行**

Run: `cd rag-local && python -m pytest tests/unit/test_evaluate_revisions_cache.py -v`
Expected: PASS

**Step 5: コミット**

```bash
git add src/core/search/multi_stage_orchestrator.py apps/revision-eval/evaluate_revisions.py tests/unit/test_evaluate_revisions_cache.py
git commit -m "perf: LLMクエリ拡張を改定単位で1回に集約 (P3)"
```

---

### Task 5: プロバイダー並列化（バッチ版）

**対象:** P4（Azure/VertexAI の完全直列実行）

**Files:**
- Modify: `apps/revision-eval/evaluate_revisions.py:600-650` (`evaluate_revision`)

**Step 1: `evaluate_revision` の hybrid 検索部分を並列化**

`evaluate_revisions.py` L615-647 を置換:

```python
        # 類似検索（hybrid）の場合
        import concurrent.futures

        azure_results_by_area, azure_areas = {}, []
        vertex_results_by_area, vertex_areas = {}, []
        llm_query = pre_enhanced_query or ""
        keywords = []

        def _search_provider(prov_key, prov_name):
            return self.search_revision_multi_stage(
                revision, revision_content, correct_ids, prov_name,
                pre_enhanced_query=pre_enhanced_query,
            )

        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            if providers in ("both", "azure"):
                futures["azure"] = executor.submit(_search_provider, "azure", "azure_openai")
            if providers in ("both", "vertex"):
                futures["vertex"] = executor.submit(_search_provider, "vertex", "vertex_ai")

            for key, future in futures.items():
                try:
                    results_by_area, q, kw, areas = future.result()
                    if key == "azure":
                        azure_results_by_area, azure_areas = results_by_area, areas
                        if q: llm_query = q
                        if kw: keywords = kw
                        total_azure, azure_correct = self._count_correct_in_results_by_area(results_by_area)
                        print_search_result("azure", total_azure, areas, azure_correct, len(correct_ids))
                    else:
                        vertex_results_by_area, vertex_areas = results_by_area, areas
                        if not llm_query and q: llm_query = q
                        if not keywords and kw: keywords = kw
                        total_vertex, vertex_correct = self._count_correct_in_results_by_area(results_by_area)
                        print_search_result("vertex", total_vertex, areas, vertex_correct, len(correct_ids))
                except Exception as e:
                    logger.error(f"{key} 検索でエラー: {e}")
```

**Step 2: 手動テスト**

```bash
cd rag-local && python apps/revision-eval/evaluate_revisions.py --provider both
```

providers=both で実行し、従来より高速化されていることを確認。結果の正確性は Excel 出力を比較。

**Step 3: コミット**

```bash
git add apps/revision-eval/evaluate_revisions.py
git commit -m "perf: Azure/VertexAI検索をThreadPoolExecutorで並列化 (P4)"
```

---

### Task 6: LLM分析並列化

**対象:** P5（judgment の直列実行）

**Files:**
- Modify: `apps/revision-eval/evaluate_revisions.py:512-545` (`_run_llm_analysis`)

**Step 1: `_run_llm_analysis` を並列化**

```python
    def _run_llm_analysis(self, results: List[Dict], revision_content: str) -> List[Dict]:
        if not self.enable_llm_analysis or self.judgment_support is None:
            for r in results:
                r["関連性判定"] = ""
                r["判定根拠"] = ""
            return results

        import concurrent.futures
        console = get_console()

        if console:
            from rich.progress import Progress
            with Progress(console=console) as progress:
                task = progress.add_task("LLM分析中...", total=len(results))
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(self._evaluate_single_result, r, revision_content): r
                        for r in results
                    }
                    for future in concurrent.futures.as_completed(futures):
                        future.result()  # 例外があればここで再raise
                        progress.update(task, advance=1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(self._evaluate_single_result, r, revision_content)
                    for r in results
                ]
                concurrent.futures.wait(futures)
                # 例外チェック
                for f in futures:
                    if f.exception():
                        logger.error(f"LLM分析エラー: {f.exception()}")

        analyzed = sum(1 for r in results if r.get("関連性判定"))
        print_status(f"LLM分析完了: {analyzed}/{len(results)}件", "success")
        return results
```

**Step 2: 手動テスト**

```bash
cd rag-local && ENABLE_LLM_ANALYSIS=true python apps/revision-eval/evaluate_revisions.py --provider azure
```

LLM分析が並列で実行され、プログレスバーが更新されることを確認。

**Step 3: コミット**

```bash
git add apps/revision-eval/evaluate_revisions.py
git commit -m "perf: LLM分析をThreadPoolExecutor(max_workers=5)で並列化 (P5)"
```

---

### Task 7: P7 バグ修正 — `_source` キー追加 + Python側フィルタ修復

**対象:** P7（source_filter が実質無効なバグ）

**Files:**
- Modify: `apps/revision-eval/ui/eval_ui.py:394-419` (`_search_with_provider` 結果dict)

**Step 1: `_search_with_provider` の結果 dict に `_source` キーを追加**

L401-411 の `all_results.append()` を修正。ただし現状のコードでは `orchestrator.execute` の戻り値に `source` メタデータが含まれていない。

`multi_stage_orchestrator.py` の `_build_result_data`（L205-249）で `metadata.get("source", "")` を返却 dict に追加:

```python
        result_data = {
            # ... 既存のキー ...
            "_source": search_result.get("metadata", {}).get("source", "unknown"),  # 追加
        }
```

`eval_ui.py` L401-411:

```python
        all_results.append({
            "Similarity": r.get("Similarity", 0),
            "Search_Result_Q": r.get("Search_Result_Q", ""),
            "Search_Result_A": r.get("Search_Result_A", ""),
            "Search_Category": r.get("Search_Category", ""),
            "Sheet_Name": r.get("Sheet_Name", ""),
            "Row_Index": r.get("Row_Index", ""),
            "Search_Query": r.get("Search_Query", ""),
            "_area": area,
            "_source": r.get("_source", "unknown"),  # 追加
        })
```

**Step 2: 影響調査モードの `collection.get` に `where` フィルタを追加**

`eval_ui.py` の `_search_with_provider` L347-352 を変更:

```python
            get_kwargs = {"include": ["documents"]}
            if source_filter:
                get_kwargs["include"].append("metadatas")
                get_kwargs["where"] = {"source": source_filter}
            result = vector_db.collection.get(**get_kwargs)
```

**注意**: `where` フィルタを使うと `source_filter` 非マッチのドキュメントが返却されなくなるため、
パディング（空文字挿入）ロジックは不要になる。ただしキーワードキャッシュのインデックスとの
整合性を維持するため、`where` フィルタ使用時は `reference_queries` を返却ドキュメントのみから
構築し、`build_cache` も返却ドキュメントのみに対して実行する。

`for idx, doc in enumerate(documents)` のパディングロジックを条件分岐:

```python
            documents = result.get("documents", [])
            reference_queries = []
            for doc in documents:
                if doc:
                    parsed = text_combiner.parse(doc)
                    reference_queries.append(parsed.query if parsed.query else doc[:100])
                else:
                    reference_queries.append("")
```

source_filter 使用時はパディング不要になるため `metadatas[idx].get("source")` のチェックは削除。

**Step 3: 手動テスト**

Streamlit UIを起動し、影響調査モードで以下を確認:
1. データソース「シナリオ」選択 → シナリオのみが返される（FAQが混入しない）
2. データソース「FAQ」選択 → FAQのみが返される
3. hybrid検索でも source_filter が機能する

**Step 4: コミット**

```bash
git add apps/revision-eval/ui/eval_ui.py src/core/search/multi_stage_orchestrator.py
git commit -m "fix: hybrid検索のsource_filterバグ修正 + DB側フィルタ導入 (P7)"
```

---

### Task 8: ターミナルログ改善 — エリア名日本語化 + 処理時間表示

**対象:** L4（エリア名変換）, L8（処理時間表示）

**Files:**
- Modify: `src/utils/business_area_translator.py` (逆引きメソッド追加)
- Modify: `apps/revision-eval/evaluate_revisions.py:1107-1177` (`main`)
- Test: `tests/unit/test_business_area_mapping.py` (追記)

**Step 1: `business_area_translator.py` に `get_display_name` 関数を追加**

ファイル末尾に追加:

```python
# エリア名の日本語表示用マッピング
_AREA_DISPLAY_NAMES = {
    "naibujimu": "内部事務",
    "smile": "スマイル",
    "souzoku": "相続",
    "torikaku": "取引時確認",
}


def get_display_name(area: str) -> str:
    """内部エリア名を日本語表示名に変換する。

    例: "rev03_naibujimu" -> "内部事務", "naibujimu" -> "内部事務"
    マッチしない場合はそのまま返す。
    """
    for key, name in _AREA_DISPLAY_NAMES.items():
        if key in area:
            return name
    return area
```

**Step 2: テスト追加**

`tests/unit/test_business_area_mapping.py` に追加:

```python
from src.utils.business_area_translator import get_display_name


class TestGetDisplayName:
    def test_rev_prefix(self):
        assert get_display_name("rev03_naibujimu") == "内部事務"

    def test_plain_name(self):
        assert get_display_name("smile") == "スマイル"

    def test_unknown_returns_as_is(self):
        assert get_display_name("unknown_area") == "unknown_area"

    def test_souzoku(self):
        assert get_display_name("rev02_souzoku") == "相続"
```

**Step 3: テスト実行**

Run: `cd rag-local && python -m pytest tests/unit/test_business_area_mapping.py::TestGetDisplayName -v`
Expected: PASS

**Step 4: `main()` に処理時間計測を追加**

`evaluate_revisions.py` の `main()` 冒頭（L1117付近）に追加:

```python
    import time
    start_time = time.time()
```

`main()` 末尾（L1173）を変更:

```python
    elapsed = time.time() - start_time
    print_completion(str(output_file), elapsed_time=elapsed)
```

**Step 5: `print_search_result` と `print_revision_header` でエリア名変換を適用**

`logger.py` の `print_search_result`（L273）で `areas` を表示する箇所を変更:

```python
    from src.utils.business_area_translator import get_display_name
    areas_display = [get_display_name(a) for a in areas] if areas else ["-"]
    areas_str = ", ".join(areas_display)
```

**Step 6: コミット**

```bash
git add src/utils/business_area_translator.py tests/unit/test_business_area_mapping.py apps/revision-eval/evaluate_revisions.py src/utils/logger.py
git commit -m "feat: エリア名日本語化 + 処理時間表示 (L4, L8)"
```

---

### Task 9: ターミナルログ改善 — 設定表示整理 + フィードバック強化

**対象:** L1（設定簡素化）, L2（logger.debug化）, L3（ステップ表示）, L5（DB MISSINGガイド）, L6（traceback統一）

**Files:**
- Modify: `apps/revision-eval/evaluate_revisions.py:1107-1177` (`main`)
- Modify: `apps/revision-eval/evaluate_revisions.py:195-242, 392-428` (traceback箇所)
- Modify: `apps/revision-eval/ui/eval_ui.py:416` (traceback箇所)

**Step 1: `main()` の評価設定表示を整理**

`evaluate_revisions.py` L1141-1168 を変更。顧客向けサマリーを上部、技術詳細を `--verbose` 時に:

```python
    # argparse に --verbose を追加（L1109付近）
    parser.add_argument("--verbose", action="store_true", help="詳細設定を表示")

    # L1141-1168 を置換
    print_section("実行設定")
    provider_labels = {"both": "Azure + VertexAI", "azure": "Azure のみ", "vertex": "VertexAI のみ"}
    print_status(f"検索プロバイダー: {provider_labels[args.provider]}", "info")
    print_status(f"LLM判定: {'有効' if enable_llm else '無効'}", "info")
    print_status(f"評価対象: {total_revisions}件の改定", "info")

    # 検索タイプの一覧（改定ごとの設定は常に表示）
    keyword_revisions = [f"{get_display_name(r)} ({r})" for r, st in REVISION_SEARCH_TYPES.items() if st == "keyword_filter"]
    hybrid_revisions = [f"{get_display_name(r)} ({r})" for r, st in REVISION_SEARCH_TYPES.items() if st != "keyword_filter"]
    if keyword_revisions:
        print_status(f"キーワード検索: {', '.join(keyword_revisions)}", "info")
    if hybrid_revisions:
        print_status(f"類似検索(hybrid): {', '.join(hybrid_revisions)}", "info")

    if args.verbose:
        print_status(f"ベクトル重み: {DEFAULT_VECTOR_WEIGHT}", "info")
        print_status(f"フィルタモード: {FILTER_MODE}", "info")
        # ... 詳細パラメータ ...
```

**Step 2: logger.info をエリア件数で debug に変更**

`evaluate_revisions.py` L423:

```python
    # 変更前
    logger.info(f"  {area}: {len(results)}件取得")
    # 変更後
    logger.debug(f"  {area}: {len(results)}件取得")
```

**Step 3: DB MISSING時にガイドメッセージを追加**

`main()` の DB確認テーブル表示後（L1129付近）に追加:

```python
    if any("MISSING" in str(row) for row in db_status_data):
        print_status(
            "MISSINGのDBがあります。build_db.py --revisions-only を実行してください",
            "warning",
        )
```

**Step 4: 検索中のステップ表示を追加**

`evaluate_revision` の Azure/VertexAI 検索前に `print_status` を追加:

```python
    if providers in ("both", "azure"):
        print_status("Azure で検索中...", "info")
        # ... 検索 ...
```

**Step 5: traceback を Rich Console 経由に統一（3箇所）**

`evaluate_revisions.py` L241, L426 と `eval_ui.py` L416:

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

**Step 6: 手動テスト**

```bash
cd rag-local && python apps/revision-eval/evaluate_revisions.py --provider azure
cd rag-local && python apps/revision-eval/evaluate_revisions.py --provider azure --verbose
```

通常実行で簡潔な出力、`--verbose` で詳細出力になることを確認。

**Step 7: コミット**

```bash
git add apps/revision-eval/evaluate_revisions.py apps/revision-eval/ui/eval_ui.py src/utils/logger.py
git commit -m "feat: ターミナルログ可読性改善 (L1-L6)"
```

---

## Task 依存関係

```
Task 1 (P1 キャッシュ) ─┐
Task 2 (P2 キャッシュ) ─┤
                        ├─ Task 4 (P3 LLMキャッシュ) ─── Task 5 (P4 並列化) ─── Task 6 (P5 LLM並列化)
Task 3 (P6 UIキャッシュ) ┘
                                                        Task 7 (P7 バグ修正) ← 独立
Task 8 (L4,L8 ログ) ──── Task 9 (L1-L6 ログ)         ← 独立
```

- Task 1-3: 並列実行可能（独立した改善）
- Task 4: Task 1-2 の後（キャッシュキーに依存）
- Task 5: Task 4 の後（pre_enhanced_query に依存）
- Task 6: Task 5 の後（evaluate_revision の構造変更後）
- Task 7: 独立（いつでも実行可能）
- Task 8-9: 独立（いつでも実行可能）
