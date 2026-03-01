# --- src/core/search/multi_stage_orchestrator.py ---
"""多段階検索オーケストレーター

原文検索 + LLMクエリ検索のOR結合を管理。
"""

from typing import List, Dict, Any, Optional, Set, Tuple

from src.core.search.vector_search_engine import VectorSearchEngine
from src.core.search.keyword_search_engine import KeywordSearchEngine
from src.core.search.query_enhancer import QueryEnhancer
from src.core.search.text_combiner import TextCombiner
from src.types.search_types import (
    SearchResultDict,
    MultiStageSearchResultDict,
    SearchResultKeys,
    MetadataKeys,
    SourceValues,
    SearchCategoryValues,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MultiStageOrchestrator:
    """多段階検索オーケストレーター

    原文検索とLLMクエリ検索を並行実行し、結果をOR結合する。

    Attributes:
        vector_engine: ベクトル検索エンジン
        keyword_engine: キーワード検索エンジン
        query_enhancer: クエリ拡張エンジン
        text_combiner: テキスト結合ユーティリティ
        vector_weight: ベクトルスコアの重み
        keyword_weight: キーワードスコアの重み
        threshold: 結果に含めるスコアしきい値
        max_results: 各検索の最大結果数
        filter_mode: フィルタリングモード ('threshold' or 'top_k')
        top_k: TOP-K件数（filter_mode='top_k'の場合に使用）
    """

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        keyword_engine: KeywordSearchEngine,
        query_enhancer: QueryEnhancer,
        text_combiner: TextCombiner,
        vector_weight: float = 0.9,
        threshold: float = 0.45,
        max_results: int = 100,
        filter_mode: str = "threshold",
        top_k: int = 50
    ):
        """MultiStageOrchestratorを初期化

        Args:
            vector_engine: ベクトル検索エンジン
            keyword_engine: キーワード検索エンジン
            query_enhancer: クエリ拡張エンジン
            text_combiner: テキスト結合ユーティリティ
            vector_weight: ベクトルスコアの重み
            threshold: 結果に含めるスコアしきい値
            max_results: 各検索の最大結果数
            filter_mode: フィルタリングモード ('threshold' or 'top_k')
            top_k: TOP-K件数（filter_mode='top_k'の場合に使用）
        """
        self.vector_engine = vector_engine
        self.keyword_engine = keyword_engine
        self.query_enhancer = query_enhancer
        self.text_combiner = text_combiner
        self.vector_weight = vector_weight
        self.keyword_weight = 1.0 - vector_weight
        self.threshold = threshold
        self.max_results = max_results
        self.filter_mode = filter_mode
        self.top_k = top_k

        logger.info("MultiStageOrchestratorを初期化しました")

    def execute(
        self,
        input_number: str,
        query_text: str,
        original_answer: str,
        filter_metadata: Dict[str, str] = None,
        pre_enhanced_query: Optional[str] = None,
    ) -> List[MultiStageSearchResultDict]:
        """多段階OR検索を実行

        Args:
            input_number: 入力番号
            query_text: 検索クエリテキスト
            original_answer: 元の回答
            filter_metadata: メタデータフィルタ（オプション）
            pre_enhanced_query: 事前に生成済みのLLMクエリ（キャッシュ用、省略時は内部で生成）

        Returns:
            List[MultiStageSearchResultDict]: 検索結果のリスト
        """
        logger.info(f"=== 多段階OR検索開始 (No.{input_number}) ===")
        filter_info = f"TOP-K: {self.top_k}" if self.filter_mode == "top_k" else f"Threshold: {self.threshold}"
        logger.info(f"  FilterMode: {self.filter_mode}, {filter_info}, Max: {self.max_results}")

        # キーワード抽出
        keywords = self.keyword_engine.extract_keywords(query_text)
        logger.info(f"  Keywords: {keywords}")

        # Stage 1: 原文検索
        original_results = self._execute_hybrid_search(
            query_text, keywords, filter_metadata
        )
        logger.info(f"  原文検索: {len(original_results)}件")

        # Stage 2: LLMクエリ検索
        if pre_enhanced_query:
            llm_query = pre_enhanced_query
        else:
            try:
                llm_query = self.query_enhancer.enhance(query_text)
            except Exception as e:
                logger.error(f"  LLMクエリ生成エラー: {e}")
                llm_query = query_text

        llm_results = self._execute_hybrid_search(
            llm_query, keywords, filter_metadata
        )
        logger.info(f"  LLMクエリ検索: {len(llm_results)}件")

        # Stage 3: OR結合と3分類
        merged_results = self._merge_results(
            original_results, llm_results,
            input_number, query_text, original_answer, llm_query
        )

        # カテゴリ別の件数をログ出力
        category_counts: Dict[str, int] = {}
        for r in merged_results:
            cat = r.get(SearchResultKeys.SEARCH_CATEGORY, 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        logger.info(f"=== 多段階OR検索完了: {len(merged_results)}件 {category_counts} ===")

        return merged_results

    def _execute_hybrid_search(
        self,
        query: str,
        keywords: List[str],
        filter_metadata: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """ハイブリッド検索を実行（閾値またはTOP-Kでフィルタリング）

        Args:
            query: 検索クエリ
            keywords: キーワードリスト
            filter_metadata: メタデータフィルタ

        Returns:
            List[Dict]: 検索結果（_doc_id付き）
        """
        # ベクトル検索実行
        search_results = self.vector_engine.search(
            query, self.max_results, filter_metadata
        )

        # キーワード類似度計算
        query_keywords_set = set(keywords)
        all_results = []

        for search_result in search_results:
            # キーワード類似度計算
            doc_id = search_result['id']
            if doc_id.startswith('doc_'):
                original_idx = int(doc_id.split('_')[1])
            else:
                original_idx = int(doc_id)

            if self.keyword_engine.has_cached_keywords(original_idx):
                ref_keywords = self.keyword_engine.get_cached_keywords(original_idx)
                keyword_sim = self.keyword_engine.calculate_similarity_fast(
                    query_keywords_set, ref_keywords
                )
            else:
                logger.warning(f"Keyword cache miss for index {original_idx}")
                keyword_sim = 0.0

            # 統合スコア計算
            combined_score = (
                self.vector_weight * search_result['similarity'] +
                self.keyword_weight * keyword_sim
            )
            combined_score = max(0.0, min(1.0, combined_score))

            result_data = self._build_result_data(search_result, combined_score)
            result_data['_doc_id'] = search_result['id']
            all_results.append(result_data)

        # スコアでソート
        all_results.sort(key=lambda x: x[SearchResultKeys.SIMILARITY], reverse=True)

        # フィルタリングモードに応じて結果をフィルタリング
        if self.filter_mode == "top_k":
            # TOP-Kモード: 上位K件を返す
            return all_results[:self.top_k]
        else:
            # 閾値モード: 閾値以上のスコアを持つ結果を返す
            return [r for r in all_results if r[SearchResultKeys.SIMILARITY] >= self.threshold]

    def _build_result_data(
        self,
        search_result: Dict[str, Any],
        combined_score: float
    ) -> SearchResultDict:
        """検索結果からresult_dataを構築

        Args:
            search_result: ベクトル検索結果
            combined_score: 統合スコア

        Returns:
            SearchResultDict: 結果データ
        """
        metadata = search_result['metadata']
        combined_text = search_result['document']
        parsed_text = self.text_combiner.parse(combined_text)

        # 表示用クエリの構築
        source = metadata.get(MetadataKeys.SOURCE, '')
        search_result_query = self.text_combiner.build_display_query(
            parsed_text, source, metadata.get(MetadataKeys.HIERARCHY)
        )

        # シナリオIDを生成
        sheet_name = metadata.get(MetadataKeys.SHEET_NAME, '')
        row_index = metadata.get(MetadataKeys.ROW_INDEX, '')
        scenario_id = f"{sheet_name}_{row_index}" if sheet_name and row_index != '' else ''

        return {
            SearchResultKeys.INPUT_NUMBER: '',
            SearchResultKeys.ORIGINAL_QUERY: '',
            SearchResultKeys.ORIGINAL_ANSWER: '',
            SearchResultKeys.SEARCH_QUERY: '',
            SearchResultKeys.SEARCH_RESULT_Q: search_result_query,
            SearchResultKeys.SEARCH_RESULT_A: parsed_text.answer,
            SearchResultKeys.SIMILARITY: combined_score,
            SearchResultKeys.SCENARIO_ID: scenario_id,
            SearchResultKeys.SHEET_NAME: sheet_name,
            SearchResultKeys.ROW_INDEX: row_index,
            SearchResultKeys.VECTOR_WEIGHT: self.vector_weight,
            SearchResultKeys.TOP_K: self.max_results,
            SearchResultKeys.HIERARCHY: metadata.get(MetadataKeys.HIERARCHY, ''),
            SearchResultKeys.LV1_CATEGORY: metadata.get(MetadataKeys.DATE, ''),
            '_source': metadata.get(MetadataKeys.SOURCE, 'unknown'),
        }

    def _merge_results(
        self,
        original_results: List[Dict[str, Any]],
        llm_results: List[Dict[str, Any]],
        input_number: str,
        query_text: str,
        original_answer: str,
        llm_query: str
    ) -> List[MultiStageSearchResultDict]:
        """多段階検索結果をOR結合して3分類

        Args:
            original_results: 原文検索結果
            llm_results: LLM検索結果
            input_number: 入力番号
            query_text: 元のクエリ
            original_answer: 元の回答
            llm_query: LLMで生成したクエリ

        Returns:
            List[MultiStageSearchResultDict]: マージされた結果
        """
        original_ids = {r['_doc_id'] for r in original_results}
        llm_ids = {r['_doc_id'] for r in llm_results}

        both_ids = original_ids & llm_ids
        original_only_ids = original_ids - llm_ids
        llm_only_ids = llm_ids - original_ids

        logger.info(f"    Both: {len(both_ids)}, Original_Only: {len(original_only_ids)}, LLM_Only: {len(llm_only_ids)}")

        merged_results: List[MultiStageSearchResultDict] = []
        processed_ids: Set[str] = set()

        # 'Both'カテゴリ: 両方に存在する結果は高スコアを優先
        for doc_id in both_ids:
            orig_result = next((r for r in original_results if r.get('_doc_id') == doc_id), None)
            llm_result = next((r for r in llm_results if r.get('_doc_id') == doc_id), None)

            if orig_result and llm_result:
                orig_score = orig_result.get(SearchResultKeys.SIMILARITY, 0)
                llm_score = llm_result.get(SearchResultKeys.SIMILARITY, 0)

                if orig_score >= llm_score:
                    best_result = orig_result
                    logger.debug(f"    Both doc_id={doc_id}: Using original score ({orig_score:.4f} >= {llm_score:.4f})")
                else:
                    best_result = llm_result
                    logger.debug(f"    Both doc_id={doc_id}: Using LLM score ({llm_score:.4f} > {orig_score:.4f})")

                result_copy = self._create_categorized_result(
                    best_result, SearchCategoryValues.BOTH,
                    input_number, query_text, original_answer, llm_query
                )
                merged_results.append(result_copy)
                processed_ids.add(doc_id)

        # 'Original_Only'カテゴリ
        for result in original_results:
            doc_id = result.get('_doc_id')
            if doc_id in original_only_ids and doc_id not in processed_ids:
                result_copy = self._create_categorized_result(
                    result, SearchCategoryValues.ORIGINAL_ONLY,
                    input_number, query_text, original_answer, query_text
                )
                merged_results.append(result_copy)
                processed_ids.add(doc_id)

        # 'LLM_Enhanced_Only'カテゴリ
        for result in llm_results:
            doc_id = result.get('_doc_id')
            if doc_id in llm_only_ids and doc_id not in processed_ids:
                result_copy = self._create_categorized_result(
                    result, SearchCategoryValues.LLM_ENHANCED_ONLY,
                    input_number, query_text, original_answer, llm_query
                )
                merged_results.append(result_copy)
                processed_ids.add(doc_id)

        # スコアでソート
        merged_results.sort(key=lambda x: x[SearchResultKeys.SIMILARITY], reverse=True)

        # TOP-Kモードの場合は上位K件に絞る
        if self.filter_mode == "top_k":
            merged_results = merged_results[:self.top_k]

        return merged_results

    def _create_categorized_result(
        self,
        result: Dict[str, Any],
        category: str,
        input_number: str,
        query_text: str,
        original_answer: str,
        search_query: str
    ) -> MultiStageSearchResultDict:
        """カテゴリ付きの結果を作成

        Args:
            result: 元の結果
            category: 検索カテゴリ
            input_number: 入力番号
            query_text: 元のクエリ
            original_answer: 元の回答
            search_query: 使用した検索クエリ

        Returns:
            MultiStageSearchResultDict: カテゴリ付き結果
        """
        result_copy = dict(result)
        result_copy.update({
            SearchResultKeys.SEARCH_CATEGORY: category,
            SearchResultKeys.INPUT_NUMBER: input_number,
            SearchResultKeys.ORIGINAL_QUERY: query_text,
            SearchResultKeys.ORIGINAL_ANSWER: original_answer,
            SearchResultKeys.SEARCH_QUERY: search_query,
        })
        result_copy.pop('_doc_id', None)
        return result_copy
