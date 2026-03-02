# --- src/core/search/search_strategy.py ---
"""検索戦略パターン

検索モードごとの処理をStrategyパターンで分離し、
Searcherクラスの責務を軽減する。

Strategies:
    - OriginalSearchStrategy: 原文ハイブリッド検索
    - LLMEnhancedSearchStrategy: LLM拡張検索
    - MultiStageSearchStrategy: 多段階OR検索
    - KeywordFilterSearchStrategy: キーワードフィルタ検索
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

from src.core.search.text_combiner import get_text_combiner
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.core.searcher import Searcher

logger = setup_logger(__name__)


class SearchStrategy(ABC):
    """検索戦略の抽象基底クラス"""

    def __init__(self, searcher: 'Searcher'):
        self.searcher = searcher
        self.config = searcher.config

    @abstractmethod
    def execute(
        self,
        input_number: str,
        query_text: str,
        original_answer: str
    ) -> List[Dict[str, Any]]:
        """検索を実行"""
        pass


class OriginalSearchStrategy(SearchStrategy):
    """原文ハイブリッド検索

    質問文をそのまま使用してベクトル+キーワードのハイブリッド検索を実行。
    """

    def execute(self, input_number, query_text, original_answer):
        logger.info(f"Row (No.{input_number}):")
        logger.info(f"  Search mode: original")
        logger.info(f"  Original query: {query_text[:100]}...")
        if original_answer:
            logger.info(f"  Original answer: {original_answer[:100]}...")

        keywords = self.searcher._extract_keywords(query_text)
        logger.info(f"  Extracted keywords: {keywords}")

        search_results = self.searcher._execute_vector_search(query_text)
        results = self.searcher._calculate_and_merge_scores(search_results, keywords)
        return self.searcher._format_final_results(
            results, input_number, query_text, original_answer, query_text
        )


class LLMEnhancedSearchStrategy(SearchStrategy):
    """LLM拡張検索

    LLMでクエリを拡張してからハイブリッド検索を実行。
    """

    def execute(self, input_number, query_text, original_answer):
        logger.info(f"Row (No.{input_number}):")
        logger.info(f"  Search mode: llm_enhanced")
        logger.info(f"  Original query: {query_text[:100]}...")
        if original_answer:
            logger.info(f"  Original answer: {original_answer[:100]}...")

        search_query = self.searcher.summarize_text(query_text)
        logger.info(f"  Generated search query: {search_query}")

        keywords = self.searcher._extract_keywords(query_text)
        logger.info(f"  Extracted keywords: {keywords}")

        search_results = self.searcher._execute_vector_search(search_query)
        results = self.searcher._calculate_and_merge_scores(search_results, keywords)
        return self.searcher._format_final_results(
            results, input_number, query_text, original_answer, search_query
        )


class MultiStageSearchStrategy(SearchStrategy):
    """多段階OR検索

    原文検索 + LLM拡張検索のOR結合で網羅性を確保。
    """

    def execute(self, input_number, query_text, original_answer):
        logger.info(f"=== 多段階OR検索開始 (No.{input_number}) ===")
        logger.info(f"  Threshold: {self.config.multi_stage_threshold}, Max: {self.config.multi_stage_max_results}")

        keywords = self.searcher._extract_keywords(query_text)
        logger.info(f"  Keywords: {keywords}")

        # Stage 1: 原文検索
        original_results = self._hybrid_search_with_threshold(query_text, keywords)
        logger.info(f"  原文検索: {len(original_results)}件")

        # Stage 2: LLMクエリ検索
        try:
            llm_query = self.searcher.summarize_text(query_text)
        except Exception as e:
            logger.error(f"  LLMクエリ生成エラー: {e}")
            llm_query = query_text

        llm_results = self._hybrid_search_with_threshold(llm_query, keywords)
        logger.info(f"  LLMクエリ検索: {len(llm_results)}件")

        # Stage 3: OR結合と3分類
        merged = self._merge_results(
            original_results, llm_results,
            input_number, query_text, original_answer, llm_query
        )

        category_counts = {}
        for r in merged:
            cat = r.get('Search_Category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        logger.info(f"=== 多段階OR検索完了: {len(merged)}件 {category_counts} ===")

        return merged

    def _hybrid_search_with_threshold(
        self, query_for_vector: str, keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """しきい値ベースのハイブリッド検索"""
        from src.utils.dynamic_db_manager import DynamicDBError

        if self.searcher.vector_db is None:
            raise DynamicDBError("VectorDB not initialized.")

        filter_metadata = self.searcher._build_source_filter()
        query_vector = self.searcher.model.encode(
            [query_for_vector], normalize_embeddings=True
        )[0]
        search_results = self.searcher.vector_db.search(
            query_embedding=query_vector,
            n_results=self.config.multi_stage_max_results,
            filter_metadata=filter_metadata
        )

        keyword_sims = self.searcher._calculate_keyword_similarities(
            search_results, keywords
        )

        filtered = []
        for i, sr in enumerate(search_results):
            score = (
                self.config.vector_weight * sr['similarity']
                + self.config.keyword_weight * keyword_sims[i]
            )
            score = max(0.0, min(1.0, score))

            if score >= self.config.multi_stage_threshold:
                rd = self.searcher._build_result_data(sr, score)
                rd['_doc_id'] = sr['id']
                filtered.append(rd)

        filtered.sort(key=lambda x: x['Similarity'], reverse=True)
        return filtered

    def _merge_results(
        self, original_results, llm_results,
        input_number, query_text, original_answer, llm_query
    ) -> List[Dict[str, Any]]:
        """OR結合して3分類（Both / Original_Only / LLM_Enhanced_Only）"""
        original_ids = {r['_doc_id'] for r in original_results}
        llm_ids = {r['_doc_id'] for r in llm_results}

        both_ids = original_ids & llm_ids
        original_only = original_ids - llm_ids
        llm_only = llm_ids - original_ids

        logger.info(f"    Both: {len(both_ids)}, Original_Only: {len(original_only)}, LLM_Only: {len(llm_only)}")

        merged = []
        processed = set()

        # 'Both': 高スコアを優先
        for doc_id in both_ids:
            orig = next((r for r in original_results if r.get('_doc_id') == doc_id), None)
            llm = next((r for r in llm_results if r.get('_doc_id') == doc_id), None)
            if orig and llm:
                best = orig if orig.get('Similarity', 0) >= llm.get('Similarity', 0) else llm
                merged.append(self._categorize(
                    best, 'Both', input_number, query_text, original_answer, llm_query
                ))
                processed.add(doc_id)

        # 'Original_Only' / 'LLM_Enhanced_Only'
        for results, ids_set, category, sq in [
            (original_results, original_only, 'Original_Only', query_text),
            (llm_results, llm_only, 'LLM_Enhanced_Only', llm_query),
        ]:
            for r in results:
                did = r.get('_doc_id')
                if did in ids_set and did not in processed:
                    merged.append(self._categorize(
                        r, category, input_number, query_text, original_answer, sq
                    ))
                    processed.add(did)

        merged.sort(key=lambda x: x['Similarity'], reverse=True)
        return merged[:self.config.top_k]

    @staticmethod
    def _categorize(result, category, input_number, query_text, original_answer, search_query):
        """結果にカテゴリ情報を付与"""
        rc = result.copy()
        rc.update({
            'Search_Category': category,
            'Input_Number': input_number,
            'Original_Query': query_text,
            'Original_Answer': original_answer,
            'Search_Query': search_query,
        })
        rc.pop('_doc_id', None)
        return rc


class KeywordFilterSearchStrategy(SearchStrategy):
    """キーワードフィルタ検索

    ベクトル検索を使用せず、キーワードマッチのみで結果を返す。
    用語の単純置換（AML→GPLEX等）の検出に適する。
    """

    def execute(self, input_number, query_text, original_answer):
        logger.info(f"Row (No.{input_number}):")
        logger.info(f"  Search type: keyword_filter")
        logger.info(f"  Original query: {query_text[:100]}...")

        keywords = self.searcher._extract_keywords(query_text)
        if not keywords:
            logger.warning("  キーワードが抽出できませんでした")
            return []
        logger.info(f"  Extracted keywords: {keywords}")

        # キーワードキャッシュでフィルタリング
        query_kw_set = set(keywords)
        matched = []
        for idx, ref_kw in self.searcher._reference_keywords_cache.items():
            mc = len(query_kw_set.intersection(ref_kw))
            if mc > 0:
                matched.append((idx, mc))

        if not matched:
            logger.info("  キーワードに一致する結果がありませんでした")
            return []

        matched.sort(key=lambda x: (-x[1], x[0]))
        logger.info(f"  Keyword filter matched: {len(matched)} results")

        # 結果をフォーマット
        results = []
        for idx, match_count in matched:
            if idx >= len(self.searcher.reference_texts):
                continue

            metadata = (
                self.searcher.reference_metadatas[idx]
                if idx < len(self.searcher.reference_metadatas) else {}
            )
            combined_text = self.searcher.reference_texts[idx]
            parsed = get_text_combiner().parse(combined_text)

            if metadata.get('source') == 'scenario':
                hierarchy = metadata.get('hierarchy', '')
                q = parsed.query
                sq = f"{hierarchy} > {q}" if hierarchy and q else (hierarchy or q)
                sa = parsed.answer
            else:
                sq = parsed.query
                sa = parsed.answer

            sheet_name = metadata.get('sheet_name', '')
            row_index = metadata.get('row_index', '')
            sim = match_count / len(query_kw_set) if query_kw_set else 0.0

            results.append({
                'Input_Number': '',
                'Original_Query': '',
                'Original_Answer': '',
                'Search_Query': '',
                'Search_Result_Q': sq,
                'Search_Result_A': sa,
                'Similarity': sim,
                'Match_Count': match_count,
                'Scenario_ID': f"{sheet_name}_{row_index}" if sheet_name and row_index != '' else '',
                'Sheet_Name': sheet_name,
                'Row_Index': row_index,
                'Vector_Weight': 0.0,
                'Top_K': self.config.top_k,
            })

        results = results[:self.config.top_k]
        logger.info(f"  Final results: {len(results)} items (top_k={self.config.top_k})")

        if results:
            results[0]['Input_Number'] = input_number
            results[0]['Original_Query'] = query_text
            results[0]['Original_Answer'] = original_answer
            results[0]['Search_Query'] = f"[キーワード必須] {', '.join(keywords)}"

        return results


def create_strategy(searcher: 'Searcher') -> SearchStrategy:
    """設定に基づいて適切なSearchStrategyを生成"""
    config = searcher.config

    if config.search_type == "keyword_filter":
        logger.info(f"SearchStrategy: KeywordFilter")
        return KeywordFilterSearchStrategy(searcher)
    elif config.search_mode == "multi_stage":
        logger.info(f"SearchStrategy: MultiStage")
        return MultiStageSearchStrategy(searcher)
    elif config.search_mode == "llm_enhanced":
        logger.info(f"SearchStrategy: LLMEnhanced")
        return LLMEnhancedSearchStrategy(searcher)
    else:
        logger.info(f"SearchStrategy: Original")
        return OriginalSearchStrategy(searcher)
