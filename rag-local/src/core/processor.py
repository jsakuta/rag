# --- processor.py ---
import logging
import pandas as pd
import os
import copy
from config import SearchConfig
from src.handlers.input_handler import InputHandlerFactory
from src.handlers.output_handler import OutputHandlerFactory
from src.core.searcher import Searcher
from src.core.judgment_support import JudgmentSupport
from src.utils.logger import setup_logger
from src.utils.auth import create_embedding_model
from tqdm import tqdm

logger = setup_logger(__name__)

class Processor:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.input_handler = InputHandlerFactory.create(config.input_type, config)
        self.output_handler = OutputHandlerFactory.create(config.output_type, config, app_prefix="answer")
        self.searcher = Searcher(config)
        # 参照データ用のハンドラーを別途作成
        self.reference_handler = InputHandlerFactory.create(config.reference_type, config)

        # 多段階検索モードの場合、判断支援モジュールを初期化
        if config.search_mode == "multi_stage":
            self.judgment_support = JudgmentSupport(config)
            logger.info("JudgmentSupport initialized for multi-stage search mode")
        else:
            self.judgment_support = None

    def process_data(self, mode: str = "batch", limit: int = None):
        """データ処理のメイン関数"""
        try:
            # 入力データの読み込み
            input_data = self.input_handler.load_data()
            if limit is not None:
                input_data = input_data[:limit]
                logger.info(f"--limit {limit}: 先頭{len(input_data)}件のみ処理")
            # 参照データの読み込み（新しいハンドラーを使用）
            reference_data = self.reference_handler.load_reference_data()

            # 検索の準備 (reference dataのベクトル化)
            self.searcher.prepare_search(reference_data)

            all_results = []
            # tqdmを使用してプログレスバーを表示
            for item in tqdm(input_data, desc="Processing data"):
                # 必須フィールドの取得（Noneチェック）
                query_number = item.get("number")
                query_text = item.get("query")
                if query_number is None or query_text is None:
                    logger.warning(f"Skipping malformed item (missing 'number' or 'query'): {item}")
                    continue
                original_answer = item.get("answer", "")

                logger.debug(f"=== 質問{query_number}の処理開始 ===")
                logger.debug(f"質問内容: {query_text[:100]}...")

                # 入力ファイル名を取得（動的DB選択用）
                input_file = getattr(self.input_handler, 'current_file', None)

                results = self.searcher.search(query_number, query_text, original_answer, input_file)

                logger.debug(f"質問{query_number}の検索結果数: {len(results)}")
                logger.debug(f"all_resultsに追加前の総数: {len(all_results)}")

                all_results.extend(results)

                logger.debug(f"all_resultsに追加後の総数: {len(all_results)}")

                # 質問ごとのall_results詳細確認（DEBUGレベル）
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"=== 質問{query_number}のall_results詳細確認 ===")
                    logger.debug(f"質問{query_number}で追加された結果数: {len(results)}")

                    # 質問ごとに追加された結果の詳細を確認
                    start_idx = len(all_results) - len(results)
                    for i, result in enumerate(results):
                        abs_idx = start_idx + i
                        logger.debug(f"  all_results[{abs_idx}]: Input_Number='{result.get('Input_Number', 'MISSING')}', Original_Query='{result.get('Original_Query', '')[:50]}...', Search_Result_Q='{result.get('Search_Result_Q', '')[:50]}...'")

                    # 質問ごとの集計確認（O(N)処理をDEBUGのみで実行）
                    current_question_count = sum(1 for item in all_results if item.get('Input_Number') == str(query_number))
                    empty_count_for_question = sum(1 for item in all_results[-len(results):] if item.get('Input_Number') == '')

                    logger.debug(f"質問{query_number}の総件数（all_results内）: {current_question_count}")
                    logger.debug(f"質問{query_number}で追加された空Input_Number数: {empty_count_for_question}")

                    # 最後の数件の結果を確認
                    if len(all_results) > 0:
                        logger.debug(f"最新の結果サンプル:")
                        for i, result in enumerate(all_results[-min(3, len(all_results)):]):
                            logger.debug(f"  結果{len(all_results)-min(3, len(all_results))+i+1}: Input_Number={result.get('Input_Number', 'N/A')}, Original_Query={result.get('Original_Query', 'N/A')[:30]}...")

                logger.info(f"質問{query_number}の処理完了（結果数: {len(results)}）")

            logger.info(f"=== 全処理完了 ===")
            logger.info(f"最終的なall_resultsの総数: {len(all_results)}")

            # 多段階検索モードの場合
            if self.config.search_mode == "multi_stage":
                # 両プロバイダー比較モードの場合
                if self.config.dual_provider_mode:
                    self._process_dual_provider_multi_stage(input_data)
                else:
                    # 従来の3シート出力
                    self._process_multi_stage_results(all_results, input_data)
            else:
                # 通常モード: 従来の出力
                self.output_handler.save_data(all_results, mode=mode)

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            raise

    def _process_multi_stage_results(self, results: list, input_data: list):
        """多段階検索結果のLLM判断支援と3シート出力"""
        logger.info("=== 多段階検索結果の後処理開始 ===")

        if self.judgment_support and self.config.multi_stage_enable_judgment_support:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Input Validation: revision_mapの構築時にデータ検証
            revision_map = {}
            for item in input_data:
                num = item.get("number")
                query = item.get("query", "")
                if num is not None:
                    # 数値型の場合は文字列に変換
                    revision_map[str(num)] = query if isinstance(query, str) else ""

            # パフォーマンス: LLM評価を並列実行
            def evaluate_single(result):
                """単一結果のLLM評価（スレッド用）"""
                input_num = result.get('Input_Number', '')
                evaluation = self.judgment_support.evaluate(
                    revision_map.get(input_num, result.get('Original_Query', '')),
                    result.get('Search_Result_Q', ''),
                    result.get('Search_Result_A', '')
                )
                result['Relevance_Judgment'] = evaluation['relevance_judgment']
                result['Judgment_Reason'] = evaluation['judgment_reason']
                return result

            max_workers = min(10, len(results))  # 最大10並列、または結果数
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(evaluate_single, result): result for result in results}
                for future in tqdm(as_completed(futures), total=len(results), desc="Evaluating relevance"):
                    try:
                        future.result()  # 例外があれば再スロー
                    except Exception as e:
                        # Sensitive Data Exposure防止: スタックトレースにAPIキーが含まれる可能性があるため
                        # exc_info=Falseに変更し、エラーメッセージのみをログ出力
                        logger.error(f"LLM評価エラー: {type(e).__name__}: {str(e)[:100]}")
                        result = futures[future]
                        result['Relevance_Judgment'] = "エラー"
                        result['Judgment_Reason'] = f"{type(e).__name__}: {str(e)[:200]}"

            logger.info("LLM判断支援完了（並列処理）")
        else:
            for result in results:
                result['Relevance_Judgment'] = ""
                result['Judgment_Reason'] = ""

        # 3シート出力
        logger.info("3シートExcel出力を実行中...")
        self.output_handler.save_data_multi_stage(results, mode="multi_stage")
        logger.info("=== 多段階検索結果の後処理完了 ===")

    def _process_dual_provider_multi_stage(self, input_data: list):
        """両プロバイダー比較モードで多段階検索を実行

        Azure OpenAIとVertex AI両方で検索を行い、結果を横並びで出力する
        """
        logger.info("=== 両プロバイダー比較モード開始 ===")

        # 参照データの読み込み（共通）
        reference_data = self.reference_handler.load_reference_data()

        # 入力ファイル名を取得（動的DB選択用）
        input_file = getattr(self.input_handler, 'current_file', None)

        # Azure OpenAIで検索
        logger.info("=== Azure OpenAI検索開始 ===")
        azure_results = self._execute_provider_search(
            input_data, reference_data, input_file, "azure_openai"
        )
        logger.info(f"Azure OpenAI検索完了: {len(azure_results)}件")

        # Vertex AIで検索
        logger.info("=== Vertex AI検索開始 ===")
        vertex_results = self._execute_provider_search(
            input_data, reference_data, input_file, "vertex_ai"
        )
        logger.info(f"Vertex AI検索完了: {len(vertex_results)}件")

        # 両プロバイダー結果を1シートに統合して出力
        logger.info("=== 両プロバイダー結果を統合出力 ===")
        self.output_handler.save_data_dual_provider(
            azure_results, vertex_results, input_data, mode="dual_provider"
        )
        logger.info("=== 両プロバイダー比較モード完了 ===")

    def _execute_provider_search(self, input_data: list, reference_data: dict,
                                  input_file: str, provider: str) -> list:
        """指定されたプロバイダーで検索を実行

        Args:
            input_data: 入力データリスト
            reference_data: 参照データ
            input_file: 入力ファイル名
            provider: 埋め込みプロバイダー ("azure_openai" or "vertex_ai")

        Returns:
            list: 検索結果リスト
        """
        # プロバイダー用の設定をコピー
        provider_config = copy.copy(self.config)
        provider_config.embedding_provider = provider

        # プロバイダーに対応する埋め込みモデル名を設定
        if provider == "azure_openai":
            provider_config.embedding_model = os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
            )
        else:  # vertex_ai
            provider_config.embedding_model = os.getenv(
                "DEFAULT_EMBEDDING_MODEL", "text-multilingual-embedding-002"
            )

        # プロバイダー用のSearcherを作成
        provider_embedding = create_embedding_model(provider_config)
        provider_searcher = Searcher(provider_config, embedding_model=provider_embedding)

        # 検索の準備
        provider_searcher.prepare_search(reference_data)

        all_results = []
        for item in tqdm(input_data, desc=f"Searching ({provider})"):
            query_number = item.get("number")
            query_text = item.get("query")
            if query_number is None or query_text is None:
                continue
            original_answer = item.get("answer", "")

            results = provider_searcher.search(query_number, query_text, original_answer, input_file)

            # 全結果にInput_Numberを設定（順位管理用）
            for result in results:
                result['Input_Number'] = str(query_number)

            all_results.extend(results)

        return all_results