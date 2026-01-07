# --- processor.py ---
import logging
import pandas as pd
import os
import json
from datetime import datetime
from config import SearchConfig
from src.handlers.input_handler import InputHandlerFactory
from src.handlers.output_handler import OutputHandlerFactory
from src.core.searcher import Searcher
from src.utils.logger import setup_logger
from tqdm import tqdm

logger = setup_logger(__name__)

class Processor:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.input_handler = InputHandlerFactory.create(config.input_type, config)
        self.output_handler = OutputHandlerFactory.create(config.output_type, config)
        self.searcher = Searcher(config)
        # 参照データ用のハンドラーを別途作成
        self.reference_handler = InputHandlerFactory.create(config.reference_type, config)

    def process_data(self, mode: str = "batch"):
        """データ処理のメイン関数"""
        try:
            # 入力データの読み込み
            input_data = self.input_handler.load_data()
            # 参照データの読み込み（新しいハンドラーを使用）
            reference_data = self.reference_handler.load_reference_data()

            # 検索の準備 (reference dataのベクトル化)
            self.searcher.prepare_search(reference_data)

            all_results = []
            # tqdmを使用してプログレスバーを表示
            for item in tqdm(input_data, desc="Processing data"):
                # 必須フィールドの取得（KeyError防止）
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
            
            # 結果の保存
            self.output_handler.save_data(all_results, mode=mode)

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            raise