# --- processor.py ---
import logging
import os
import sys
import time
from tqdm import tqdm
from config import SearchConfig
from input_handler import InputHandlerFactory
from output_handler import OutputHandlerFactory
from searcher import Searcher
from utils.logger import setup_logger

logger = setup_logger(__name__)

class Processor:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.searcher = Searcher(config)
        self.input_handler = InputHandlerFactory.create(config.input_type, config)
        self.output_handler = OutputHandlerFactory.create(config.output_type, config)

    def process_data(self, mode: str = "batch"):
        """データ処理のメイン関数"""
        try:
            # 入力データの読み込み
            input_data = self.input_handler.load_data()
            reference_data = self.input_handler.load_reference_data()

            # 検索の準備 (reference dataのベクトル化)
            self.searcher.prepare_search(reference_data)

            all_results = []
            # tqdmを使用してプログレスバーを表示
            for item in tqdm(input_data, desc="Processing data"):
                # 検索実行
                query_number = item["number"]
                query_text = item["query"]
                original_answer = item.get("answer", "") # 回答がない場合も考慮

                results = self.searcher.search(query_number, query_text, original_answer)
                all_results.extend(results)
            
            # 結果の保存
            self.output_handler.save_data(all_results, mode=mode)

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}", exc_info=True)
            raise