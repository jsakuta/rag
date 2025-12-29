import os
import glob
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from config import SearchConfig
from search import HybridSearchMixin
from utils import setup_logger
from langchain.schema import HumanMessage, SystemMessage

logger = setup_logger(__name__)

class ExcelVectorProcessor(HybridSearchMixin):
    def __init__(self, config: SearchConfig):
        self.config = config
        # 修正箇所：HybridSearchMixinの初期化時にconfigを渡す
        super().__init__(config=config)
        
        # ディレクトリパスの設定
        self.base_dir = config.base_dir
        self.input_dir = os.path.join(self.base_dir, "input")
        self.reference_dir = os.path.join(self.base_dir, "reference")
        self.output_dir = os.path.join(self.base_dir, "output")
        self.prompt_dir = os.path.join(self.base_dir, "prompt")
        
        # キャッシュディレクトリの設定
        self.cache_dir = os.path.join(self.reference_dir, "vector_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
        
        # ディレクトリの作成
        for directory in [self.input_dir, self.reference_dir, self.output_dir, self.prompt_dir, self.cache_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
        
        # LLMの設定（継承元のクラスに移動）
        self.llm = self._setup_llm()
        
        # プロンプトテンプレートの読み込み
        self.prompt_template = self._load_latest_prompt()
        
    def _load_latest_prompt(self) -> str:
        """最新のプロンプトファイルを読み込む"""
        prompt_files = glob.glob(os.path.join(self.prompt_dir, "*"))
        if not prompt_files:
            raise FileNotFoundError(f"No prompt files found in {self.prompt_dir}")
        
        latest_prompt_file = max(prompt_files, key=os.path.getctime)
        print(f"Using prompt file: {os.path.basename(latest_prompt_file)}")
        
        with open(latest_prompt_file, 'r', encoding='utf-8') as f:
            return f.read()

    def summarize_text(self, text: str) -> str:
        """LLMを使用してテキストを要約"""
        messages = [
            SystemMessage(content=self.prompt_template),
            HumanMessage(content=text)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return text  # エラー時は元のテキストを返す

    def get_latest_files(self) -> tuple:
        """最新のファイルを取得"""
        input_files = glob.glob(os.path.join(self.input_dir, "*.xlsx"))
        reference_files = glob.glob(os.path.join(self.reference_dir, "*.xlsx"))
        
        if not input_files:
            raise FileNotFoundError(f"No Excel files found in {self.input_dir}")
        if not reference_files:
            raise FileNotFoundError(f"No Excel files found in {self.reference_dir}")
            
        latest_input = max(input_files, key=os.path.getctime)
        latest_reference = max(reference_files, key=os.path.getctime)
        
        return latest_input, latest_reference

    def _cache_vectors(self, vectors: np.ndarray, texts: List[str], reference_file: str):
        """ベクトルとテキストをキャッシュに保存"""
        cache_data = {
            'vectors': vectors.tolist(),
            'texts': texts,
            'timestamp': datetime.now().isoformat(),
            'reference_file': reference_file
        }
        
        cache_file = os.path.join(
            self.cache_dir, 
            f"cache_{os.path.splitext(os.path.basename(reference_file))[0]}.json"
        )
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached vectors to: {cache_file}")

    def _load_cached_vectors(self, reference_file: str) -> Optional[tuple]:
        """キャッシュからベクトルを読み込む"""
        cache_file = os.path.join(
            self.cache_dir, 
            f"cache_{os.path.splitext(os.path.basename(reference_file))[0]}.json"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logger.info(f"Loaded vectors from cache: {cache_file}")
                return np.array(cache_data['vectors']), cache_data['texts']
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _get_column_names(self, df: pd.DataFrame) -> tuple:
        """入力ファイルの最初の3列を取得"""
        if len(df.columns) < 2:
            raise ValueError("Input file must have at least 2 columns (Number and Query)")
        
        number_col = df.columns[0]  # 1列目をNo.として使用
        query_col = df.columns[1]   # 2列目を問合せ内容として使用
        answer_col = df.columns[2] if len(df.columns) > 2 else None  # 3列目があれば回答として使用
        
        logger.info(f"Using columns: Number='{number_col}', Query='{query_col}', Answer='{answer_col}'")
        return number_col, query_col, answer_col

    def _format_excel(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """Excelファイルの書式設定"""
        worksheet = writer.sheets['Sheet1']
        workbook = writer.book

        # セルの書式設定
        cell_format = workbook.add_format({
            'font_name': 'メイリオ',  # フォントをメイリオに
            'font_size': 10,
            'border': 1,  # 格子線を追加
            'text_wrap': True,  # テキストの折り返しを有効に
        })

        # ヘッダーの書式設定
        header_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'bold': True,
            'border': 1,
            'bg_color': '#D9D9D9',  # ヘッダーの背景色
            'text_wrap': True,
        })

        # 列幅の設定
        worksheet.set_column('A:A', 10)  # Input_Number
        worksheet.set_column('B:B', 40)  # Original_Query
        worksheet.set_column('C:C', 40)  # Original_Answer
        worksheet.set_column('D:D', 30)  # Summarized_Query
        worksheet.set_column('E:E', 40)  # Search_Result_Q
        worksheet.set_column('F:F', 40)  # Search_Result_A
        worksheet.set_column('G:G', 10)  # Similarity
        worksheet.set_column('H:H', 10)  # Vector_Weight
        worksheet.set_column('I:I', 10)  # Top_K
        
        # ヘッダーの文字列をカスタム文字列に変更
        header_names = {
            'Input_Number': '#',
            'Original_Query': 'ユーザーの質問',
            'Original_Answer': 'オリジナルの回答',  # 新しいヘッダーを追加
            'Summarized_Query': '検索クエリ（AI処理）',
            'Search_Result_Q': '類似質問',
            'Search_Result_A': '類似回答',
            'Similarity': '類似度',
            'Vector_Weight': 'ベクトルの重み',
            'Top_K': '候補数'
        }


        # ヘッダー行の書式を適用
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, header_names.get(value, value), header_format)


        # データ部分の書式を適用
        for row_num in range(df.shape[0]):
            for col_num in range(df.shape[1]):
                value = df.iloc[row_num, col_num]
                # NaN値を空文字列に変換
                if pd.isna(value):
                    value = ''
                worksheet.write(row_num + 1, col_num, value, cell_format)

    def _process_query(self, input_number, query_text, original_answer, reference_texts, reference_vectors):
        """共通のクエリ処理ロジック"""
        summarized_text = self.summarize_text(query_text)
        logger.info(f"Row (No.{input_number}):")
        logger.info(f"  Original query: {query_text[:100]}...")
        logger.info(f"  Original answer: {original_answer[:100]}..." if original_answer else "  No original answer")
        logger.info(f"  Summarized query: {summarized_text}")
        
        search_results = self._get_hybrid_search_results(
            query_text=query_text,
            summarized_text=summarized_text,
            reference_texts=reference_texts,
            reference_vectors=reference_vectors,
            top_k=self.config.top_k
        )
        
        results = []
        for i, (ref_idx, similarity) in enumerate(search_results):
            new_row = {
                'Input_Number': input_number if i == 0 else '',
                'Original_Query': query_text if i == 0 else '',
                'Original_Answer': original_answer if i == 0 else '',  # 原回答を追加
                'Summarized_Query': summarized_text if i == 0 else '',
                'Search_Result_Q': self.reference_df.iloc[ref_idx]['問合せ内容'],
                'Search_Result_A': self.reference_df.iloc[ref_idx]['回答'],
                'Similarity': similarity,
                'Vector_Weight': self.config.vector_weight,  # 各行のvector_weightを追加
                'Top_K': self.config.top_k   # 各行のtop_kを追加
            }
            results.append(new_row)
            logger.info(f"  Added result with similarity: {similarity:.4f}")
            
        return results


    def process_files(self):
        """ファイル処理のメイン関数"""
        try:
            # ファイルの取得と読み込み
            input_file, reference_file = self.get_latest_files()
            logger.info(f"Processing input file: {os.path.basename(input_file)}")
            logger.info(f"Using reference file: {os.path.basename(reference_file)}")
            
            # 入力データの読み込みと前処理
            input_df = pd.read_excel(input_file)
            
            # 列名の検証と取得
            number_col, query_col, answer_col = self._get_column_names(input_df)
            
            # 空でないデータのフィルタリング
            valid_input_df = input_df[input_df[query_col].notna() & (input_df[query_col].str.strip() != '')]
            empty_count = len(input_df) - len(valid_input_df)
            
            logger.info(f"Total input rows: {len(input_df)}")
            logger.info(f"Empty or invalid rows: {empty_count}")
            logger.info(f"Valid rows to process: {len(valid_input_df)}")
            
            if len(valid_input_df) == 0:
                raise ValueError("No valid data found in input file")
            
            # リファレンスデータの読み込みと検証
            self.reference_df = pd.read_excel(reference_file)
            if '問合せ内容' not in self.reference_df.columns or '回答' not in self.reference_df.columns:
                raise ValueError("Required columns not found in reference file")
            
            # 結果用のDataFrame
             # 結果用のDataFrame
            result_columns = [
                'Input_Number', 'Original_Query', 'Original_Answer',  # Original_Answer を追加
                'Summarized_Query', 
                'Search_Result_Q', 'Search_Result_A', 'Similarity',
                'Vector_Weight', 'Top_K'
            ]
            all_results = []  # すべての結果を一時保存するリスト
            
            # リファレンスデータのベクトル化
            logger.info("Vectorizing reference data...")
            reference_texts = self.reference_df['問合せ内容'].fillna('').astype(str).tolist()
            logger.info(f"Reference data size: {len(reference_texts)} entries")
            
            # キャッシュの確認
            cached_data = self._load_cached_vectors(reference_file)
            if cached_data:
                reference_vectors, cached_texts = cached_data
                if cached_texts == reference_texts:
                    logger.info("Using cached vectors")
                else:
                    logger.info("Cache mismatch, generating new vectors")
                    reference_vectors = self.model.encode(reference_texts, normalize_embeddings=True)
                    self._cache_vectors(reference_vectors, reference_texts, reference_file)
            else:
                reference_vectors = self.model.encode(reference_texts, normalize_embeddings=True)
                self._cache_vectors(reference_vectors, reference_texts, reference_file)
            
            # 有効な入力データの処理
            total_rows = len(valid_input_df)
            processed_count = 0
            matched_count = 0
            
            for idx, row in valid_input_df.iterrows():
                progress = (processed_count + 1) / total_rows * 100
                logger.info(f"Processing row {processed_count + 1}/{total_rows} ({progress:.1f}%)")
                
                input_number = str(row[number_col])
                query_text = str(row[query_col])
                original_answer = str(row[answer_col]) if answer_col else ''  # 回答列があれば取得

                # 共通のクエリ処理を呼び出し
                results = self._process_query(input_number, query_text, original_answer, reference_texts, reference_vectors)
                all_results.extend(results)
                matched_count += len(results)
                
                processed_count += 1
            
            # すべての結果をDataFrameに変換
            result_df = pd.DataFrame(all_results, columns=result_columns)
            
            # 処理サマリーの出力
            logger.info("=== Processing Summary ===")
            logger.info(f"Total input rows: {len(input_df)}")
            logger.info(f"Empty/invalid rows: {empty_count}")
            logger.info(f"Processed rows: {processed_count}")
            logger.info(f"Matched results: {matched_count}")
            logger.info(f"Average matches per row: {matched_count/processed_count if processed_count > 0 else 0:.2f}")
            
            # 結果の保存前に確認
            if result_df.empty:
                logger.error("No results were generated!")
            else:
                logger.info(f"Generated {len(result_df)} rows of results")
                
            # パラメータ情報を含むファイル名を生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            param_summary = self.config.get_param_summary()
            output_file = os.path.join(
                self.output_dir, 
                f"{input_basename}_result_{param_summary}_{timestamp}.xlsx"
            )
            
            # NaN値を適切に処理するためのオプションを追加
            writer_options = {
                'engine': 'xlsxwriter',
                'engine_kwargs': {'options': {'nan_inf_to_errors': True}}
            }
            
            # NaN値を空文字列に置換
            result_df = result_df.fillna('')
            
            with pd.ExcelWriter(output_file, **writer_options) as writer:
                result_df.to_excel(writer, index=False, sheet_name='Sheet1')
                self._format_excel(writer, result_df)
            
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"First few rows of results:\n{result_df.head()}")
            
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}", exc_info=True)
            raise