# --- input_handler.py ---
import os
import glob
import pandas as pd
from config import SearchConfig
from src.utils.logger import setup_logger
from typing import List, Tuple, Optional, Dict, Any

logger = setup_logger(__name__)


class InputHandler:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.input_dir = os.path.join(config.base_dir, "input")
        self.reference_dir = os.path.join(config.base_dir, "reference")

    def load_data(self) -> list:
        """入力データを読み込み、共通の形式に変換"""
        raise NotImplementedError

    def load_reference_data(self) -> dict:
        """参照データを読み込み、共通の形式に変換"""
        raise NotImplementedError

    def _get_column_names(self, df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
        """Excelファイルの列名を取得・検証"""
        if len(df.columns) < 2:
            raise ValueError("Input file must have at least 2 columns (Number and Query)")

        number_col = df.columns[0]
        query_col = df.columns[1]
        answer_col = df.columns[2] if len(df.columns) > 2 else None
        logger.info(f"Using columns: Number='{number_col}', Query='{query_col}', Answer='{answer_col}'")
        return number_col, query_col, answer_col

    def _build_combined_text(self, hierarchy: str, query: str, answer: str) -> str:
        """結合テキストを生成"""
        text_parts = []
        if hierarchy and hierarchy.strip():
            text_parts.append(f"分類: {hierarchy}")
        if query and query.strip():
            text_parts.append(f"質問: {query}")
        if answer and answer.strip():
            text_parts.append(f"回答: {answer}")
        return " | ".join(text_parts) if text_parts else ""

    def _get_latest_file(self, directory: str, file_pattern: str) -> str:
        """指定ディレクトリ内の最新ファイルを検索"""
        files = glob.glob(os.path.join(directory, file_pattern))
        # 一時ファイル・隠しファイルを除外
        files = [f for f in files
                 if not os.path.basename(f).startswith('~$')
                 and not f.endswith('.tmp')
                 and not os.path.basename(f).startswith('.')]
        if not files:
            raise FileNotFoundError(f"No files found matching pattern '{file_pattern}' in {directory}")
        return max(files, key=os.path.getctime)

class ExcelInputHandler(InputHandler):
    def load_data(self) -> list:
        input_file = self._get_latest_file(self.input_dir, "*.xlsx")
        self.current_file = os.path.basename(input_file)  # DB選択用
        logger.info(f"Processing input file: {self.current_file}")
        input_df = pd.read_excel(input_file)

        # 列名チェックとデータ抽出
        number_col, query_col, answer_col = self._get_column_names(input_df)
        valid_input_df = input_df.dropna(subset=[query_col]) # query_col が NaN の行を除外

        data = []
        for _, row in valid_input_df.iterrows():
            data.append({
                "number": str(row[number_col]),
                "query": str(row[query_col]),
                "answer": str(row[answer_col]) if answer_col and pd.notna(row[answer_col]) else ""
            })
        return data

    def load_reference_data(self) -> dict:
      reference_file = self._get_latest_file(self.reference_dir, "*.xlsx")
      logger.info(f"Using reference file: {os.path.basename(reference_file)}")
      reference_df = pd.read_excel(reference_file)
      
      # 列名の柔軟な対応
      query_col = None
      answer_col = None
      tag_col = None
      date_col = None
      
      # 問合せ内容の列名を検索
      possible_query_cols = ['分割後質問', '問合せ内容', '質問内容', '問い合わせ', '質問', 'query', 'Query']
      for col in possible_query_cols:
          if col in reference_df.columns:
              query_col = col
              break
      
      # 回答の列名を検索
      possible_answer_cols = ['分割後回答', '回答', '既存回答', 'answer', 'Answer']
      for col in possible_answer_cols:
          if col in reference_df.columns:
              answer_col = col
              break
      
      # タグ付けの列名を検索
      possible_tag_cols = ['タグ付け', 'タグ', '分類', 'category', 'Category', 'tag', 'Tag']
      for col in possible_tag_cols:
          if col in reference_df.columns:
              tag_col = col
              break
      
      # 日付列を検索（質問列の左側）
      if query_col and query_col in reference_df.columns:
          query_col_index = reference_df.columns.get_loc(query_col)
          if query_col_index > 0:
              date_col = reference_df.columns[query_col_index - 1]
      
      if query_col is None or answer_col is None:
          raise ValueError(f"Required columns not found. Available columns: {list(reference_df.columns)}")

      logger.info(f"Using columns: Query='{query_col}', Answer='{answer_col}', Tag='{tag_col}', Date='{date_col}'")

      # メタデータ対応のデータ構造
      queries = []
      answers = []
      combined_texts = []  # 質問と回答を結合したテキスト
      metadatas = []
      
      for idx, row in reference_df.iterrows():
          query_text = str(row[query_col]) if pd.notna(row[query_col]) else ""
          answer_text = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
          tag_text = str(row[tag_col]) if tag_col and pd.notna(row[tag_col]) else ""
          date_text = str(row[date_col]) if date_col and pd.notna(row[date_col]) else ""
          
          # 質問と回答を結合したテキストを生成（ベクトル化対象）
          # 新形式に統一
          text_parts = []
          if query_text.strip():
              text_parts.append(f"質問: {query_text}")
          if answer_text.strip():
              text_parts.append(f"回答: {answer_text}")
          combined_text = " | ".join(text_parts) if text_parts else ""
          combined_texts.append(combined_text)
          
          # 個別の質問と回答も保持（表示用）
          queries.append(query_text)
          answers.append(answer_text)
          
          # メタデータを構築
          metadata = {
              'tags': [tag_text] if tag_text.strip() else [],
              'date': date_text if date_text.strip() else "",
              'source': 'history_data',
              'row_index': idx
          }
          metadatas.append(metadata)

      return {
          'queries': queries,
          'answers': answers,
          'combined_texts': combined_texts,  # ベクトル化対象
          'metadatas': metadatas
      }


class HierarchicalExcelInputHandler(InputHandler):
    """階層構造Excelファイルを読み込むハンドラー（位置ベースの質問・回答判定）"""
    
    def load_data(self) -> list:
        return []
    
    def load_reference_data(self) -> dict:
        """マージ版シナリオExcelから参照データを抽出（位置ベースの質問・回答判定）"""
        reference_file = self._get_latest_file(self.reference_dir, "*.xlsx")
        logger.info(f"Processing hierarchical reference file: {os.path.basename(reference_file)}")
        
        all_sheets = pd.read_excel(reference_file, sheet_name=None)
        logger.info(f"Found {len(all_sheets)} sheets in the file")
        
        all_queries: List[str] = []
        all_answers: List[str] = []
        all_metadatas: List[dict] = []
        
        for sheet_name, df in all_sheets.items():
            logger.info(f"Processing sheet: {sheet_name}")
            
            # タグ列の存在確認（タグレス対応）
            tag_col_exists = 'タグ付け' in df.columns
            if not tag_col_exists:
                logger.info(f"Column 'タグ付け' not found in sheet '{sheet_name}', processing without tags")
            
            # 作成日列の取得（一番左の列）
            creation_date_col = df.columns[0]
            
            # 各行を処理
            for idx, row in df.iterrows():
                # タグ付け列の確認（タグレス対応）
                if tag_col_exists:
                    tag_text = str(row['タグ付け']).strip() if pd.notna(row['タグ付け']) else ""
                    if not tag_text:
                        continue  # タグが存在する場合のみ、空タグ行をスキップ
                else:
                    tag_text = ""  # タグ列が存在しない場合は空文字列
                
                # 作成日の取得（列名から日付を抽出）
                if '作成日' in creation_date_col:
                    creation_date = creation_date_col.replace('作成日', '')
                else:
                    creation_date = str(row[creation_date_col]) if pd.notna(row[creation_date_col]) else ""
                
                # 質問・回答の位置判定（右から左にスキャンして最初の非空セルを特定）
                answer = ""
                answer_col_idx = -1
                
                # 右から左にスキャンして最初の非空セルを見つける
                for col_idx in range(len(df.columns) - 1, 0, -1):  # 作成日列以外を右から左へ
                    col = df.columns[col_idx]
                    if col != creation_date_col:
                        cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
                        if cell_value:
                            answer = cell_value
                            answer_col_idx = col_idx
                            break
                
                # 有効な回答が見つからない場合はスキップ
                if not answer:
                    continue
                
                # 原則文の判定
                is_principle = "以下の選択肢から選んでください" in answer
                
                if is_principle:
                    # 原則文の場合：回答の左隣を質問として処理
                    if answer_col_idx > 1:
                        question_col_idx = answer_col_idx - 1
                        query = str(row[df.columns[question_col_idx]]).strip() if pd.notna(row[df.columns[question_col_idx]]) else ""
                    else:
                        query = ""
                    
                    # 質問の左側が階層構造
                    hierarchy_parts = []
                    hierarchy_end_idx = answer_col_idx - 1 if query else answer_col_idx
                    for col_idx in range(1, hierarchy_end_idx):  # 作成日列と質問・回答列以外
                        col = df.columns[col_idx]
                        if col != creation_date_col:
                            cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
                            if cell_value:
                                hierarchy_parts.append(cell_value)
                else:
                    # 通常の場合：回答の左隣が質問
                    if answer_col_idx > 1:
                        question_col_idx = answer_col_idx - 1
                        query = str(row[df.columns[question_col_idx]]).strip() if pd.notna(row[df.columns[question_col_idx]]) else ""
                    else:
                        query = ""
                    
                    # 質問の左側が階層構造
                    hierarchy_parts = []
                    hierarchy_end_idx = answer_col_idx - 1 if query else answer_col_idx
                    for col_idx in range(1, hierarchy_end_idx):  # 作成日列と質問・回答列以外
                        col = df.columns[col_idx]
                        if col != creation_date_col:
                            cell_value = str(row[col]).strip() if pd.notna(row[col]) else ""
                            if cell_value:
                                hierarchy_parts.append(cell_value)
                
                # 階層構造を「>」で連結
                hierarchy_text = " > ".join(hierarchy_parts) if hierarchy_parts else ""
                
                # データを追加
                all_queries.append(query)
                all_answers.append(answer)
                
                metadata = {
                    'tags': [tag_text] if tag_text else [],  # タグが存在する場合のみ格納
                    'date': creation_date,
                    'source': 'scenario',
                    'sheet_name': sheet_name,
                    'row_index': idx,
                    'hierarchy': hierarchy_text,  # 階層構造を保存
                    'is_principle': is_principle
                }
                all_metadatas.append(metadata)
            
            logger.info(f"Extracted {len(all_metadatas)} items so far")
        
        all_combined_texts = []
        for query, answer, metadata in zip(all_queries, all_answers, all_metadatas):
            hierarchy = metadata.get('hierarchy', '') if metadata else ''
            
            # 新形式に統一：階層構造、質問、回答をラベル付きで結合
            text_parts = []
            
            # 階層構造を追加（存在する場合）
            if hierarchy.strip():
                text_parts.append(f"分類: {hierarchy}")
            
            # 質問を追加（存在する場合）
            if query.strip():
                text_parts.append(f"質問: {query}")
            
            # 回答を追加（存在する場合）
            if answer.strip():
                text_parts.append(f"回答: {answer}")
            
            # 全体を結合
            combined_text = " | ".join(text_parts) if text_parts else ""
            all_combined_texts.append(combined_text)
        
        logger.info(f"Total extracted {len(all_queries)} reference items from all sheets")
        
        return {
            'queries': all_queries,
            'answers': all_answers,
            'combined_texts': all_combined_texts,
            'metadatas': all_metadatas
        }

class MultiFolderInputHandler(InputHandler):
    """複数フォルダから参照データを読み込むハンドラー"""

    def load_data(self) -> list:
        # 入力データの読み込み（従来通り）
        input_file = self._get_latest_file(self.input_dir, "*.xlsx")
        self.current_file = os.path.basename(input_file)  # DB選択用
        logger.info(f"Processing input file: {self.current_file}")
        input_df = pd.read_excel(input_file)

        # 列名チェックとデータ抽出
        number_col, query_col, answer_col = self._get_column_names(input_df)
        valid_input_df = input_df.dropna(subset=[query_col])

        data = []
        for _, row in valid_input_df.iterrows():
            data.append({
                "number": str(row[number_col]),
                "query": str(row[query_col]),
                "answer": str(row[answer_col]) if answer_col and pd.notna(row[answer_col]) else ""
            })
        return data

    def load_reference_data(self) -> dict:
        """複数フォルダから参照データを読み込み、統合"""
        all_queries = []
        all_answers = []
        all_metadatas = []
        
        # マージシナリオフォルダから読み込み
        scenario_dir = os.path.join(self.reference_dir, "scenario")
        if os.path.exists(scenario_dir):
            try:
                scenario_file = self._get_latest_file(scenario_dir, "*.xlsx")
                logger.info(f"Processing scenario file: {os.path.basename(scenario_file)}")
                
                scenario_handler = HierarchicalExcelInputHandler(self.config)
                scenario_handler.reference_dir = scenario_dir
                scenario_data = scenario_handler.load_reference_data()
                
                all_queries.extend(scenario_data['queries'])
                all_answers.extend(scenario_data['answers'])
                all_metadatas.extend(scenario_data['metadatas'])
                    
                logger.info(f"Added {len(scenario_data['queries'])} items from scenario file")
            except Exception as e:
                logger.warning(f"Error processing scenario file: {e}")
        
        # 履歴データフォルダから読み込み
        history_dir = os.path.join(self.reference_dir, "faq_data")
        if os.path.exists(history_dir):
            try:
                history_file = self._get_latest_file(history_dir, "*.xlsx")
                logger.info(f"Processing history file: {os.path.basename(history_file)}")
                
                history_handler = ExcelInputHandler(self.config)
                history_handler.reference_dir = history_dir
                history_data = history_handler.load_reference_data()
                
                all_queries.extend(history_data['queries'])
                all_answers.extend(history_data['answers'])
                all_metadatas.extend(history_data['metadatas'])
                
                logger.info(f"Added {len(history_data['queries'])} items from history file")
            except Exception as e:
                logger.warning(f"Error processing history file: {e}")
        
        # 質問、回答、階層構造を結合したテキストを生成
        all_combined_texts = []
        for query, answer, metadata in zip(all_queries, all_answers, all_metadatas):
            hierarchy = metadata.get('hierarchy', '') if metadata else ''
            
            # 階層構造、質問、回答を組み合わせてベクトル化対象テキストを作成
            text_parts = []
            
            # 階層構造を追加（存在する場合）
            if hierarchy.strip():
                text_parts.append(f"分類: {hierarchy}")
            
            # 質問を追加（存在する場合）
            if query.strip():
                text_parts.append(f"質問: {query}")
            
            # 回答を追加（存在する場合）
            if answer.strip():
                text_parts.append(f"回答: {answer}")
            
            # 全体を結合
            combined_text = " | ".join(text_parts) if text_parts else ""
            all_combined_texts.append(combined_text)
        
        if not all_queries:
            raise ValueError("No reference data found in any folder")
        
        logger.info(f"Total reference items: {len(all_queries)}")
        
        return {
            'queries': all_queries,
            'answers': all_answers,
            'combined_texts': all_combined_texts,  # ベクトル化対象
            'metadatas': all_metadatas
        }


# 他の入力形式 (CSV, JSONなど) のハンドラーもここに追加可能

class InputHandlerFactory:
    @staticmethod
    def create(input_type: str, config: SearchConfig) -> InputHandler:
        if input_type == "excel":
            return ExcelInputHandler(config)
        elif input_type == "hierarchical_excel":
            return HierarchicalExcelInputHandler(config)
        elif input_type == "multi_folder":
            return MultiFolderInputHandler(config)
        # 他の入力形式に対応するハンドラーをここに追加
        else:
            raise ValueError(f"Unsupported input type: {input_type}")