# --- input_handler.py ---
import os
import glob
import pandas as pd
from config import SearchConfig
from utils.logger import setup_logger

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

    def _get_latest_file(self, directory: str, file_pattern: str) -> str:
        """指定ディレクトリ内の最新ファイルを検索"""
        files = glob.glob(os.path.join(directory, file_pattern))
        if not files:
            raise FileNotFoundError(f"No files found matching pattern '{file_pattern}' in {directory}")
        return max(files, key=os.path.getctime)

class ExcelInputHandler(InputHandler):
    def load_data(self) -> list:
        input_file = self._get_latest_file(self.input_dir, "*.xlsx")
        logger.info(f"Processing input file: {os.path.basename(input_file)}")
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
      if '問合せ内容' not in reference_df.columns or '回答' not in reference_df.columns:
          raise ValueError("Required columns ('問合せ内容', '回答') not found in reference file")

      return {
          'queries': reference_df['問合せ内容'].fillna('').astype(str).tolist(),
          'answers': reference_df['回答'].fillna('').astype(str).tolist()
      }

    def _get_column_names(self, df: pd.DataFrame) -> tuple[str, str, str]:
        """Excelファイルの列名を取得・検証"""
        if len(df.columns) < 2:
            raise ValueError("Input file must have at least 2 columns (Number and Query)")

        number_col = df.columns[0]
        query_col = df.columns[1]
        answer_col = df.columns[2] if len(df.columns) > 2 else None
        logger.info(f"Using columns: Number='{number_col}', Query='{query_col}', Answer='{answer_col}'")
        return number_col, query_col, answer_col

# 他の入力形式 (CSV, JSONなど) のハンドラーもここに追加可能

class InputHandlerFactory:
    @staticmethod
    def create(input_type: str, config: SearchConfig) -> InputHandler:
        if input_type == "excel":
            return ExcelInputHandler(config)
        # 他の入力形式に対応するハンドラーをここに追加
        else:
            raise ValueError(f"Unsupported input type: {input_type}")