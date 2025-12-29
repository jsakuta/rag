# --- output_handler.py ---
import os
import pandas as pd
from config import SearchConfig
from utils.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)

class OutputHandler:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.output_dir = os.path.join(config.base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def save_data(self, data: list):
        """データを保存"""
        raise NotImplementedError

class ExcelOutputHandler(OutputHandler):
    def save_data(self, data: list, mode: str = "batch"):
        if not data:
            logger.warning("No data to save.")
            return
        
        # dataが空でないことを確認してからDataFrameを作成
        df = pd.DataFrame(data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_summary = self.config.get_param_summary()
        # ファイル名にモードを追加
        output_file = os.path.join(
            self.output_dir,
            f"output_{mode}_{param_summary}_{timestamp}.xlsx"
        )

        # ExcelWriter のオプションを修正
        writer_options = {
            'engine': 'xlsxwriter',
            'engine_kwargs': {'options': {'nan_inf_to_errors': True}}
        }
        
        try:
            with pd.ExcelWriter(output_file, **writer_options) as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                self._format_excel(writer, df)  # processor.py から移動
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving data to Excel: {e}")

    def _format_excel(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """Excelファイルの書式設定 (processor.pyから移動、調整)"""
        worksheet = writer.sheets['Sheet1']
        workbook = writer.book

        cell_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'border': 1,
            'text_wrap': True,
        })

        header_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'bold': True,
            'border': 1,
            'bg_color': '#D9D9D9',
            'text_wrap': True,
        })

        worksheet.set_column('A:A', 10)  # Input_Number
        worksheet.set_column('B:B', 40)  # Original_Query
        worksheet.set_column('C:C', 30)  # Search_Query
        worksheet.set_column('D:D', 40)  # Search_Result_Q
        worksheet.set_column('E:E', 40)  # Search_Result_A
        worksheet.set_column('F:F', 40)  # Similarity (幅を40に変更)
        worksheet.set_column('G:G', 10)  # Vector_Weight
        worksheet.set_column('H:H', 10)  # Top_K

        header_names = {
            'Input_Number': '#',
            'Original_Query': 'ユーザーの質問',
            'Original_Answer': 'ユーザーの回答',
            'Search_Query': '検索クエリ',
            'Search_Result_Q': '類似質問',
            'Search_Result_A': '類似回答',
            'Similarity': '類似度',
            'Vector_Weight': 'ベクトルの重み',
            'Top_K': '候補数',
            'Generated_Tags': '生成タグ'
        }

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, header_names.get(value, value), header_format)

        for row_num in range(df.shape[0]):
            for col_num in range(df.shape[1]):
                value = df.iloc[row_num, col_num]
                if pd.isna(value):
                    value = ''
                worksheet.write(row_num + 1, col_num, value, cell_format)
# 他の出力形式 (CSV, JSONなど) のハンドラーもここに追加可能

class OutputHandlerFactory:
    @staticmethod
    def create(output_type: str, config: SearchConfig) -> OutputHandler:
        if output_type == "excel":
            return ExcelOutputHandler(config)
        # 他の出力形式に対応するハンドラーをここに追加
        else:
            raise ValueError(f"Unsupported output type: {output_type}")