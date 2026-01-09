# --- output_handler.py ---
import os
import pandas as pd
from config import SearchConfig
from src.utils.logger import setup_logger
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
            logger.error(f"Error saving data to Excel: {e}", exc_info=True)
            raise  # 呼び出し元に例外を伝播

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

    def save_data_multi_stage(self, data: list, mode: str = "multi_stage"):
        """多段階検索結果を3シートに分けて保存"""
        if not data:
            logger.warning("No data to save.")
            return

        df = pd.DataFrame(data)
        has_category = 'Search_Category' in df.columns

        categories = {
            name: df[df['Search_Category'] == name] if has_category else pd.DataFrame()
            for name in ['Both', 'Original_Only', 'LLM_Enhanced_Only']
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.output_dir,
            f"output_{mode}_{self.config.get_param_summary()}_{timestamp}.xlsx"
        )

        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter',
                                engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
                expected_columns = self._get_multi_stage_columns()

                for sheet_name, sheet_df in categories.items():
                    if sheet_df.empty:
                        output_df = pd.DataFrame(columns=expected_columns)
                    else:
                        output_df = sheet_df.drop(columns=['Search_Category'], errors='ignore')
                        available_cols = [c for c in expected_columns if c in output_df.columns]
                        output_df = output_df[available_cols]

                    output_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    self._format_excel_multi_stage(writer, sheet_name, output_df)

            logger.info(f"Multi-stage results saved to: {output_file}")
            for name, cat_df in categories.items():
                logger.info(f"  {name}: {len(cat_df)} rows")

        except Exception as e:
            logger.error(f"Error saving multi-stage data to Excel: {e}", exc_info=True)
            raise

    def _get_multi_stage_columns(self):
        """多段階検索出力の列名リスト"""
        return [
            'Input_Number',
            'Original_Query',
            'Original_Answer',
            'Search_Query',
            'Search_Result_Q',
            'Search_Result_A',
            'Similarity',
            'Impact_Reason',
            'Modification_Suggestion',
            'Vector_Weight',
            'Top_K'
        ]

    def _format_excel_multi_stage(self, writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame):
        """多段階検索結果のExcel書式設定"""
        worksheet = writer.sheets[sheet_name]
        workbook = writer.book

        sheet_colors = {
            'Both': '#E2EFDA',
            'Original_Only': '#FFF2CC',
            'LLM_Enhanced_Only': '#DEEBF7'
        }

        header_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'bold': True,
            'border': 1,
            'bg_color': sheet_colors.get(sheet_name, '#D9D9D9'),
            'text_wrap': True,
        })

        cell_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'border': 1,
            'text_wrap': True,
        })

        column_widths = [8, 50, 30, 40, 50, 50, 10, 50, 50, 10, 8]
        for i, width in enumerate(column_widths):
            worksheet.set_column(i, i, width)

        header_names = {
            'Input_Number': '#', 'Original_Query': '改定内容', 'Original_Answer': '元回答',
            'Search_Query': '検索クエリ', 'Search_Result_Q': '検索結果Q', 'Search_Result_A': '検索結果A',
            'Similarity': '類似度', 'Impact_Reason': '影響の根拠', 'Modification_Suggestion': '修正案',
            'Vector_Weight': 'ベクトル重み', 'Top_K': '候補数'
        }

        # ヘッダー行の書式設定
        for col_num, col_name in enumerate(self._get_multi_stage_columns()):
            worksheet.write(0, col_num, header_names.get(col_name, col_name), header_format)

        # データセルの書式設定
        for row_num in range(len(df)):
            for col_num in range(len(df.columns)):
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