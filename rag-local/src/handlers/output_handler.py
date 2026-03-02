# --- output_handler.py ---
import os
import pandas as pd
from config import SearchConfig
from src.utils.logger import setup_logger
from datetime import datetime

logger = setup_logger(__name__)

class OutputHandler:
    def __init__(self, config: SearchConfig, app_prefix: str = ""):
        self.config = config
        base_output_dir = os.path.join(config.base_dir, "data", "output", "latest")
        self.output_dir = os.path.join(base_output_dir, app_prefix) if app_prefix else base_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_data(self, data: list):
        """データを保存"""
        raise NotImplementedError

class ExcelOutputHandler(OutputHandler):
    def _make_output_path(self, mode: str) -> str:
        """タイムスタンプ付き出力ファイルパスを生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"answer_{mode}_{timestamp}.xlsx")

    def save_data(self, data: list, mode: str = "batch"):
        if not data:
            logger.warning("No data to save.")
            return
        
        # dataが空でないことを確認してからDataFrameを作成
        df = pd.DataFrame(data)
        output_file = self._make_output_path(mode)

        # ExcelWriter のオプションを修正
        writer_options = {
            'engine': 'xlsxwriter',
            'engine_kwargs': {'options': {'nan_inf_to_errors': True}}
        }
        
        try:
            with pd.ExcelWriter(output_file, **writer_options) as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                self._format_excel(writer, df)  # processor.py から移動
                self._write_metadata_sheet(writer)
            logger.info(f"Results saved to: {output_file}")
            return output_file
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

        output_file = self._make_output_path(mode)

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

                self._write_metadata_sheet(writer)

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
            'Scenario_ID',
            'Relevance_Judgment',
            'Judgment_Reason',
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

        column_widths = [8, 50, 30, 40, 50, 50, 10, 15, 50, 50, 10, 8]
        for i, width in enumerate(column_widths):
            worksheet.set_column(i, i, width)

        header_names = {
            'Input_Number': '#', 'Original_Query': '改定内容', 'Original_Answer': '元回答',
            'Search_Query': '検索クエリ', 'Search_Result_Q': '検索結果Q', 'Search_Result_A': '検索結果A',
            'Similarity': '類似度', 'Scenario_ID': 'シナリオID', 'Relevance_Judgment': '関連性判定',
            'Judgment_Reason': '判定根拠',
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

    def save_data_dual_provider(self, azure_results: list, vertex_results: list,
                                 input_data: list, mode: str = "dual_provider"):
        """両プロバイダー比較結果を1シートに横並びで保存

        Args:
            azure_results: Azure OpenAI検索結果（Input_Number順位でソート済み）
            vertex_results: Vertex AI検索結果（Input_Number順位でソート済み）
            input_data: 入力データ（correct_ids含む）
            mode: 出力モード
        """
        if not azure_results and not vertex_results:
            logger.warning("No data to save.")
            return

        # 正解IDマップを構築
        correct_id_map = {}
        for item in input_data:
            correct_id_map[str(item.get('number', ''))] = item.get('correct_ids', [])

        # 結果をマージ（順位合わせ）
        merged_data = self._merge_provider_results(azure_results, vertex_results, correct_id_map)

        if not merged_data:
            logger.warning("No merged data to save.")
            return

        df = pd.DataFrame(merged_data)

        output_file = self._make_output_path(mode)

        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter',
                                engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
                df.to_excel(writer, index=False, sheet_name='比較結果')
                self._format_excel_dual_provider(writer, df)
                self._write_metadata_sheet(writer)

            logger.info(f"Dual provider results saved to: {output_file}")
            logger.info(f"  Total rows: {len(df)}")

        except Exception as e:
            logger.error(f"Error saving dual provider data to Excel: {e}", exc_info=True)
            raise

    def _merge_provider_results(self, azure_results: list, vertex_results: list,
                                 correct_id_map: dict) -> list:
        """Azure/VertexAI結果を順位で横並びにマージ"""
        # Input_Number別にグルーピング
        azure_by_input = {}
        vertex_by_input = {}

        for r in azure_results:
            input_num = r.get('Input_Number', '')
            if input_num:
                if input_num not in azure_by_input:
                    azure_by_input[input_num] = []
                azure_by_input[input_num].append(r)

        for r in vertex_results:
            input_num = r.get('Input_Number', '')
            if input_num:
                if input_num not in vertex_by_input:
                    vertex_by_input[input_num] = []
                vertex_by_input[input_num].append(r)

        # 全入力番号を取得してソート
        all_input_nums = sorted(set(list(azure_by_input.keys()) + list(vertex_by_input.keys())),
                                key=lambda x: int(x) if x.isdigit() else 0)

        merged_data = []

        for input_num in all_input_nums:
            azure_list = azure_by_input.get(input_num, [])
            vertex_list = vertex_by_input.get(input_num, [])

            # 各リストを類似度でソート
            azure_list.sort(key=lambda x: x.get('Similarity', 0), reverse=True)
            vertex_list.sort(key=lambda x: x.get('Similarity', 0), reverse=True)

            # 最大順位数
            max_rank = max(len(azure_list), len(vertex_list))
            correct_ids = correct_id_map.get(input_num, [])

            # 改定内容を取得
            revision_content = ''
            if azure_list:
                revision_content = azure_list[0].get('Original_Query', '')
            elif vertex_list:
                revision_content = vertex_list[0].get('Original_Query', '')

            for rank in range(max_rank):
                azure_result = azure_list[rank] if rank < len(azure_list) else {}
                vertex_result = vertex_list[rank] if rank < len(vertex_list) else {}

                # Azure側の正解判定
                azure_id = azure_result.get('Scenario_ID', '')
                azure_correct = azure_id in correct_ids if azure_id else False

                # VertexAI側の正解判定
                vertex_id = vertex_result.get('Scenario_ID', '')
                vertex_correct = vertex_id in correct_ids if vertex_id else False

                row = {
                    '#': input_num,
                    '改定内容': revision_content if rank == 0 else '',
                    '順位': rank + 1,
                    'Azure_Q': azure_result.get('Search_Result_Q', ''),
                    'Azure_A': azure_result.get('Search_Result_A', ''),
                    'Azure類似度': azure_result.get('Similarity', ''),
                    'Azure_ID': azure_id,
                    'Azure正解': 'TRUE' if azure_correct else ('FALSE' if azure_id else ''),
                    'Azure_Cat': azure_result.get('Search_Category', ''),
                    'VertexAI_Q': vertex_result.get('Search_Result_Q', ''),
                    'VertexAI_A': vertex_result.get('Search_Result_A', ''),
                    'VertexAI類似度': vertex_result.get('Similarity', ''),
                    'VertexAI_ID': vertex_id,
                    'VertexAI正解': 'TRUE' if vertex_correct else ('FALSE' if vertex_id else ''),
                    'VertexAI_Cat': vertex_result.get('Search_Category', ''),
                }
                merged_data.append(row)

        return merged_data

    def _format_excel_dual_provider(self, writer: pd.ExcelWriter, df: pd.DataFrame):
        """両プロバイダー比較結果のExcel書式設定"""
        worksheet = writer.sheets['比較結果']
        workbook = writer.book

        # ヘッダー書式
        header_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'bold': True,
            'border': 1,
            'bg_color': '#D9D9D9',
            'text_wrap': True,
        })

        # 通常セル書式
        cell_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'border': 1,
            'text_wrap': True,
        })

        # 正解セル書式（緑色）
        correct_format = workbook.add_format({
            'font_name': 'メイリオ',
            'font_size': 10,
            'border': 1,
            'text_wrap': True,
            'bg_color': '#C6EFCE',
            'font_color': '#006100',
        })

        # 列幅設定
        column_widths = {
            '#': 4,
            '改定内容': 25,
            '順位': 4,
            'Azure_Q': 30,
            'Azure_A': 30,
            'Azure類似度': 8,
            'Azure_ID': 12,
            'Azure正解': 6,
            'Azure_Cat': 8,
            'VertexAI_Q': 30,
            'VertexAI_A': 30,
            'VertexAI類似度': 8,
            'VertexAI_ID': 12,
            'VertexAI正解': 6,
            'VertexAI_Cat': 8,
        }

        for col_num, col_name in enumerate(df.columns):
            width = column_widths.get(col_name, 10)
            worksheet.set_column(col_num, col_num, width)

        # ヘッダー行の書式設定
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)

        # データセルの書式設定（正解セルは緑色）
        azure_correct_col = list(df.columns).index('Azure正解') if 'Azure正解' in df.columns else -1
        vertex_correct_col = list(df.columns).index('VertexAI正解') if 'VertexAI正解' in df.columns else -1

        for row_num in range(len(df)):
            for col_num in range(len(df.columns)):
                value = df.iloc[row_num, col_num]
                if pd.isna(value):
                    value = ''

                # 正解列のTRUEセルは緑色
                if col_num == azure_correct_col and value == 'TRUE':
                    worksheet.write(row_num + 1, col_num, value, correct_format)
                elif col_num == vertex_correct_col and value == 'TRUE':
                    worksheet.write(row_num + 1, col_num, value, correct_format)
                else:
                    worksheet.write(row_num + 1, col_num, value, cell_format)

    def _write_metadata_sheet(self, writer: pd.ExcelWriter):
        """検索パラメータを Metadata シートに記録"""
        metadata = {
            "Parameter": [
                "vector_weight", "keyword_weight", "search_mode",
                "search_type", "top_k", "embedding_provider",
                "embedding_model", "timestamp",
            ],
            "Value": [
                self.config.vector_weight,
                self.config.keyword_weight,
                self.config.search_mode,
                self.config.search_type,
                self.config.top_k,
                self.config.embedding_provider,
                self.config.embedding_model,
                datetime.now().isoformat(),
            ],
        }
        pd.DataFrame(metadata).to_excel(writer, index=False, sheet_name="Metadata")


# 他の出力形式 (CSV, JSONなど) のハンドラーもここに追加可能

class OutputHandlerFactory:
    @staticmethod
    def create(output_type: str, config: SearchConfig, app_prefix: str = "") -> OutputHandler:
        if output_type == "excel":
            return ExcelOutputHandler(config, app_prefix=app_prefix)
        # 他の出力形式に対応するハンドラーをここに追加
        else:
            raise ValueError(f"Unsupported output type: {output_type}")