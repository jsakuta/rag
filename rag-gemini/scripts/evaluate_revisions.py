"""
事務改定評価スクリプト（多段階検索・横並び比較版）

多段階ハイブリッド検索を使用して、Azure/VertexAI両方で検索を実行。
結果を横並びで比較できるExcelファイルに出力。

入力: input/multi_stage_input.xlsx
出力: output/revision_evaluation_{timestamp}.xlsx

前提条件:
- scripts/rebuild_before_scenario_db.py を実行済みで、
  reference/vector_db/rev*/{azure_openai,vertex_ai}/ が構築済みであること
"""

import os
import sys
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from itertools import zip_longest
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# .envファイルを読み込み
load_dotenv(PROJECT_ROOT / ".env")

from config import SearchConfig
from src.utils.logger import setup_logger
from src.utils.auth import create_embedding_model, create_llm
from src.utils.vector_db import MetadataVectorDB
from src.core.search.vector_search_engine import VectorSearchEngine
from src.core.search.keyword_search_engine import KeywordSearchEngine
from src.core.search.query_enhancer import QueryEnhancer
from src.core.search.text_combiner import TextCombiner, get_text_combiner
from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
from src.core.judgment_support import JudgmentSupport
from src.types.search_types import SearchResultKeys, SearchCategoryValues

logger = setup_logger(__name__)

# 改定番号 → rev*業務分野のマッピング
REVISION_TO_AREAS = {
    '①': ['rev01smile'],
    '②': ['rev02souzoku'],
    '③': ['rev03naibujimu', 'rev03smile', 'rev03souzoku', 'rev03torikaku'],
    '④': ['rev04naibujimu'],
    '⑤': ['rev05smile'],
    '⑥': ['rev06smile'],
}

# ボット名マッピング
AREA_TO_BOT = {
    'smile': 'smile-bot',
    'naibujimu': 'naibujimu-bot',
    'souzoku': 'souzoku-bot',
    'torikaku': 'torikaku-bot',
}

# 入力ファイル
INPUT_FILE = PROJECT_ROOT / "input" / "multi_stage_input.xlsx"

# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "output"

# ベクトルDBベースパス
VECTOR_DB_BASE = PROJECT_ROOT / "reference" / "vector_db"

# 検索設定
THRESHOLD = 0.45  # 閾値
VECTOR_WEIGHT = 0.9  # ベクトル重み
MAX_RESULTS = 100  # 最大検索結果数


class RevisionEvaluator:
    """事務改定評価クラス（多段階検索・横並び比較版）"""

    def __init__(self, config: SearchConfig, enable_llm_analysis: bool = True):
        """初期化

        Args:
            config: 検索設定
            enable_llm_analysis: LLM分析（JudgmentSupport）を有効にするか
        """
        self.config = config
        self.enable_llm_analysis = enable_llm_analysis
        self.text_combiner = get_text_combiner()

        # LLM判断支援を初期化
        if enable_llm_analysis:
            judgment_config = copy.copy(config)
            judgment_config.multi_stage_enable_judgment_support = True
            self.judgment_support = JudgmentSupport(judgment_config)
        else:
            self.judgment_support = None

        # LLM初期化（クエリ拡張用）
        self.llm = create_llm(config)

    def load_input_data(self) -> pd.DataFrame:
        """入力データを読み込み"""
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {INPUT_FILE}")

        df = pd.read_excel(INPUT_FILE)
        logger.info(f"入力データを読み込み: {len(df)}件")
        return df

    def _get_embedding_model(self, provider: str):
        """プロバイダー別の埋め込みモデルを取得"""
        provider_config = copy.copy(self.config)
        provider_config.embedding_provider = provider

        if provider == "azure_openai":
            provider_config.embedding_model = os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
            )
        else:  # vertex_ai
            provider_config.embedding_model = os.getenv(
                "VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"
            )

        return create_embedding_model(provider_config)

    def _extract_bot_name_from_area(self, area: str) -> str:
        """エリア名からボット名を抽出"""
        area_lower = area.lower()
        for keyword, bot_name in AREA_TO_BOT.items():
            if keyword in area_lower:
                return bot_name
        return 'unknown-bot'

    def _create_orchestrator(
        self,
        provider: str,
        area: str,
        reference_queries: List[str]
    ) -> Optional[MultiStageOrchestrator]:
        """多段階検索オーケストレーターを作成

        Args:
            provider: プロバイダー名（azure_openai / vertex_ai）
            area: エリア名（rev01smile等）
            reference_queries: 参照クエリリスト（キーワードキャッシュ用）

        Returns:
            MultiStageOrchestrator or None
        """
        db_path = str(VECTOR_DB_BASE / area / provider)

        if not Path(db_path).exists():
            logger.warning(f"DBが存在しません: {db_path}")
            return None

        chroma_file = Path(db_path) / "chroma.sqlite3"
        if not chroma_file.exists():
            logger.warning(f"ChromaDBファイルが存在しません: {chroma_file}")
            return None

        try:
            # 埋め込みモデルを取得
            embedding_model = self._get_embedding_model(provider)

            # VectorDBを読み込み
            vector_db = MetadataVectorDB(
                db_path=db_path,
                collection_name="default"
            )

            # ベクトル検索エンジン
            vector_engine = VectorSearchEngine(
                embedding_model=embedding_model,
                vector_db=vector_db
            )

            # キーワード検索エンジン
            keyword_engine = KeywordSearchEngine(
                stop_words=self.config.STOP_WORDS,
                position_weight=self.config.POSITION_WEIGHT
            )

            # キーワードキャッシュを構築
            keyword_engine.build_cache(reference_queries)

            # クエリ拡張エンジン
            query_enhancer = QueryEnhancer(
                llm=self.llm,
                base_dir=str(PROJECT_ROOT)
            )

            # オーケストレーター
            orchestrator = MultiStageOrchestrator(
                vector_engine=vector_engine,
                keyword_engine=keyword_engine,
                query_enhancer=query_enhancer,
                text_combiner=self.text_combiner,
                vector_weight=VECTOR_WEIGHT,
                threshold=THRESHOLD,
                max_results=MAX_RESULTS
            )

            return orchestrator

        except Exception as e:
            logger.error(f"オーケストレーター作成エラー ({area}/{provider}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_reference_queries(self, area: str, provider: str) -> List[str]:
        """DBからクエリを取得してキャッシュ用に返す"""
        db_path = str(VECTOR_DB_BASE / area / provider)

        if not Path(db_path).exists():
            return []

        try:
            vector_db = MetadataVectorDB(
                db_path=db_path,
                collection_name="default"
            )
            # 全ドキュメントを取得（countは信頼できないため直接取得）
            collection = vector_db.collection
            result = collection.get(include=['documents'])

            if not result.get('documents'):
                return []
            documents = result.get('documents', [])

            # ドキュメントからクエリ部分を抽出
            queries = []
            for doc in documents:
                if doc:
                    parsed = self.text_combiner.parse(doc)
                    if parsed.query:
                        queries.append(parsed.query)
                    else:
                        queries.append(doc[:100])  # フォールバック

            return queries

        except Exception as e:
            logger.error(f"参照クエリ取得エラー ({area}/{provider}): {e}")
            return []

    def _convert_result_to_dict(
        self,
        result: Dict[str, Any],
        correct_ids: List[str],
        area: str
    ) -> Dict[str, Any]:
        """検索結果を出力用辞書に変換

        Args:
            result: MultiStageOrchestrator からの結果
            correct_ids: 正解IDリスト
            area: エリア名

        Returns:
            出力用辞書
        """
        # シナリオIDを構築
        sheet_name = result.get(SearchResultKeys.SHEET_NAME, '')
        row_index = result.get(SearchResultKeys.ROW_INDEX, '')

        if sheet_name and row_index != '':
            # row_indexがDataFrameインデックス（0始まり）の場合、Excel行番号に変換
            try:
                excel_row = int(row_index) + 2  # 0-based → Excel行番号（ヘッダー+1）
                bot_name = self._extract_bot_name_from_area(area)
                scenario_id = f"{bot_name}_{excel_row}"
            except (ValueError, TypeError):
                scenario_id = result.get(SearchResultKeys.SCENARIO_ID, '')
        else:
            scenario_id = result.get(SearchResultKeys.SCENARIO_ID, '')

        # 正解判定
        is_correct = scenario_id in correct_ids

        return {
            'シナリオID': scenario_id,
            '類似度': round(result.get(SearchResultKeys.SIMILARITY, 0), 4),
            'カテゴリ': result.get(SearchResultKeys.SEARCH_CATEGORY, ''),
            '正解フラグ': 'TRUE' if is_correct else 'FALSE',
            '質問': result.get(SearchResultKeys.SEARCH_RESULT_Q, ''),
            '回答': result.get(SearchResultKeys.SEARCH_RESULT_A, ''),
            'ソース': area,
        }

    def search_revision_multi_stage(
        self,
        revision: str,
        query: str,
        correct_ids: List[str],
        provider: str
    ) -> Tuple[List[Dict], str, str, List[str]]:
        """多段階検索を実行

        Args:
            revision: 改定番号
            query: 検索クエリ
            correct_ids: 正解IDリスト
            provider: プロバイダー

        Returns:
            (検索結果リスト, LLM強化クエリ, 抽出キーワード, 検索エリアリスト)
        """
        areas = REVISION_TO_AREAS.get(revision, [])
        if not areas:
            logger.warning(f"改定 {revision} に対応するDBがありません")
            return [], "", [], ""

        all_results = []
        searched_areas = []
        llm_query = ""
        keywords = []

        for area in areas:
            # 参照クエリを取得してキャッシュ構築用に使用
            reference_queries = self._get_reference_queries(area, provider)
            if not reference_queries:
                logger.warning(f"  {area}: 参照クエリが空です")
                continue

            # オーケストレーターを作成
            orchestrator = self._create_orchestrator(provider, area, reference_queries)
            if orchestrator is None:
                continue

            try:
                # 多段階検索を実行
                results = orchestrator.execute(
                    input_number=revision,
                    query_text=query,
                    original_answer="",  # 評価では不要
                    filter_metadata=None
                )

                # キーワードとLLMクエリを取得（最初のエリアから）
                if not keywords:
                    keywords = orchestrator.keyword_engine.extract_keywords(query)
                if not llm_query and results:
                    llm_query = results[0].get(SearchResultKeys.SEARCH_QUERY, query)

                # 結果を変換
                for result in results:
                    converted = self._convert_result_to_dict(result, correct_ids, area)
                    all_results.append(converted)

                searched_areas.append(area)
                logger.info(f"  {area}: {len(results)}件取得")

            except Exception as e:
                logger.error(f"  {area} の検索エラー: {e}")
                import traceback
                traceback.print_exc()

        return all_results, llm_query, keywords, ', '.join(searched_areas)

    def _run_llm_analysis(
        self,
        results: List[Dict],
        revision_content: str
    ) -> List[Dict]:
        """LLM分析を実行

        Args:
            results: 検索結果リスト
            revision_content: 改定内容

        Returns:
            LLM分析結果を追加した検索結果リスト
        """
        if not self.enable_llm_analysis or self.judgment_support is None:
            for r in results:
                r['関連性判定'] = ''
                r['判定根拠'] = ''
                r['修正案'] = ''
            return results

        logger.info(f"LLM分析を実行中: {len(results)}件")

        for i, result in enumerate(results):
            try:
                evaluation = self.judgment_support.evaluate(
                    revision_content,
                    result.get('質問', ''),
                    result.get('回答', '')
                )
                result['関連性判定'] = evaluation.get('relevance_judgment', '')
                result['判定根拠'] = evaluation.get('judgment_reason', '')
                result['修正案'] = evaluation.get('modification_suggestion', '')

                if (i + 1) % 10 == 0:
                    logger.info(f"  LLM分析: {i + 1}/{len(results)}件完了")

            except Exception as e:
                logger.error(f"LLM分析エラー: {e}")
                result['関連性判定'] = 'エラー'
                result['判定根拠'] = str(e)[:50]
                result['修正案'] = ''

        return results

    def evaluate_revision(
        self,
        revision: str,
        revision_content: str,
        correct_ids: List[str]
    ) -> Dict[str, Any]:
        """単一の改定を評価

        Args:
            revision: 改定番号
            revision_content: 改定内容
            correct_ids: 正解IDリスト

        Returns:
            評価結果辞書 (azure_results, vertex_results, llm_query, keywords)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"改定 {revision} の評価開始")
        logger.info(f"正解ID数: {len(correct_ids)}")

        evaluation_result = {
            'revision': revision,
            'revision_content': revision_content,
            'correct_ids': correct_ids,
            'azure_results': [],
            'vertex_results': [],
            'llm_query': '',
            'keywords': [],
        }

        # Azure検索
        logger.info(f"\nAzure で検索中...")
        azure_results, llm_query, keywords, azure_areas = self.search_revision_multi_stage(
            revision, revision_content, correct_ids, 'azure_openai'
        )
        evaluation_result['azure_results'] = azure_results
        evaluation_result['llm_query'] = llm_query
        evaluation_result['keywords'] = keywords
        logger.info(f"  Azure: {len(azure_results)}件, エリア: {azure_areas}")

        # VertexAI検索
        logger.info(f"\nVertexAI で検索中...")
        vertex_results, _, _, vertex_areas = self.search_revision_multi_stage(
            revision, revision_content, correct_ids, 'vertex_ai'
        )
        evaluation_result['vertex_results'] = vertex_results
        logger.info(f"  VertexAI: {len(vertex_results)}件, エリア: {vertex_areas}")

        # LLM分析（Azure結果に対して実行）
        if self.enable_llm_analysis and azure_results:
            evaluation_result['azure_results'] = self._run_llm_analysis(
                azure_results, revision_content
            )

        # LLM分析（VertexAI結果に対して実行）
        if self.enable_llm_analysis and vertex_results:
            evaluation_result['vertex_results'] = self._run_llm_analysis(
                vertex_results, revision_content
            )

        return evaluation_result

    def evaluate_all_revisions(self) -> Dict[str, Dict[str, Any]]:
        """全改定を評価"""
        input_df = self.load_input_data()
        results_by_revision = {}

        for idx, row in input_df.iterrows():
            revision = row['番号']
            revision_content = row['改定内容']
            correct_ids_str = row.get('正解ID', '')

            # 正解IDをリストに変換
            correct_ids = [
                id.strip() for id in str(correct_ids_str).split(',')
                if id.strip()
            ]

            # 評価実行
            result = self.evaluate_revision(revision, revision_content, correct_ids)
            results_by_revision[revision] = result

        return results_by_revision

    def save_results(self, results: Dict[str, Dict[str, Any]]) -> Path:
        """結果をExcelに保存（横並びレイアウト）"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"revision_evaluation_{timestamp}.xlsx"

        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book

            # 書式定義
            header_format = workbook.add_format({
                'font_name': 'メイリオ',
                'font_size': 10,
                'bold': True,
                'border': 1,
                'bg_color': '#D9D9D9',
                'text_wrap': True,
                'valign': 'vcenter',
            })

            azure_header_format = workbook.add_format({
                'font_name': 'メイリオ',
                'font_size': 10,
                'bold': True,
                'border': 1,
                'bg_color': '#DCE6F1',  # 青系
                'text_wrap': True,
                'valign': 'vcenter',
            })

            vertex_header_format = workbook.add_format({
                'font_name': 'メイリオ',
                'font_size': 10,
                'bold': True,
                'border': 1,
                'bg_color': '#E2EFDA',  # 緑系
                'text_wrap': True,
                'valign': 'vcenter',
            })

            cell_format = workbook.add_format({
                'font_name': 'メイリオ',
                'font_size': 10,
                'border': 1,
                'text_wrap': True,
                'valign': 'top',
            })

            correct_format = workbook.add_format({
                'font_name': 'メイリオ',
                'font_size': 10,
                'border': 1,
                'bg_color': '#C6EFCE',
                'font_color': '#006100',
                'valign': 'top',
            })

            # サマリーシートを作成
            self._write_summary_sheet(writer, results, header_format, cell_format)

            # 改定ごとのシートを作成
            for revision, data in results.items():
                self._write_detail_sheet(
                    writer, revision, data,
                    header_format, azure_header_format, vertex_header_format,
                    cell_format, correct_format
                )

        logger.info(f"\n結果を保存しました: {output_file}")
        return output_file

    def _write_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        results: Dict[str, Dict[str, Any]],
        header_format,
        cell_format
    ):
        """サマリーシートを書き込み"""
        summary_data = []

        for revision, data in results.items():
            revision_content = data['revision_content']
            correct_ids = data['correct_ids']
            azure_results = data['azure_results']
            vertex_results = data['vertex_results']

            # Azure集計
            azure_count = len(azure_results)
            azure_correct = sum(1 for r in azure_results if r.get('正解フラグ') == 'TRUE')

            # VertexAI集計
            vertex_count = len(vertex_results)
            vertex_correct = sum(1 for r in vertex_results if r.get('正解フラグ') == 'TRUE')

            summary_data.append({
                '改定番号': revision,
                '改定内容（先頭50文字）': revision_content[:50] + '...' if len(revision_content) > 50 else revision_content,
                '正解ID数': len(correct_ids),
                'Azure_候補数': azure_count,
                'Azure_正解一致数': azure_correct,
                'VertexAI_候補数': vertex_count,
                'VertexAI_正解一致数': vertex_correct,
            })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, index=False, sheet_name='サマリー')

            worksheet = writer.sheets['サマリー']

            # ヘッダー書式
            for col_num, col_name in enumerate(summary_df.columns):
                worksheet.write(0, col_num, col_name, header_format)

            # 列幅
            column_widths = [10, 50, 10, 15, 18, 18, 20]
            for col_num, width in enumerate(column_widths):
                worksheet.set_column(col_num, col_num, width)

    def _write_detail_sheet(
        self,
        writer: pd.ExcelWriter,
        revision: str,
        data: Dict[str, Any],
        header_format,
        azure_header_format,
        vertex_header_format,
        cell_format,
        correct_format
    ):
        """詳細シートを書き込み（横並びレイアウト）"""
        sheet_name = revision  # シート名は改定番号のみ
        worksheet = writer.book.add_worksheet(sheet_name)

        # 共通列のヘッダー
        common_headers = ['改定内容', '正解ID一覧', 'LLM強化クエリ', '抽出キーワード', 'ベクトル重み']

        # Azure列のヘッダー
        azure_headers = [
            'Azure_シナリオID', 'Azure_類似度', 'Azure_カテゴリ', 'Azure_正解フラグ',
            'Azure_質問', 'Azure_回答', 'Azure_関連性判定', 'Azure_判定根拠',
            'Azure_修正案', 'Azure_ソース'
        ]

        # VertexAI列のヘッダー
        vertex_headers = [
            'VertexAI_シナリオID', 'VertexAI_類似度', 'VertexAI_カテゴリ', 'VertexAI_正解フラグ',
            'VertexAI_質問', 'VertexAI_回答', 'VertexAI_関連性判定', 'VertexAI_判定根拠',
            'VertexAI_修正案', 'VertexAI_ソース'
        ]

        # ヘッダー書き込み
        col = 0
        for header in common_headers:
            worksheet.write(0, col, header, header_format)
            col += 1
        for header in azure_headers:
            worksheet.write(0, col, header, azure_header_format)
            col += 1
        for header in vertex_headers:
            worksheet.write(0, col, header, vertex_header_format)
            col += 1

        # データ準備
        revision_content = data['revision_content']
        correct_ids = data['correct_ids']
        llm_query = data.get('llm_query', '')
        keywords = data.get('keywords', [])
        azure_results = data['azure_results']
        vertex_results = data['vertex_results']

        # 行数を揃える（多い方に合わせる）
        max_rows = max(len(azure_results), len(vertex_results), 1)

        # データ書き込み
        for row_num, (azure_row, vertex_row) in enumerate(
            zip_longest(azure_results, vertex_results, fillvalue={}),
            start=1
        ):
            col = 0

            # 共通列（1行目のみ）
            if row_num == 1:
                worksheet.write(row_num, col, revision_content, cell_format)
                worksheet.write(row_num, col + 1, ', '.join(correct_ids), cell_format)
                worksheet.write(row_num, col + 2, llm_query, cell_format)
                worksheet.write(row_num, col + 3, ', '.join(keywords), cell_format)
                worksheet.write(row_num, col + 4, VECTOR_WEIGHT, cell_format)
            else:
                for i in range(len(common_headers)):
                    worksheet.write(row_num, col + i, '', cell_format)
            col += len(common_headers)

            # Azure列
            azure_values = [
                azure_row.get('シナリオID', ''),
                azure_row.get('類似度', ''),
                azure_row.get('カテゴリ', ''),
                azure_row.get('正解フラグ', ''),
                azure_row.get('質問', ''),
                azure_row.get('回答', ''),
                azure_row.get('関連性判定', ''),
                azure_row.get('判定根拠', ''),
                azure_row.get('修正案', ''),
                azure_row.get('ソース', ''),
            ]
            for i, value in enumerate(azure_values):
                if i == 3 and value == 'TRUE':  # 正解フラグ
                    worksheet.write(row_num, col + i, value, correct_format)
                else:
                    worksheet.write(row_num, col + i, value if value != '' else '', cell_format)
            col += len(azure_headers)

            # VertexAI列
            vertex_values = [
                vertex_row.get('シナリオID', ''),
                vertex_row.get('類似度', ''),
                vertex_row.get('カテゴリ', ''),
                vertex_row.get('正解フラグ', ''),
                vertex_row.get('質問', ''),
                vertex_row.get('回答', ''),
                vertex_row.get('関連性判定', ''),
                vertex_row.get('判定根拠', ''),
                vertex_row.get('修正案', ''),
                vertex_row.get('ソース', ''),
            ]
            for i, value in enumerate(vertex_values):
                if i == 3 and value == 'TRUE':  # 正解フラグ
                    worksheet.write(row_num, col + i, value, correct_format)
                else:
                    worksheet.write(row_num, col + i, value if value != '' else '', cell_format)

        # 列幅設定
        column_widths = {
            0: 60,   # 改定内容
            1: 30,   # 正解ID一覧
            2: 50,   # LLM強化クエリ
            3: 25,   # 抽出キーワード
            4: 12,   # ベクトル重み
            # Azure列
            5: 18,   # シナリオID
            6: 10,   # 類似度
            7: 18,   # カテゴリ
            8: 12,   # 正解フラグ
            9: 50,   # 質問
            10: 50,  # 回答
            11: 15,  # 関連性判定
            12: 40,  # 判定根拠
            13: 40,  # 修正案
            14: 15,  # ソース
            # VertexAI列
            15: 18,  # シナリオID
            16: 10,  # 類似度
            17: 18,  # カテゴリ
            18: 12,  # 正解フラグ
            19: 50,  # 質問
            20: 50,  # 回答
            21: 15,  # 関連性判定
            22: 40,  # 判定根拠
            23: 40,  # 修正案
            24: 15,  # ソース
        }

        for col_num, width in column_widths.items():
            worksheet.set_column(col_num, col_num, width)

        # 行高さ設定
        for row_num in range(max_rows + 1):
            worksheet.set_row(row_num, 60)


def main():
    """メイン処理"""
    logger.info("事務改定評価を開始します（多段階検索・横並び比較版）")

    # DBの存在確認
    logger.info("\n=== DB存在確認 ===")
    for revision, areas in REVISION_TO_AREAS.items():
        for area in areas:
            for provider in ['azure_openai', 'vertex_ai']:
                db_path = VECTOR_DB_BASE / area / provider / "chroma.sqlite3"
                status = "OK" if db_path.exists() else "MISSING"
                logger.info(f"  {revision} {area}/{provider}: {status}")

    # 設定を初期化
    config = SearchConfig(
        base_dir=str(PROJECT_ROOT),
        top_k=MAX_RESULTS,
        multi_stage_threshold=THRESHOLD,
        multi_stage_max_results=MAX_RESULTS,
        multi_stage_enable_judgment_support=True,
    )

    # LLM分析を有効化するかの確認
    enable_llm = os.getenv("ENABLE_LLM_ANALYSIS", "true").lower() == "true"
    logger.info(f"LLM分析: {'有効' if enable_llm else '無効'}")

    # 評価を実行
    evaluator = RevisionEvaluator(config, enable_llm_analysis=enable_llm)
    results = evaluator.evaluate_all_revisions()

    # 結果を保存
    output_file = evaluator.save_results(results)

    logger.info("\n" + "=" * 60)
    logger.info("評価完了")
    logger.info(f"出力ファイル: {output_file}")


if __name__ == "__main__":
    main()
