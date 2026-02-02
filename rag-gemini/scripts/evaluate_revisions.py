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

import copy
import os
import sys
import traceback
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from config import SearchConfig, load_settings
from src.core.judgment_support import JudgmentSupport
from src.core.search.keyword_search_engine import KeywordSearchEngine
from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
from src.core.search.query_enhancer import QueryEnhancer
from src.core.search.text_combiner import get_text_combiner
from src.core.search.vector_search_engine import VectorSearchEngine
from src.types.search_types import SearchResultKeys
from src.utils.auth import create_embedding_model, create_llm
from src.utils.logger import (
    get_console,
    print_completion,
    print_revision_header,
    print_search_result,
    print_section,
    print_status,
    print_table,
    setup_logger,
)
from src.utils.vector_db import MetadataVectorDB

try:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = setup_logger(__name__)

# 設定読み込み
_settings = load_settings("evaluation")
AREA_TO_BOT = _settings["area_to_bot"]
THRESHOLD_BY_PROVIDER = _settings["thresholds"]
DEFAULT_VECTOR_WEIGHT = _settings["vector_weight"]
MAX_RESULTS = _settings["max_results"]
FILTER_MODE = _settings.get("filter_mode", "threshold")
TOP_K = _settings.get("top_k", 50)

# 新しい形式に対応（areas/vector_weightを含む辞書）
_raw_revision_areas = _settings["revision_areas"]
REVISION_TO_AREAS = {}
REVISION_VECTOR_WEIGHTS = {}

for rev, config in _raw_revision_areas.items():
    if isinstance(config, dict):
        REVISION_TO_AREAS[rev] = config.get("areas", [])
        REVISION_VECTOR_WEIGHTS[rev] = config.get("vector_weight", DEFAULT_VECTOR_WEIGHT)
    else:
        # 旧形式（リスト直接指定）への後方互換性
        REVISION_TO_AREAS[rev] = config
        REVISION_VECTOR_WEIGHTS[rev] = DEFAULT_VECTOR_WEIGHT

# パス定数
INPUT_FILE = PROJECT_ROOT / "input" / "multi_stage_input.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"
VECTOR_DB_BASE = PROJECT_ROOT / "reference" / "vector_db"


class RevisionEvaluator:
    """事務改定評価クラス（多段階検索・横並び比較版）"""

    def __init__(self, config: SearchConfig, enable_llm_analysis: bool = True):
        self.config = config
        self.enable_llm_analysis = enable_llm_analysis
        self.text_combiner = get_text_combiner()
        self.llm = create_llm(config)

        if enable_llm_analysis:
            judgment_config = copy.copy(config)
            judgment_config.multi_stage_enable_judgment_support = True
            self.judgment_support = JudgmentSupport(judgment_config)
        else:
            self.judgment_support = None

    def load_input_data(self) -> pd.DataFrame:
        if not INPUT_FILE.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {INPUT_FILE}")
        df = pd.read_excel(INPUT_FILE)
        # 変更内容列がない場合は空文字で初期化
        if "変更内容" not in df.columns:
            df["変更内容"] = ""
        logger.info(f"入力データを読み込み: {len(df)}件")
        return df

    def _fetch_scenario_content(
        self, scenario_id: str, area: str, provider: str = "azure_openai"
    ) -> Optional[Dict[str, str]]:
        """シナリオIDから質問・回答を取得"""
        try:
            bot_name, excel_row = scenario_id.rsplit("_", 1)
            row_index = int(excel_row) - 2  # Excel行→0-based index

            db_path = VECTOR_DB_BASE / area / provider
            if not db_path.exists():
                return None

            vector_db = MetadataVectorDB(db_path=str(db_path), collection_name="default")
            result = vector_db.collection.get(
                where={"row_index": row_index},
                include=["documents", "metadatas"]
            )

            if not result["documents"]:
                return None

            parsed = self.text_combiner.parse(result["documents"][0])
            return {
                "質問": parsed.query,
                "回答": parsed.answer,
                "カテゴリ": parsed.hierarchy,
            }
        except Exception as e:
            logger.warning(f"シナリオ取得エラー ({scenario_id}): {e}")
            return None

    def _get_embedding_model(self, provider: str):
        provider_config = copy.copy(self.config)
        provider_config.embedding_provider = provider

        if provider == "azure_openai":
            provider_config.embedding_model = os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
            )
        else:
            provider_config.embedding_model = os.getenv(
                "VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"
            )

        return create_embedding_model(provider_config)

    def _extract_bot_name_from_area(self, area: str) -> str:
        area_lower = area.lower()
        for keyword, bot_name in AREA_TO_BOT.items():
            if keyword in area_lower:
                return bot_name
        return "unknown-bot"

    def _filter_correct_ids_by_area(self, correct_ids: List[str], area: str) -> List[str]:
        bot_name = self._extract_bot_name_from_area(area)
        return [id for id in correct_ids if id.startswith(f"{bot_name}_")]

    def _create_orchestrator(
        self,
        provider: str,
        area: str,
        reference_queries: List[str],
        vector_weight: float,
    ) -> Optional[MultiStageOrchestrator]:
        db_path = VECTOR_DB_BASE / area / provider
        chroma_file = db_path / "chroma.sqlite3"

        if not db_path.exists():
            logger.warning(f"DBが存在しません: {db_path}")
            return None

        if not chroma_file.exists():
            logger.warning(f"ChromaDBファイルが存在しません: {chroma_file}")
            return None

        try:
            embedding_model = self._get_embedding_model(provider)
            vector_db = MetadataVectorDB(db_path=str(db_path), collection_name="default")
            vector_engine = VectorSearchEngine(
                embedding_model=embedding_model, vector_db=vector_db
            )

            keyword_engine = KeywordSearchEngine(
                stop_words=self.config.STOP_WORDS,
                position_weight=self.config.POSITION_WEIGHT,
            )
            keyword_engine.build_cache(reference_queries)

            query_enhancer = QueryEnhancer(llm=self.llm, base_dir=str(PROJECT_ROOT))

            return MultiStageOrchestrator(
                vector_engine=vector_engine,
                keyword_engine=keyword_engine,
                query_enhancer=query_enhancer,
                text_combiner=self.text_combiner,
                vector_weight=vector_weight,
                threshold=THRESHOLD_BY_PROVIDER[provider],
                max_results=MAX_RESULTS,
                filter_mode=FILTER_MODE,
                top_k=TOP_K,
            )
        except Exception as e:
            logger.error(f"オーケストレーター作成エラー ({area}/{provider}): {e}")
            traceback.print_exc()
            return None

    def _get_reference_queries(self, area: str, provider: str) -> List[str]:
        db_path = VECTOR_DB_BASE / area / provider
        if not db_path.exists():
            return []

        try:
            vector_db = MetadataVectorDB(db_path=str(db_path), collection_name="default")
            result = vector_db.collection.get(include=["documents"])
            documents = result.get("documents", [])
            if not documents:
                return []

            queries = []
            for doc in documents:
                if doc:
                    parsed = self.text_combiner.parse(doc)
                    queries.append(parsed.query if parsed.query else doc[:100])
            return queries
        except Exception as e:
            logger.error(f"参照クエリ取得エラー ({area}/{provider}): {e}")
            return []

    def _convert_result_to_dict(
        self, result: Dict[str, Any], correct_ids: List[str], area: str
    ) -> Dict[str, Any]:
        sheet_name = result.get(SearchResultKeys.SHEET_NAME, "")
        row_index = result.get(SearchResultKeys.ROW_INDEX, "")

        scenario_id = result.get(SearchResultKeys.SCENARIO_ID, "")
        if sheet_name and row_index != "":
            try:
                excel_row = int(row_index) + 2
                bot_name = self._extract_bot_name_from_area(area)
                scenario_id = f"{bot_name}_{excel_row}"
            except (ValueError, TypeError):
                pass

        return {
            "シナリオID": scenario_id,
            "類似度": round(result.get(SearchResultKeys.SIMILARITY, 0), 4),
            "カテゴリ": result.get(SearchResultKeys.SEARCH_CATEGORY, ""),
            "正解フラグ": "TRUE" if scenario_id in correct_ids else "FALSE",
            "質問": result.get(SearchResultKeys.SEARCH_RESULT_Q, ""),
            "回答": result.get(SearchResultKeys.SEARCH_RESULT_A, ""),
            "ソース": area,
        }

    def search_revision_multi_stage(
        self, revision: str, query: str, correct_ids: List[str], provider: str
    ) -> Tuple[Dict[str, List[Dict]], str, List[str], List[str]]:
        areas = REVISION_TO_AREAS.get(revision, [])
        if not areas:
            logger.warning(f"改定 {revision} に対応するDBがありません")
            return {}, "", [], []

        # 改定番号別のベクトル重みを取得
        vector_weight = REVISION_VECTOR_WEIGHTS.get(revision, DEFAULT_VECTOR_WEIGHT)

        results_by_area = {}
        searched_areas = []
        llm_query = ""
        keywords = []

        for area in areas:
            reference_queries = self._get_reference_queries(area, provider)
            if not reference_queries:
                logger.warning(f"  {area}: 参照クエリが空です")
                continue

            orchestrator = self._create_orchestrator(provider, area, reference_queries, vector_weight)
            if orchestrator is None:
                continue

            try:
                results = orchestrator.execute(
                    input_number=revision,
                    query_text=query,
                    original_answer="",
                    filter_metadata=None,
                )

                if not keywords:
                    keywords = orchestrator.keyword_engine.extract_keywords(query)
                if not llm_query and results:
                    llm_query = results[0].get(SearchResultKeys.SEARCH_QUERY, query)

                results_by_area[area] = [
                    self._convert_result_to_dict(r, correct_ids, area) for r in results
                ]
                searched_areas.append(area)
                logger.info(f"  {area}: {len(results)}件取得")
            except Exception as e:
                logger.error(f"  {area} の検索エラー: {e}")
                traceback.print_exc()

        return results_by_area, llm_query, keywords, searched_areas

    def _calculate_metrics(
        self, results: List[Dict], correct_ids: List[str]
    ) -> Dict[str, Any]:
        candidate_count = len(results)
        found_correct_count = sum(1 for r in results if r.get("正解フラグ") == "TRUE")
        total_correct = len(correct_ids)
        discovery_rate = (found_correct_count / total_correct) if total_correct > 0 else 0

        last_correct_rank = 0
        for i, r in enumerate(results, start=1):
            if r.get("正解フラグ") == "TRUE":
                last_correct_rank = i

        return {
            "候補数": candidate_count,
            "正解発見数": found_correct_count,
            "正解発見率": discovery_rate,
            "必要確認件数": last_correct_rank if last_correct_rank > 0 else "-",
        }

    def _evaluate_single_result(self, result: Dict, revision_content: str) -> None:
        try:
            evaluation = self.judgment_support.evaluate(
                revision_content,
                result.get("質問", ""),
                result.get("回答", ""),
            )
            result["関連性判定"] = evaluation.get("relevance_judgment", "")
            result["判定根拠"] = evaluation.get("judgment_reason", "")
        except Exception as e:
            logger.error(f"LLM分析エラー: {e}")
            result["関連性判定"] = "エラー"
            result["判定根拠"] = str(e)[:50]

    def _run_llm_analysis(
        self, results: List[Dict], revision_content: str
    ) -> List[Dict]:
        if not self.enable_llm_analysis or self.judgment_support is None:
            for r in results:
                r["関連性判定"] = ""
                r["判定根拠"] = ""
            return results

        total = len(results)

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=get_console(),
                transient=True,
            ) as progress:
                task = progress.add_task("[cyan]LLM分析中...", total=total)
                for result in results:
                    self._evaluate_single_result(result, revision_content)
                    progress.update(task, advance=1)
            print_status(f"LLM分析完了: {total}件", "success")
        else:
            logger.info(f"LLM分析を実行中: {total}件")
            for i, result in enumerate(results):
                self._evaluate_single_result(result, revision_content)
                if (i + 1) % 10 == 0:
                    logger.info(f"  LLM分析: {i + 1}/{total}件完了")

        return results

    def evaluate_revision(
        self,
        revision: str,
        revision_content: str,
        correct_ids: List[str],
        change_details_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        # 改定番号別のベクトル重みを取得
        vector_weight = REVISION_VECTOR_WEIGHTS.get(revision, DEFAULT_VECTOR_WEIGHT)

        # change_details_mapがない場合は空の辞書で初期化
        if change_details_map is None:
            change_details_map = {}

        evaluation_result = {
            "revision": revision,
            "revision_content": revision_content,
            "correct_ids": correct_ids,
            "change_details_map": change_details_map,
            "areas": [],
            "by_area": {},
            "llm_query": "",
            "keywords": [],
            "vector_weight": vector_weight,
        }

        # Azure検索
        azure_results_by_area, llm_query, keywords, azure_areas = (
            self.search_revision_multi_stage(
                revision, revision_content, correct_ids, "azure_openai"
            )
        )
        evaluation_result["llm_query"] = llm_query
        evaluation_result["keywords"] = keywords

        total_azure = sum(len(r) for r in azure_results_by_area.values())
        azure_correct = sum(
            1
            for area_results in azure_results_by_area.values()
            for r in area_results
            if r.get("正解フラグ") == "TRUE"
        )
        print_search_result(
            "azure", total_azure, azure_areas, azure_correct, len(correct_ids)
        )

        # VertexAI検索
        vertex_results_by_area, _, _, vertex_areas = self.search_revision_multi_stage(
            revision, revision_content, correct_ids, "vertex_ai"
        )
        total_vertex = sum(len(r) for r in vertex_results_by_area.values())
        vertex_correct = sum(
            1
            for area_results in vertex_results_by_area.values()
            for r in area_results
            if r.get("正解フラグ") == "TRUE"
        )
        print_search_result(
            "vertex", total_vertex, vertex_areas, vertex_correct, len(correct_ids)
        )

        all_areas = sorted(set(azure_areas) | set(vertex_areas))
        evaluation_result["areas"] = all_areas

        for area in all_areas:
            area_correct_ids = self._filter_correct_ids_by_area(correct_ids, area)
            azure_results = azure_results_by_area.get(area, [])
            vertex_results = vertex_results_by_area.get(area, [])

            if self.enable_llm_analysis:
                if azure_results:
                    azure_results = self._run_llm_analysis(azure_results, revision_content)
                if vertex_results:
                    vertex_results = self._run_llm_analysis(vertex_results, revision_content)

            # 検索結果から発見済みシナリオIDを収集
            found_ids = set()
            for result in azure_results:
                if result.get("正解フラグ") == "TRUE":
                    found_ids.add(result["シナリオID"])
            for result in vertex_results:
                if result.get("正解フラグ") == "TRUE":
                    found_ids.add(result["シナリオID"])

            # 未発見シナリオを特定（エリアに該当する正解IDのみ）
            unfound_scenarios = []
            for scenario_id in area_correct_ids:
                if scenario_id not in found_ids:
                    # ベクトルDBから内容を取得
                    content = self._fetch_scenario_content(scenario_id, area)
                    unfound_scenarios.append({
                        "シナリオID": scenario_id,
                        "変更内容": change_details_map.get(scenario_id, ""),
                        "質問": content.get("質問", "") if content else "",
                        "回答": content.get("回答", "") if content else "",
                        "カテゴリ": content.get("カテゴリ", "") if content else "",
                    })

            evaluation_result["by_area"][area] = {
                "azure_results": azure_results,
                "vertex_results": vertex_results,
                "correct_ids": area_correct_ids,
                "unfound_scenarios": unfound_scenarios,
            }

        return evaluation_result

    def evaluate_all_revisions(self) -> Dict[str, Dict[str, Any]]:
        input_df = self.load_input_data()
        results_by_revision = {}

        # 改定番号でグループ化
        grouped = input_df.groupby("番号", sort=False)
        total_revisions = len(grouped)

        print_section(f"評価対象: {total_revisions}件の改定")

        revision_list_data = []
        for revision, group in grouped:
            content = group.iloc[0]["改定内容"]
            content_preview = content[:40] + "..." if len(content) > 40 else content
            correct_count = len(group)
            revision_list_data.append((revision, content_preview, correct_count))

        print_table("改定一覧", revision_list_data, ["番号", "改定内容", "正解数"])

        for idx, (revision, group) in enumerate(grouped):
            revision_content = group.iloc[0]["改定内容"]
            correct_ids = group["正解ID"].tolist()

            # 正解IDと変更内容の辞書を構築
            change_details_map = {
                row["正解ID"]: row["変更内容"]
                for _, row in group.iterrows()
            }

            print_revision_header(
                revision=revision,
                content=revision_content,
                correct_count=len(correct_ids),
                current=idx + 1,
                total=total_revisions,
            )

            results_by_revision[revision] = self.evaluate_revision(
                revision, revision_content, correct_ids, change_details_map
            )

        return results_by_revision

    def save_results(self, results: Dict[str, Dict[str, Any]]) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"revision_evaluation_{timestamp}.xlsx"

        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            workbook = writer.book
            formats = self._create_excel_formats(workbook)
            self._write_summary_sheet(writer, results, formats)

            for revision, data in results.items():
                self._write_detail_sheets(writer, revision, data, formats)

        logger.info(f"\n結果を保存しました: {output_file}")
        return output_file

    def _create_excel_formats(self, workbook) -> Dict[str, Any]:
        base_style = {"font_name": "Meiryo UI", "font_size": 10, "border": 1}
        header_style = {**base_style, "bold": True, "align": "center", "valign": "vcenter"}

        return {
            "header": workbook.add_format({**header_style, "bg_color": "#D9D9D9", "text_wrap": True}),
            "azure_header": workbook.add_format({**header_style, "bg_color": "#DCE6F1", "text_wrap": True}),
            "vertex_header": workbook.add_format({**header_style, "bg_color": "#E2EFDA", "text_wrap": True}),
            "unfound_header": workbook.add_format({**header_style, "bg_color": "#FDE9D9", "text_wrap": True}),
            "cell": workbook.add_format({**base_style, "valign": "top", "text_wrap": True}),
            "correct": workbook.add_format({
                **base_style, "bg_color": "#C6EFCE", "font_color": "#006100", "valign": "top"
            }),
            "unfound_cell": workbook.add_format({
                **base_style, "bg_color": "#FDE9D9", "valign": "top", "text_wrap": True
            }),
            "percent": workbook.add_format({**base_style, "num_format": "0.0%", "valign": "top"}),
        }

    def _write_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        results: Dict[str, Dict[str, Any]],
        formats: Dict[str, Any],
    ) -> None:
        workbook = writer.book
        summary_data = []

        for revision, data in results.items():
            revision_content = data["revision_content"]
            areas = data.get("areas", [])
            by_area = data.get("by_area", {})

            if not areas:
                correct_ids = data.get("correct_ids", [])
                summary_data.append(
                    self._create_empty_summary_row(revision, revision_content, correct_ids)
                )
                continue

            for area in areas:
                area_data = by_area.get(area, {})
                area_correct_ids = area_data.get("correct_ids", [])
                azure_metrics = self._calculate_metrics(
                    area_data.get("azure_results", []), area_correct_ids
                )
                vertex_metrics = self._calculate_metrics(
                    area_data.get("vertex_results", []), area_correct_ids
                )

                # 未発見情報
                unfound_scenarios = area_data.get("unfound_scenarios", [])
                unfound_count = len(unfound_scenarios)
                unfound_ids = ", ".join([s["シナリオID"] for s in unfound_scenarios])

                content_preview = (
                    revision_content[:50] + "..."
                    if len(revision_content) > 50
                    else revision_content
                )
                summary_data.append({
                    "改定番号": revision,
                    "エリア": area,
                    "改定内容": content_preview,
                    "正解数": len(area_correct_ids),
                    "Azure_候補数": azure_metrics["候補数"],
                    "Azure_正解発見数": azure_metrics["正解発見数"],
                    "Azure_正解発見率": azure_metrics["正解発見率"],
                    "Azure_必要確認件数": azure_metrics["必要確認件数"],
                    "VertexAI_候補数": vertex_metrics["候補数"],
                    "VertexAI_正解発見数": vertex_metrics["正解発見数"],
                    "VertexAI_正解発見率": vertex_metrics["正解発見率"],
                    "VertexAI_必要確認件数": vertex_metrics["必要確認件数"],
                    "未発見数": unfound_count,
                    "未発見ID": unfound_ids,
                })

        if not summary_data:
            return

        worksheet = workbook.add_worksheet("サマリー")
        self._write_summary_headers(worksheet, formats)
        self._write_summary_data(worksheet, summary_data, formats)

        column_widths = [10, 20, 50, 8, 8, 12, 12, 18, 8, 12, 12, 18, 10, 40]
        for col_num, width in enumerate(column_widths):
            worksheet.set_column(col_num, col_num, width)

    def _create_empty_summary_row(
        self, revision: str, revision_content: str, correct_ids: List[str]
    ) -> Dict[str, Any]:
        content_preview = (
            revision_content[:50] + "..."
            if len(revision_content) > 50
            else revision_content
        )
        return {
            "改定番号": revision,
            "エリア": "-",
            "改定内容": content_preview,
            "正解数": len(correct_ids),
            "Azure_候補数": 0,
            "Azure_正解発見数": 0,
            "Azure_正解発見率": 0,
            "Azure_必要確認件数": "-",
            "VertexAI_候補数": 0,
            "VertexAI_正解発見数": 0,
            "VertexAI_正解発見率": 0,
            "VertexAI_必要確認件数": "-",
            "未発見数": len(correct_ids),
            "未発見ID": ", ".join(correct_ids),
        }

    def _write_summary_headers(self, worksheet, formats: Dict[str, Any]) -> None:
        header_fmt = formats["header"]
        azure_fmt = formats["azure_header"]
        vertex_fmt = formats["vertex_header"]
        unfound_fmt = formats["unfound_header"]

        for col in range(4):
            worksheet.write(0, col, "", header_fmt)
        worksheet.merge_range("E1:H1", "Azure", azure_fmt)
        worksheet.merge_range("I1:L1", "VertexAI", vertex_fmt)
        worksheet.merge_range("M1:N1", "未発見", unfound_fmt)

        headers = [
            "改定番号", "エリア", "改定内容", "正解数",
            "候補数", "正解発見数", "正解発見率", "必要確認件数",
            "候補数", "正解発見数", "正解発見率", "必要確認件数",
            "未発見数", "未発見ID",
        ]
        for col, header in enumerate(headers):
            if col < 4:
                fmt = header_fmt
            elif col < 8:
                fmt = azure_fmt
            elif col < 12:
                fmt = vertex_fmt
            else:
                fmt = unfound_fmt
            worksheet.write(1, col, header, fmt)

    def _write_summary_data(
        self, worksheet, summary_data: List[Dict], formats: Dict[str, Any]
    ) -> None:
        cell_fmt = formats["cell"]
        percent_fmt = formats["percent"]
        unfound_fmt = formats["unfound_cell"]

        for row_num, row_data in enumerate(summary_data, start=2):
            worksheet.write(row_num, 0, row_data["改定番号"], cell_fmt)
            worksheet.write(row_num, 1, row_data["エリア"], cell_fmt)
            worksheet.write(row_num, 2, row_data["改定内容"], cell_fmt)
            worksheet.write(row_num, 3, row_data["正解数"], cell_fmt)
            worksheet.write(row_num, 4, row_data["Azure_候補数"], cell_fmt)
            worksheet.write(row_num, 5, row_data["Azure_正解発見数"], cell_fmt)
            worksheet.write(row_num, 6, row_data["Azure_正解発見率"], percent_fmt)
            worksheet.write(row_num, 7, row_data["Azure_必要確認件数"], cell_fmt)
            worksheet.write(row_num, 8, row_data["VertexAI_候補数"], cell_fmt)
            worksheet.write(row_num, 9, row_data["VertexAI_正解発見数"], cell_fmt)
            worksheet.write(row_num, 10, row_data["VertexAI_正解発見率"], percent_fmt)
            worksheet.write(row_num, 11, row_data["VertexAI_必要確認件数"], cell_fmt)
            worksheet.write(row_num, 12, row_data.get("未発見数", 0), unfound_fmt)
            worksheet.write(row_num, 13, row_data.get("未発見ID", ""), unfound_fmt)

    def _write_detail_sheets(
        self,
        writer: pd.ExcelWriter,
        revision: str,
        data: Dict[str, Any],
        formats: Dict[str, Any],
    ) -> None:
        """複数エリアの場合、エリアごとに詳細シートを作成"""
        areas = data.get("areas", [])

        if len(areas) <= 1:
            # 単一エリアの場合は従来通り
            self._write_single_detail_sheet(writer, revision, data, formats)
        else:
            # 複数エリアの場合はエリアごとにシートを作成
            for area in areas:
                # エリア名から短縮名を抽出（例: rev03naibujimu → naibujimu）
                area_short = area
                for prefix in ["rev01", "rev02", "rev03", "rev04", "rev05", "rev06"]:
                    if area.startswith(prefix):
                        area_short = area[len(prefix):]
                        break

                sheet_name = f"{revision}_{area_short}"

                # エリア固有のデータを構築
                area_data = {
                    "revision_content": data["revision_content"],
                    "correct_ids": self._filter_correct_ids_by_area(data["correct_ids"], area),
                    "llm_query": data.get("llm_query", ""),
                    "keywords": data.get("keywords", []),
                    "areas": [area],
                    "by_area": {area: data.get("by_area", {}).get(area, {})},
                    "vector_weight": data.get("vector_weight", DEFAULT_VECTOR_WEIGHT),
                }
                self._write_single_detail_sheet(writer, sheet_name, area_data, formats)

    def _write_single_detail_sheet(
        self,
        writer: pd.ExcelWriter,
        sheet_name: str,
        data: Dict[str, Any],
        formats: Dict[str, Any],
    ) -> None:
        worksheet = writer.book.add_worksheet(sheet_name)

        common_headers = ["改定内容", "正解ID一覧", "LLM強化クエリ", "抽出キーワード", "ベクトル重み"]
        result_headers = ["シナリオID", "類似度", "カテゴリ", "正解フラグ", "質問", "回答", "関連性判定", "判定根拠", "ソース"]
        unfound_headers = ["未発見ID", "変更内容", "カテゴリ", "質問", "回答"]

        col = 0
        for header in common_headers:
            worksheet.write(0, col, header, formats["header"])
            col += 1
        for header in result_headers:
            worksheet.write(0, col, f"Azure_{header}", formats["azure_header"])
            col += 1
        for header in result_headers:
            worksheet.write(0, col, f"VertexAI_{header}", formats["vertex_header"])
            col += 1
        for header in unfound_headers:
            worksheet.write(0, col, f"未発見_{header}", formats["unfound_header"])
            col += 1

        azure_results = []
        vertex_results = []
        unfound_scenarios = []
        for area in data.get("areas", []):
            area_data = data.get("by_area", {}).get(area, {})
            azure_results.extend(area_data.get("azure_results", []))
            vertex_results.extend(area_data.get("vertex_results", []))
            unfound_scenarios.extend(area_data.get("unfound_scenarios", []))

        max_rows = max(len(azure_results), len(vertex_results), len(unfound_scenarios), 1)

        for row_num, (azure_row, vertex_row, unfound_row) in enumerate(
            zip_longest(azure_results, vertex_results, unfound_scenarios, fillvalue={}), start=1
        ):
            col = 0

            if row_num == 1:
                worksheet.write(row_num, 0, data["revision_content"], formats["cell"])
                worksheet.write(row_num, 1, ", ".join(data["correct_ids"]), formats["cell"])
                worksheet.write(row_num, 2, data.get("llm_query", ""), formats["cell"])
                worksheet.write(row_num, 3, ", ".join(data.get("keywords", [])), formats["cell"])
                worksheet.write(row_num, 4, data.get("vector_weight", DEFAULT_VECTOR_WEIGHT), formats["cell"])
            else:
                for i in range(len(common_headers)):
                    worksheet.write(row_num, i, "", formats["cell"])

            col = len(common_headers)
            self._write_result_row(worksheet, row_num, col, azure_row, formats)
            col += len(result_headers)
            self._write_result_row(worksheet, row_num, col, vertex_row, formats)
            col += len(result_headers)
            self._write_unfound_row(worksheet, row_num, col, unfound_row, formats)

        # 列幅設定（common + azure + vertex + unfound）
        column_widths = [60, 30, 50, 25, 12] + [18, 10, 18, 12, 50, 50, 15, 40, 15] * 2 + [18, 12, 18, 50, 50]
        for col_num, width in enumerate(column_widths):
            worksheet.set_column(col_num, col_num, width)

        for row_num in range(max_rows + 1):
            worksheet.set_row(row_num, 60)

    def _write_result_row(
        self, worksheet, row_num: int, start_col: int, row_data: Dict, formats: Dict[str, Any]
    ) -> None:
        keys = ["シナリオID", "類似度", "カテゴリ", "正解フラグ", "質問", "回答", "関連性判定", "判定根拠", "ソース"]
        for i, key in enumerate(keys):
            value = row_data.get(key, "")
            fmt = formats["correct"] if key == "正解フラグ" and value == "TRUE" else formats["cell"]
            worksheet.write(row_num, start_col + i, value if value != "" else "", fmt)

    def _write_unfound_row(
        self, worksheet, row_num: int, start_col: int, row_data: Dict, formats: Dict[str, Any]
    ) -> None:
        keys = ["シナリオID", "変更内容", "カテゴリ", "質問", "回答"]
        for i, key in enumerate(keys):
            value = row_data.get(key, "")
            worksheet.write(row_num, start_col + i, value if value != "" else "", formats["unfound_cell"])


def main() -> None:
    print_section("事務改定評価 (多段階検索・横並び比較版)")

    print_section("DB存在確認")
    db_status_data = []
    for revision, areas in REVISION_TO_AREAS.items():
        for area in areas:
            azure_path = VECTOR_DB_BASE / area / "azure_openai" / "chroma.sqlite3"
            vertex_path = VECTOR_DB_BASE / area / "vertex_ai" / "chroma.sqlite3"
            azure_status = "[green]OK[/green]" if azure_path.exists() else "[red]MISSING[/red]"
            vertex_status = "[green]OK[/green]" if vertex_path.exists() else "[red]MISSING[/red]"
            db_status_data.append((revision, area, azure_status, vertex_status))

    print_table("ベクトルDB状態", db_status_data, ["改定", "エリア", "Azure", "VertexAI"])

    config = SearchConfig(
        base_dir=str(PROJECT_ROOT),
        top_k=MAX_RESULTS,
        multi_stage_threshold=THRESHOLD_BY_PROVIDER["azure_openai"],
        multi_stage_max_results=MAX_RESULTS,
        multi_stage_enable_judgment_support=True,
    )

    enable_llm = os.getenv("ENABLE_LLM_ANALYSIS", "false").lower() == "true"

    print_section("評価設定")
    print_status(f"LLM分析: {'[green]有効[/green]' if enable_llm else '[yellow]無効[/yellow]'}", "info")
    print_status(f"最大検索結果数: {MAX_RESULTS}", "info")
    print_status(f"デフォルトベクトル重み: {DEFAULT_VECTOR_WEIGHT}", "info")

    # 改定番号別ベクトル重み表示
    custom_weights = [(rev, w) for rev, w in REVISION_VECTOR_WEIGHTS.items() if w != DEFAULT_VECTOR_WEIGHT]
    if custom_weights:
        weight_str = ", ".join([f"{rev}={w}" for rev, w in custom_weights])
        print_status(f"カスタム重み: {weight_str}", "info")

    print_status(f"フィルタモード: {FILTER_MODE}", "info")
    if FILTER_MODE == "top_k":
        print_status(f"TOP-K: {TOP_K}件", "info")
    else:
        print_status(f"閾値 (Azure): {THRESHOLD_BY_PROVIDER['azure_openai']}", "info")
        print_status(f"閾値 (VertexAI): {THRESHOLD_BY_PROVIDER['vertex_ai']}", "info")

    evaluator = RevisionEvaluator(config, enable_llm_analysis=enable_llm)
    results = evaluator.evaluate_all_revisions()
    output_file = evaluator.save_results(results)
    print_completion(str(output_file))


if __name__ == "__main__":
    main()
