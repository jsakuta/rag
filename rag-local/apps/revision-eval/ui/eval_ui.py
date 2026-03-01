"""事務改定評価AI — 評価 Streamlit UI

改定番号を選択し、Azure OpenAI / VertexAI 両方で検索して正解IDとのマッチを評価する。
バッチ版は evaluate_revisions.py（Excel出力）。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
import copy
import html
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import SearchConfig, load_settings
from src.utils.logger import setup_logger

# 共通UI部品
from ui.shared import (
    format_message, format_response_card, render_vector_weight_slider,
    apply_common_styles, initialize_common_session_state,
)

logger = setup_logger(__name__)

# 評価設定
_eval_settings = load_settings("evaluation")
AREA_TO_BOT = _eval_settings.get("area_to_bot", {})
AREA_TO_CATEGORY = _eval_settings.get("area_to_category", {})
CATEGORY_TO_AREA = {v: k for k, v in AREA_TO_CATEGORY.items()}

# 改定番号設定を解析
_raw_revision_areas = _eval_settings.get("revision_areas", {})
REVISION_TO_AREAS = {}
REVISION_VECTOR_WEIGHTS = {}
REVISION_SEARCH_TYPES = {}
DEFAULT_VECTOR_WEIGHT = _eval_settings.get("vector_weight", 0.9)

for rev, config in _raw_revision_areas.items():
    if isinstance(config, dict):
        REVISION_TO_AREAS[rev] = config.get("areas", [])
        REVISION_VECTOR_WEIGHTS[rev] = config.get("vector_weight", DEFAULT_VECTOR_WEIGHT)
        REVISION_SEARCH_TYPES[rev] = config.get("search_type", "hybrid")
    else:
        REVISION_TO_AREAS[rev] = config
        REVISION_VECTOR_WEIGHTS[rev] = DEFAULT_VECTOR_WEIGHT
        REVISION_SEARCH_TYPES[rev] = "hybrid"

INPUT_FILE = PROJECT_ROOT / "data" / "input" / "multi_stage_input.xlsx"


@st.cache_data(ttl=300)
def load_revision_correct_ids() -> Dict[str, Tuple[str, List[str]]]:
    """改定番号 → (改定内容, 正解IDリスト) マッピングを読み込み"""
    if not INPUT_FILE.exists():
        logger.warning(f"正解IDファイルが見つかりません: {INPUT_FILE}")
        return {}

    try:
        df = pd.read_excel(INPUT_FILE)
        result = {}
        for revision, group in df.groupby("番号", sort=False):
            revision_content = group.iloc[0]["改定内容"]
            correct_ids = group["正解ID"].tolist()
            result[revision] = (revision_content, correct_ids)

        logger.info(f"正解IDデータを読み込み: {len(result)}改定, {sum(len(v[1]) for v in result.values())}件")
        return result
    except Exception as e:
        logger.error(f"正解IDデータの読み込みエラー: {e}")
        return {}


def extract_bot_name_from_category(category: str) -> str:
    """カテゴリ名からボット名を抽出"""
    area = CATEGORY_TO_AREA.get(category)
    if area:
        return AREA_TO_BOT.get(area, "unknown-bot")
    return "unknown-bot"


def extract_bot_name_from_area(area: str) -> str:
    """エリア名からボット名を抽出"""
    area_lower = area.lower()
    for keyword, bot_name in AREA_TO_BOT.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"


def build_scenario_id(result: Dict) -> str:
    """検索結果からシナリオIDを構築"""
    sheet_name = result.get("Sheet_Name", "")
    row_index = result.get("Row_Index", "")

    if not sheet_name or row_index == "":
        return ""

    try:
        excel_row = int(row_index) + 2
        bot_name = extract_bot_name_from_category(sheet_name)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""


def build_scenario_id_from_area(result: Dict, area: str) -> str:
    """エリア名を使用してシナリオIDを構築"""
    row_index = result.get("Row_Index", "")
    if row_index == "":
        return ""

    try:
        excel_row = int(row_index) + 2
        bot_name = extract_bot_name_from_area(area)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""


def check_if_correct(result: Dict, correct_ids: List[str], area: Optional[str] = None) -> Tuple[str, bool]:
    """検索結果が正解IDとマッチするか判定"""
    if area:
        scenario_id = build_scenario_id_from_area(result, area)
    else:
        scenario_id = build_scenario_id(result)

    is_correct = scenario_id in correct_ids if scenario_id else False
    return scenario_id, is_correct


@st.cache_resource(ttl=3600)
def _get_cached_keyword_searcher():
    """ChromaDBKeywordSearcher をキャッシュ（TTL=1時間）"""
    from src.core.search.keyword_search_engine import KeywordSearchEngine
    from src.core.search.chromadb_keyword_search import ChromaDBKeywordSearcher

    config = SearchConfig(base_dir=str(PROJECT_ROOT))
    keyword_engine = KeywordSearchEngine(
        stop_words=config.STOP_WORDS,
        position_weight=config.POSITION_WEIGHT,
    )
    return ChromaDBKeywordSearcher(
        base_db_path=str(PROJECT_ROOT / "data" / "vector_db"),
        keyword_engine=keyword_engine,
        area_to_bot=AREA_TO_BOT,
        area_to_category=AREA_TO_CATEGORY,
    )


def initialize_session_state():
    initialize_common_session_state()
    if "correct_ids" not in st.session_state:
        st.session_state.correct_ids = []
    if "selected_revision" not in st.session_state:
        st.session_state.selected_revision = None
    if "azure_results" not in st.session_state:
        st.session_state.azure_results = []
    if "vertex_results" not in st.session_state:
        st.session_state.vertex_results = []
    # 影響調査モード用
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "evaluation"
    if "impact_categories" not in st.session_state:
        st.session_state.impact_categories = ["naibujimu"]
    if "impact_source_filter" not in st.session_state:
        st.session_state.impact_source_filter = "scenario"
    if "selected_providers" not in st.session_state:
        st.session_state.selected_providers = "both"


def execute_dual_provider_search(query: str, revision: str) -> Tuple[List[Dict], List[Dict], str]:
    """Azure/VertexAI両方で検索を実行"""
    config = st.session_state.config
    app_mode = st.session_state.get("app_mode", "evaluation")

    # 影響調査モード
    if app_mode == "impact_analysis":
        search_type = getattr(config, "search_type", "hybrid")
        categories = st.session_state.get("impact_categories", ["naibujimu"])
        source_filter = st.session_state.get("impact_source_filter")

        if search_type == "keyword_filter":
            results = _execute_impact_keyword_search(query, categories, source_filter)
            selected_providers = st.session_state.get("selected_providers", "both")
            if selected_providers == "vertex_ai":
                return [], results, ""
            return results, [], ""
        else:
            # hybrid: 意味検索（Azure/VertexAI）
            vector_weight = getattr(config, "vector_weight", DEFAULT_VECTOR_WEIGHT)
            selected_providers = st.session_state.get("selected_providers", "both")
            azure_results = []
            vertex_results = []
            if selected_providers in ("both", "azure_openai"):
                azure_results = _search_with_provider(query, "", "azure_openai", categories, vector_weight, source_filter=source_filter)
            if selected_providers in ("both", "vertex_ai"):
                vertex_results = _search_with_provider(query, "", "vertex_ai", categories, vector_weight, source_filter=source_filter)
            # source_filter を Python 側で適用
            if source_filter:
                azure_results = [r for r in azure_results if r.get("_source", "") == source_filter or "_source" not in r]
                vertex_results = [r for r in vertex_results if r.get("_source", "") == source_filter or "_source" not in r]
            llm_query = ""
            if azure_results:
                llm_query = azure_results[0].get("Search_Query", query)
            elif vertex_results:
                llm_query = vertex_results[0].get("Search_Query", query)
            return azure_results, vertex_results, llm_query

    # 評価モード
    search_type = getattr(config, "search_type", None) or REVISION_SEARCH_TYPES.get(revision, "hybrid")
    areas = REVISION_TO_AREAS.get(revision, [])
    vector_weight = REVISION_VECTOR_WEIGHTS.get(revision, DEFAULT_VECTOR_WEIGHT)

    if not areas:
        logger.warning(f"改定 {revision} に対応するエリアがありません")
        return [], [], ""

    if search_type == "keyword_filter":
        keyword_results = _execute_keyword_filter_search(query, revision, areas)
        selected_providers = st.session_state.get("selected_providers", "both")
        if selected_providers == "vertex_ai":
            return [], keyword_results, ""
        return keyword_results, [], ""

    selected_providers = st.session_state.get("selected_providers", "both")
    azure_results = []
    vertex_results = []
    if selected_providers in ("both", "azure_openai"):
        azure_results = _search_with_provider(query, revision, "azure_openai", areas, vector_weight)
    if selected_providers in ("both", "vertex_ai"):
        vertex_results = _search_with_provider(query, revision, "vertex_ai", areas, vector_weight)

    llm_query = ""
    if azure_results:
        llm_query = azure_results[0].get("Search_Query", query)
    elif vertex_results:
        llm_query = vertex_results[0].get("Search_Query", query)

    return azure_results, vertex_results, llm_query


def _execute_keyword_filter_search(query: str, revision: str, areas: List[str]) -> List[Dict]:
    """キーワード検索（ChromaDB、LLM不使用）"""
    searcher = _get_cached_keyword_searcher()

    # areas は既に "rev02_souzoku" 等のフルネーム（旧コードは全件返却）
    matches = searcher.search(areas, query, provider="azure_openai", max_results=10000)

    # UI版フォーマットに変換
    return [
        {
            "Similarity": m.similarity,
            "Search_Result_Q": m.question,
            "Search_Result_A": m.answer,
            "Search_Category": "Keyword",
            "Sheet_Name": AREA_TO_CATEGORY.get(m.area, m.collection_name),
            "Row_Index": m.row_index,
            "Search_Query": "",
            "_area": m.area,
        }
        for m in matches
    ]


def _execute_impact_keyword_search(query: str, categories: List[str], source_filter: Optional[str] = None) -> List[Dict]:
    """影響調査モード: キーワード検索"""
    searcher = _get_cached_keyword_searcher()

    matches = searcher.search(
        categories, query, provider="azure_openai",
        max_results=10000, source_filter=source_filter,
    )

    return [
        {
            "Similarity": m.similarity,
            "Search_Result_Q": m.question,
            "Search_Result_A": m.answer,
            "Search_Category": "Keyword",
            "Sheet_Name": AREA_TO_CATEGORY.get(m.collection_name, m.collection_name),
            "Row_Index": m.row_index,
            "Search_Query": "",
            "_area": m.collection_name,
            "_source": m.source,
        }
        for m in matches
    ]


def _search_with_provider(query: str, revision: str, provider: str, areas: List[str], vector_weight: float, source_filter: Optional[str] = None) -> List[Dict]:
    """特定のプロバイダーでハイブリッド検索を実行"""
    from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
    from src.core.search.vector_search_engine import VectorSearchEngine
    from src.core.search.keyword_search_engine import KeywordSearchEngine
    from src.core.search.query_enhancer import QueryEnhancer
    from src.core.search.text_combiner import get_text_combiner
    from src.utils.auth import create_embedding_model, create_llm
    from src.utils.vector_db import MetadataVectorDB

    config = st.session_state.config
    VECTOR_DB_BASE = PROJECT_ROOT / "data" / "vector_db"
    all_results = []

    for area in areas:
        db_path = VECTOR_DB_BASE / area / provider
        if not db_path.exists():
            logger.warning(f"DBが存在しません: {db_path}")
            continue

        try:
            provider_config = copy.copy(config)
            provider_config.embedding_provider = provider
            if provider == "azure_openai":
                provider_config.embedding_model = os.getenv(
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
                )
            else:
                provider_config.embedding_model = os.getenv(
                    "VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"
                )
            embedding_model = create_embedding_model(provider_config)

            vector_db = MetadataVectorDB(db_path=str(db_path), collection_name="default")

            text_combiner = get_text_combiner()
            if source_filter:
                result = vector_db.collection.get(include=["documents", "metadatas"])
                metadatas = result.get("metadatas", [])
            else:
                result = vector_db.collection.get(include=["documents"])
                metadatas = None
            documents = result.get("documents", [])
            reference_queries = []
            for idx, doc in enumerate(documents):
                # source_filter: 非マッチ文書はスキップ（インデックス維持のため空文字）
                if source_filter and metadatas and metadatas[idx].get("source") != source_filter:
                    reference_queries.append("")
                    continue
                if doc:
                    parsed = text_combiner.parse(doc)
                    reference_queries.append(parsed.query if parsed.query else doc[:100])
                else:
                    reference_queries.append("")

            if not any(reference_queries):
                continue

            vector_engine = VectorSearchEngine(
                embedding_model=embedding_model, vector_db=vector_db
            )
            keyword_engine = KeywordSearchEngine(
                stop_words=config.STOP_WORDS,
                position_weight=config.POSITION_WEIGHT,
            )
            keyword_engine.build_cache(reference_queries)

            llm = create_llm(config)
            query_enhancer = QueryEnhancer(llm=llm, base_dir=str(PROJECT_ROOT))

            ui_top_k = config.top_k if hasattr(config, 'top_k') else _eval_settings.get("top_k", 50)
            orchestrator = MultiStageOrchestrator(
                vector_engine=vector_engine,
                keyword_engine=keyword_engine,
                query_enhancer=query_enhancer,
                text_combiner=text_combiner,
                vector_weight=vector_weight,
                threshold=_eval_settings["thresholds"].get(provider, 0.5),
                max_results=_eval_settings.get("max_results", 100),
                filter_mode="top_k",
                top_k=ui_top_k,
            )

            results = orchestrator.execute(
                input_number=revision,
                query_text=query,
                original_answer="",
                filter_metadata=None,
            )

            for r in results:
                all_results.append({
                    "Similarity": r.get("Similarity", 0),
                    "Search_Result_Q": r.get("Search_Result_Q", ""),
                    "Search_Result_A": r.get("Search_Result_A", ""),
                    "Search_Category": r.get("Search_Category", ""),
                    "Sheet_Name": r.get("Sheet_Name", ""),
                    "Row_Index": r.get("Row_Index", ""),
                    "Search_Query": r.get("Search_Query", ""),
                    "_area": area,
                })

        except Exception as e:
            logger.error(f"検索エラー ({area}/{provider}): {e}")
            import traceback
            traceback.print_exc()

    all_results.sort(key=lambda x: x.get("Similarity", 0), reverse=True)
    return all_results


def save_chat_history():
    """チャット履歴を保存"""
    try:
        chat_data = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_query = st.session_state.chat_history[i]["text"]
                bot_response = st.session_state.chat_history[i + 1]["text"]

                if isinstance(bot_response, dict) and bot_response.get("mode") == "dual_provider":
                    for provider_key in ["azure", "vertex"]:
                        for response in bot_response.get(provider_key, []):
                            chat_data.append({
                                'Provider': provider_key,
                                'Original_Query': user_query,
                                'Search_Query': response.get('Search_Query', ''),
                                'Search_Result_Q': response.get('Search_Result_Q', ''),
                                'Search_Result_A': response.get('Search_Result_A', ''),
                                'Similarity': response.get('Similarity', ''),
                            })

        if chat_data:
            df = pd.DataFrame(chat_data)
            output_dir = PROJECT_ROOT / "data" / "output" / "latest"
            output_dir.mkdir(parents=True, exist_ok=True)
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"eval_chat_history_{timestamp}.xlsx"
            df.to_excel(str(output_path), index=False)
            st.sidebar.success(f"チャット履歴を保存しました: {output_path.name}")
        else:
            st.sidebar.warning("保存するチャット履歴がありません。")

    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)
        st.sidebar.error("チャット履歴の保存中にエラーが発生しました。")


def process_query(query: str):
    st.session_state.processing_query = True
    try:
        query_number = len(st.session_state.chat_history) // 2 + 1
        app_mode = st.session_state.get("app_mode", "evaluation")
        revision = st.session_state.selected_revision

        if app_mode == "impact_analysis":
            logger.info(f"=== 影響調査クエリ {query_number} ===")
        else:
            search_type = REVISION_SEARCH_TYPES.get(revision, "hybrid")
            logger.info(f"=== 評価クエリ {query_number}: 改定番号={revision}, 検索タイプ={search_type} ===")

        azure_results, vertex_results, llm_query = execute_dual_provider_search(query, revision)

        st.session_state.azure_results = azure_results
        st.session_state.vertex_results = vertex_results

        logger.info(f"Azure検索結果数: {len(azure_results)}件, VertexAI検索結果数: {len(vertex_results)}件")

        st.session_state.chat_history.append({
            "type": "bot",
            "text": {
                "mode": "dual_provider",
                "azure": azure_results,
                "vertex": vertex_results,
                "llm_query": llm_query,
                "providers": st.session_state.get("selected_providers", "both"),
            }
        })

    except Exception as e:
        escaped_error = html.escape(str(e))
        error_message = f"エラーが発生しました: {escaped_error}"
        st.error(error_message)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({"type": "bot", "text": error_message})
    finally:
        st.session_state.processing_query = False


def _render_provider_results(results: List[Dict], correct_ids: List[str], is_vertex: bool = False) -> None:
    """プロバイダー検索結果を表示"""
    if not results:
        if is_vertex:
            app_mode = st.session_state.get("app_mode", "evaluation")
            search_type = getattr(st.session_state.config, "search_type", "hybrid")
            if app_mode == "impact_analysis" and search_type == "keyword_filter":
                st.info("キーワード検索のためスキップ（Azureタブの結果をご確認ください）")
            elif app_mode == "evaluation" and search_type == "keyword_filter":
                st.info("キーワード検索のためスキップ（Azureタブの結果をご確認ください）")
            else:
                st.info("該当する結果がありません")
        else:
            st.info("該当する結果がありません")
        return

    correct_count = sum(
        1 for r in results
        if check_if_correct(r, correct_ids, r.get("_area"))[1]
    )
    st.caption(f"検索結果: {len(results)}件（正解: {correct_count}件）")

    with st.container(height=600):
        for idx, response in enumerate(results, 1):
            area = response.get("_area", "")
            scenario_id, is_correct = check_if_correct(response, correct_ids, area)
            card_html = format_response_card(
                idx, response["Similarity"],
                response["Search_Result_Q"], response["Search_Result_A"],
                category=response.get("Search_Category"),
                scenario_id=scenario_id, is_correct=is_correct
            )
            st.markdown(card_html, unsafe_allow_html=True)


def run_streamlit_ui():
    st.set_page_config(page_title="事務改定評価", layout="wide", initial_sidebar_state="expanded")
    apply_common_styles()
    initialize_session_state()

    with st.sidebar:
        st.title("事務改定 AI")

        # モード選択（最上部）
        app_mode = st.radio(
            "モード",
            options=["evaluation", "impact_analysis"],
            format_func=lambda x: {"evaluation": "評価モード", "impact_analysis": "影響調査モード"}[x],
            key="app_mode_radio",
            horizontal=True,
        )
        st.session_state.app_mode = app_mode

        st.markdown("---")

        if app_mode == "evaluation":
            # === 評価モード ===
            revision_data = load_revision_correct_ids()
            revision_options = list(revision_data.keys())

            if not revision_options:
                st.warning("正解IDデータが見つかりません")
                st.session_state.selected_revision = None
                st.session_state.correct_ids = []
            else:
                current_revision_idx = 0
                if st.session_state.selected_revision in revision_options:
                    current_revision_idx = revision_options.index(st.session_state.selected_revision)

                selected_revision = st.selectbox(
                    "改定番号",
                    revision_options,
                    index=current_revision_idx,
                    key="revision_select",
                    help="改定番号を選択すると、Azure/VertexAI両方で検索し、正解IDとマッチした結果にバッジを表示します"
                )

                st.session_state.selected_revision = selected_revision

                if selected_revision in revision_data:
                    content, correct_ids = revision_data[selected_revision]
                    st.session_state.correct_ids = correct_ids
                    st.success(f"正解ID: {len(correct_ids)}件")

                    areas = REVISION_TO_AREAS.get(selected_revision, [])
                    if areas:
                        st.caption(f"対象エリア: {', '.join(areas)}")

            st.markdown("---")
            st.subheader("検索設定")

            default_search_type = REVISION_SEARCH_TYPES.get(
                st.session_state.get("selected_revision", ""), "hybrid"
            )
            eval_search_type_labels = {
                "hybrid": "意味検索",
                "keyword_filter": "キーワード検索",
            }
            eval_selected_search_type = st.radio(
                "検索タイプ",
                options=["hybrid", "keyword_filter"],
                format_func=lambda x: eval_search_type_labels[x],
                index=0 if default_search_type == "hybrid" else 1,
                key="eval_search_type_radio",
                horizontal=True,
            )
            st.session_state.config.search_type = eval_selected_search_type

            if eval_selected_search_type == "hybrid":
                default_vector_weight = REVISION_VECTOR_WEIGHTS.get(
                    st.session_state.get("selected_revision", ""), DEFAULT_VECTOR_WEIGHT
                )
                weight = render_vector_weight_slider(default_vector_weight, key="eval_vector_weight")
                st.session_state.config.vector_weight = weight
                st.session_state.config.keyword_weight = 1.0 - weight

                st.markdown("---")
                eval_top_k = st.number_input(
                    "候補数",
                    min_value=10,
                    max_value=200,
                    value=max(10, st.session_state.config.top_k),
                    step=10,
                    key="eval_top_k",
                    help="検索結果の最大件数（評価用に多めに設定）"
                )
                st.session_state.config.top_k = eval_top_k
            else:
                st.caption("キーワード検索: マッチする全件を返却します")

            st.markdown("---")
            eval_provider_options = {
                "both": "両方",
                "azure_openai": "Azure",
                "vertex_ai": "VertexAI",
            }
            eval_providers = st.radio(
                "検索プロバイダー",
                options=list(eval_provider_options.keys()),
                format_func=lambda x: eval_provider_options[x],
                key="eval_provider_radio",
                horizontal=True,
            )
            st.session_state.selected_providers = eval_providers

        else:
            # === 影響調査モード ===
            st.session_state.correct_ids = []  # 正解判定なし

            impact_category_options = {
                "naibujimu": "内部事務",
                "smile": "スマイル",
            }
            impact_category = st.radio(
                "対象カテゴリ",
                options=list(impact_category_options.keys()),
                format_func=lambda x: impact_category_options[x],
                key="impact_category_radio",
                horizontal=True,
            )
            st.session_state.impact_categories = [impact_category]

            st.markdown("---")
            source_options = {
                "scenario": "シナリオ",
                "history_data": "FAQ",
            }
            source_selection = st.radio(
                "データソース",
                options=list(source_options.keys()),
                format_func=lambda x: source_options[x],
                key="impact_source_radio",
                horizontal=True,
            )
            st.session_state.impact_source_filter = source_selection

            st.markdown("---")
            st.subheader("検索設定")

            impact_search_type_labels = {
                "hybrid": "意味検索",
                "keyword_filter": "キーワード検索",
            }
            impact_search_type = st.radio(
                "検索タイプ",
                options=["hybrid", "keyword_filter"],
                format_func=lambda x: impact_search_type_labels[x],
                key="impact_search_type_radio",
                horizontal=True,
            )
            st.session_state.config.search_type = impact_search_type

            if impact_search_type == "hybrid":
                weight = render_vector_weight_slider(DEFAULT_VECTOR_WEIGHT, key="impact_vector_weight")
                st.session_state.config.vector_weight = weight
                st.session_state.config.keyword_weight = 1.0 - weight

                impact_top_k = st.number_input(
                    "候補数",
                    min_value=10,
                    max_value=200,
                    value=max(10, st.session_state.config.top_k),
                    step=10,
                    key="impact_top_k",
                )
                st.session_state.config.top_k = impact_top_k
            else:
                st.caption("キーワード検索: マッチする全件を返却します")

            st.markdown("---")
            impact_provider_options = {
                "both": "両方",
                "azure_openai": "Azure",
                "vertex_ai": "VertexAI",
            }
            impact_providers = st.radio(
                "検索プロバイダー",
                options=list(impact_provider_options.keys()),
                format_func=lambda x: impact_provider_options[x],
                key="impact_provider_radio",
                horizontal=True,
            )
            st.session_state.selected_providers = impact_providers

        st.markdown("---")
        if st.button("チャット履歴を保存", use_container_width=True, key="save_chat_history_button"):
            save_chat_history()

        if st.button("キャッシュクリア", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    # メインエリア タイトル
    if st.session_state.get("app_mode") == "impact_analysis":
        cats = st.session_state.get("impact_categories", [])
        cat_label = " + ".join(AREA_TO_CATEGORY.get(c, c) for c in cats)
        st.title(f"影響調査【{cat_label}】")
    elif st.session_state.selected_revision:
        st.title(f"事務改定評価【改定{st.session_state.selected_revision}】")
    else:
        st.title("事務改定評価")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                st.markdown(format_message(msg["text"], True), unsafe_allow_html=True)
            else:
                if isinstance(msg["text"], dict) and msg["text"].get("mode") == "dual_provider":
                    azure_results = msg["text"].get("azure", [])
                    vertex_results = msg["text"].get("vertex", [])
                    llm_query = msg["text"].get("llm_query", "")
                    providers = msg["text"].get("providers", "both")
                    correct_ids = st.session_state.correct_ids

                    if llm_query:
                        st.markdown(f"<div style='background-color: #f0f7ff; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9em;'><strong>LLM強化クエリ:</strong> {html.escape(llm_query)}</div>", unsafe_allow_html=True)

                    if providers == "both":
                        tab_azure, tab_vertex = st.tabs(["Azure", "VertexAI"])
                        with tab_azure:
                            _render_provider_results(azure_results, correct_ids)
                        with tab_vertex:
                            _render_provider_results(vertex_results, correct_ids, is_vertex=True)
                    elif providers == "azure_openai":
                        _render_provider_results(azure_results, correct_ids)
                    else:
                        _render_provider_results(vertex_results, correct_ids)
                else:
                    st.markdown(format_message(str(msg["text"]), False), unsafe_allow_html=True)

        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("検索クエリを入力", key="query", label_visibility="collapsed", placeholder="改定影響を検索...")
        submit_button = st.form_submit_button("検索", use_container_width=True, disabled=st.session_state.processing_query)

    if submit_button and query.strip():
        app_mode = st.session_state.get("app_mode", "evaluation")
        if app_mode == "evaluation" and not st.session_state.selected_revision:
            st.warning("改定番号を選択してください。")
        else:
            st.session_state.chat_history.append({"type": "user", "text": query})
            process_query(query.strip())
            st.rerun()

    if st.session_state.processing_query:
        st.markdown('<div class="processing-indicator">処理中...</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    run_streamlit_ui()
