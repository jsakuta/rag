"""回答支援AI（類似回答検索）— Streamlit UI"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import os
import re
import html
import time
from typing import Dict, List

from config import SearchConfig, load_settings
from src.core.processor import Processor
from src.utils.logger import setup_logger, suppress_noise, print_startup_summary, print_query_panel
from src.utils.dynamic_db_manager import DynamicDBManager

# 共通UI部品
from ui.shared import (
    format_message, format_response_card, render_vector_weight_slider,
    apply_common_styles, initialize_common_session_state,
)

logger = setup_logger(__name__)
_ui_settings = load_settings("ui")

# デフォルトの業務分野リスト（DBが存在しない場合のフォールバック）
DEFAULT_BUSINESS_AREAS = ["預金", "融資", "外貨", "投信", "住宅ローン", "カード", "保険", "年金", "総則"]


@st.cache_data(ttl=60)
def get_available_business_areas() -> list:
    """利用可能な業務分野を動的に取得"""
    try:
        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        db_manager = DynamicDBManager(config)
        areas = db_manager.get_all_business_areas(include_revisions=False)
        if areas:
            return areas
    except Exception as e:
        logger.warning(f"業務分野一覧の取得に失敗: {e}")
    return DEFAULT_BUSINESS_AREAS


def initialize_session_state():
    initialize_common_session_state()
    if "business_area" not in st.session_state:
        st.session_state.business_area = "預金"


def _needs_processor_reinit() -> bool:
    """Processorの再初期化が必要かどうかを判定"""
    if "processor" not in st.session_state:
        return True

    config = st.session_state.config
    return (
        st.session_state.get("last_business_area") != st.session_state.business_area
        or st.session_state.get("last_search_type") != config.search_type
        or st.session_state.get("last_search_mode") != config.search_mode
        or st.session_state.get("last_search_source") != config.search_source
    )


def _load_reference_data_for_business(config, business_area: str) -> dict:
    """業務分野に応じた参照データを読み込む"""
    try:
        with DynamicDBManager(config) as db_manager:
            business_areas = db_manager.analyze_reference_files(include_revisions=False)

            if business_area not in business_areas:
                raise ValueError(f"業務分野 '{business_area}' の参照ファイルが見つかりません（検出済み: {list(business_areas.keys())}）")

            area_data = business_areas[business_area]
            faq_list = area_data.get("faq", [])
            scenario_list = area_data.get("scenario", [])

            all_queries, all_answers, all_combined_texts, all_metadatas = [], [], [], []

            if scenario_list:
                scenario_file = db_manager.get_latest_file(scenario_list)
                if scenario_file:
                    scenario_path = os.path.join(db_manager.reference_scenario_path, scenario_file)
                    from src.handlers.input_handler import HierarchicalExcelInputHandler
                    handler = HierarchicalExcelInputHandler(config, scenario_path)
                    data = handler.load_reference_data()
                    all_queries.extend(data['queries'])
                    all_answers.extend(data['answers'])
                    all_combined_texts.extend(data['combined_texts'])
                    all_metadatas.extend(data['metadatas'])

            if faq_list:
                faq_file = db_manager.get_latest_file(faq_list)
                if faq_file:
                    faq_path = os.path.join(db_manager.reference_faq_path, faq_file)
                    import copy
                    faq_config = copy.copy(config)
                    faq_config.REFERENCE_FILE_PATTERN = re.escape(faq_file) + "$"
                    from src.handlers.input_handler import ExcelInputHandler
                    handler = ExcelInputHandler(faq_config)
                    handler.reference_dir = db_manager.reference_faq_path
                    data = handler.load_reference_data()
                    all_queries.extend(data['queries'])
                    all_answers.extend(data['answers'])
                    all_combined_texts.extend(data['combined_texts'])
                    all_metadatas.extend(data['metadatas'])

            if not all_queries:
                raise ValueError(f"業務分野 '{business_area}' の参照データが空です")

            logger.info(f"業務分野 '{business_area}' の参照データ読み込み完了: {len(all_queries)}件")
            return {
                'queries': all_queries,
                'answers': all_answers,
                'combined_texts': all_combined_texts,
                'metadatas': all_metadatas,
            }
    except Exception as e:
        import traceback
        logger.error(f"参照データの読み込みエラー ({business_area}): {e}")
        logger.error(traceback.format_exc())
        raise


def _initialize_processor():
    """Processorを初期化してセッションステートを更新"""
    st.session_state.processor = Processor(st.session_state.config)

    business_area = st.session_state.business_area
    reference_data = _load_reference_data_for_business(st.session_state.config, business_area)

    st.session_state.processor.searcher.prepare_search(reference_data)
    st.session_state.processor.searcher._select_db_for_business(business_area)
    st.session_state.last_business_area = business_area
    st.session_state.last_search_type = st.session_state.config.search_type
    st.session_state.last_search_mode = st.session_state.config.search_mode
    st.session_state.last_search_source = st.session_state.config.search_source


def process_query(query: str):
    st.session_state.processing_query = True
    try:
        query_number = len(st.session_state.chat_history) // 2 + 1
        start_time = time.time()

        if _needs_processor_reinit():
            _initialize_processor()

        processor = st.session_state.processor

        results = processor.searcher.search(str(query_number), query, "")

        elapsed = time.time() - start_time
        result_count = len(results) if results else 0
        max_sim = max((r.get('Similarity', 0) for r in results), default=0) if results else 0
        meta = {"業務": st.session_state.business_area, "モード": st.session_state.config.search_mode}
        if results:
            meta["最高類似度"] = f"{max_sim:.2f}"
        print_query_panel(
            query_number=query_number,
            query_text=query,
            metadata=meta,
            results={"結果": result_count},
            elapsed=elapsed,
        )

        if results:
            st.session_state.last_query = query
            st.session_state.last_results = results
            st.session_state.chat_history.append({"type": "bot", "text": results})
        else:
            st.session_state.last_query = None
            st.session_state.last_results = None
            st.session_state.chat_history.append({"type": "bot", "text": "該当する結果が見つかりませんでした。"})

    except Exception as e:
        escaped_error = html.escape(str(e))
        error_message = f"エラーが発生しました: {escaped_error}"
        st.error(error_message)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({"type": "bot", "text": error_message})
    finally:
        st.session_state.processing_query = False


def save_chat_history():
    """チャット履歴を保存"""
    try:
        chat_data = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_query = st.session_state.chat_history[i]["text"]
                bot_response = st.session_state.chat_history[i + 1]["text"]

                if isinstance(bot_response, list):
                    for response in bot_response:
                        chat_data.append({
                            'Input_Number': response.get('Input_Number', ''),
                            'Original_Query': user_query,
                            'Search_Query': response.get('Search_Query', ''),
                            'Search_Result_Q': response.get('Search_Result_Q', ''),
                            'Search_Result_A': response.get('Search_Result_A', ''),
                            'Similarity': response.get('Similarity', ''),
                            'Vector_Weight': response.get('Vector_Weight', ''),
                            'Top_K': response.get('Top_K', '')
                        })

        if chat_data:
            processor = Processor(st.session_state.config)
            output_path = processor.output_handler.save_data(chat_data, mode="chat")
            if output_path:
                st.sidebar.success(f"チャット履歴を保存しました: {Path(output_path).name}")
        else:
            st.sidebar.warning("保存するチャット履歴がありません。")

    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)
        st.sidebar.error("チャット履歴の保存中にエラーが発生しました。")


def run_streamlit_ui():
    suppress_noise()
    st.set_page_config(page_title="類似回答検索ボット", layout="wide", initial_sidebar_state="expanded")
    apply_common_styles()
    initialize_session_state()

    if "startup_logged" not in st.session_state:
        areas = get_available_business_areas()
        checks = [
            ("DB接続", len(areas) > 0, f"({', '.join(areas)})"),
            ("キーワードキャッシュ", True, "OK"),
        ]
        print_startup_summary("回答支援AI（類似回答検索）v1.0", checks)
        logger.info("⚡ Ready")
        st.session_state.startup_logged = True

    with st.sidebar:
        st.title("設定")

        st.markdown("---")
        with st.expander("検索パラメータ", expanded=True):
            st.session_state.config.search_type = "hybrid"

            weight = render_vector_weight_slider(st.session_state.config.vector_weight)
            st.session_state.config.vector_weight = weight
            st.session_state.config.keyword_weight = 1.0 - weight

            search_modes = ["original", "llm_enhanced"]
            mode_labels = {"original": "原文検索", "llm_enhanced": "LLMクエリ検索"}
            current_mode_index = search_modes.index(st.session_state.config.search_mode) if st.session_state.config.search_mode in search_modes else 0
            selected_mode = st.selectbox(
                "検索モード",
                search_modes,
                format_func=lambda x: mode_labels[x],
                index=current_mode_index
            )
            st.session_state.config.search_mode = selected_mode

            search_sources = ["scenario", "history_data"]
            source_labels = {"scenario": "シナリオのみ", "history_data": "FAQのみ"}
            current_source_index = search_sources.index(st.session_state.config.search_source) if st.session_state.config.search_source in search_sources else 0
            selected_source = st.selectbox(
                "検索対象",
                search_sources,
                format_func=lambda x: source_labels[x],
                index=current_source_index
            )
            st.session_state.config.search_source = selected_source

            st.session_state.config.top_k = st.number_input(
                "表示する候補数", min_value=1, max_value=10,
                value=min(10, st.session_state.config.top_k), step=1
            )

        st.markdown("---")
        st.subheader("業務分野")
        business_areas = get_available_business_areas()
        current_area = st.session_state.business_area
        if current_area not in business_areas:
            current_area = business_areas[0] if business_areas else "預金"
            st.session_state.business_area = current_area
        st.session_state.business_area = st.selectbox(
            "業務分野",
            business_areas,
            index=business_areas.index(current_area),
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button("チャット履歴を保存", use_container_width=True, key="save_chat_history_button"):
            save_chat_history()

    st.title(f"類似回答検索ボット【{st.session_state.business_area}】")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                st.markdown(format_message(msg["text"], True), unsafe_allow_html=True)
            else:
                if isinstance(msg["text"], list):
                    for idx, response in enumerate(msg["text"], 1):
                        card_html = format_response_card(
                            idx, response["Similarity"],
                            response["Search_Result_Q"], response["Search_Result_A"],
                            category=response.get("Search_Category"),
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.markdown(format_message(msg["text"], False), unsafe_allow_html=True)

        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("質問を入力してください", key="query", label_visibility="collapsed", placeholder="質問を入力...")
        submit_button = st.form_submit_button("送信", use_container_width=True, disabled=st.session_state.processing_query)

    if submit_button and query.strip():
        st.session_state.chat_history.append({"type": "user", "text": query})
        process_query(query.strip())
        st.rerun()

    if st.session_state.processing_query:
        st.markdown('<div class="processing-indicator">処理中...</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    run_streamlit_ui()
