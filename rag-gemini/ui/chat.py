# --- chat.py (旧 ui.py) ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # .envファイルから環境変数を読み込み

from config import SearchConfig
from src.core.processor import Processor
from src.utils.logger import setup_logger
import datetime
import os
import html

logger = setup_logger(__name__)

def initialize_session_state():
    """セッションステートの初期化"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    if "config" not in st.session_state:
        st.session_state.config = SearchConfig(
            search_mode="original",  # UIで切り替え可能
            top_k=3,
            llm_provider="gemini",
            llm_model="gemini-2.0-flash-001",
            vector_weight=0.7,
            embedding_provider="vertex_ai",
            embedding_model="gemini-embedding-001",
            base_dir="."
        )
    if "business_area" not in st.session_state:
        st.session_state.business_area = "預金"

def format_message(message, is_user=False):
    escaped_message = html.escape(str(message))
    style = f"""
        <div style="display: flex; justify-content: {'flex-end' if is_user else 'flex-start'}; margin: 5px 0;">
            <div style="background-color: {'#e6f3ff' if is_user else '#f5f5f5'}; padding: 10px 15px;
                border-radius: 15px; max-width: 80%; {'margin-left: auto;' if is_user else ''}">
                {escaped_message}
            </div>
        </div>
    """
    return style

def format_response_card(number, similarity, query, answer, category=None):
    # XSS対策: ユーザー入力をエスケープ
    query = html.escape(str(query))
    answer = html.escape(str(answer))

    # カテゴリバッジの色設定
    category_colors = {
        'Both': '#E2EFDA',
        'Original_Only': '#FFF2CC',
        'LLM_Enhanced_Only': '#DEEBF7'
    }
    category_labels = {
        'Both': '両方',
        'Original_Only': '原文のみ',
        'LLM_Enhanced_Only': 'LLMのみ'
    }
    category_badge = ""
    if category:
        bg_color = category_colors.get(category, '#f0f0f0')
        label = category_labels.get(category, category)
        category_badge = f" <span style='background-color: {bg_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;'>{label}</span>"

    return f"""
        <div class="response-card" style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px;
            margin: 10px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #666; margin-bottom: 10px; font-size: 0.95em; padding-bottom: 8px;
                border-bottom: 1px solid #eee;">候補 {number} (類似度: {similarity:.4f}){category_badge}
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
                <div style="font-weight: 600; margin-bottom: 5px;">類似質問内容:</div>
                <div style="white-space: pre-wrap;">{query}</div>
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
                <div style="font-weight: 600; margin-bottom: 5px;">回答:</div>
                <div style="white-space: pre-wrap;">{answer}</div>
            </div>
        </div>
    """

def process_query(query: str):
    st.session_state.processing_query = True
    try:
        # Processorをキャッシュ（業務分野またはsearch_mode変更時は再初期化）
        needs_reinit = (
            "processor" not in st.session_state
            or st.session_state.get("last_business_area") != st.session_state.business_area
            or st.session_state.get("last_search_mode") != st.session_state.config.search_mode
        )
        if needs_reinit:
            st.session_state.processor = Processor(st.session_state.config)
            reference_data = st.session_state.processor.reference_handler.load_reference_data()
            st.session_state.processor.searcher.prepare_search(reference_data)
            st.session_state.processor.searcher._select_db_for_business(st.session_state.business_area)
            st.session_state.last_business_area = st.session_state.business_area
            st.session_state.last_search_mode = st.session_state.config.search_mode

        processor = st.session_state.processor
        query_number = len(st.session_state.chat_history) // 2 + 1
        results = processor.searcher.search(str(query_number), query, "")

        if results:
            st.session_state.chat_history.append({"type": "bot", "text": results})
        else:
            st.session_state.chat_history.append({"type": "bot", "text": "該当する結果が見つかりませんでした。"})

    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
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
                            'Summarized_Query': response.get('Summarized_Query', ''),
                            'Search_Result_Q': response.get('Search_Result_Q', ''),
                            'Search_Result_A': response.get('Search_Result_A', ''),
                            'Similarity': response.get('Similarity', ''),
                            'Vector_Weight': response.get('Vector_Weight', ''),
                            'Top_K': response.get('Top_K', '')
                        })

        if chat_data:
            processor = Processor(st.session_state.config)
            # OutputHandlerを使用してチャット履歴を保存
            processor.output_handler.save_data(chat_data, mode="chat")  # modeを"chat"に設定
            st.sidebar.success("チャット履歴を保存しました。")
        else:
            st.sidebar.warning("保存するチャット履歴がありません。")

    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)
        st.sidebar.error("チャット履歴の保存中にエラーが発生しました。")

def run_streamlit_ui():
    st.set_page_config(page_title="類似回答検索ボット【預金】", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        div.stButton > button:hover { background-color: #007bff; color: white; }
        div.stButton > button:focus { background-color: #007bff; color: white; }
        [data-baseweb="button"]:has(#save_chat_history_button) { background-color: #28a745 !important; color: white !important; }
        [data-baseweb="button"]:has(#save_chat_history_button):hover { background-color: #1e7e34 !important; }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    with st.sidebar:
        st.title("設定")
        with st.expander("パラメータ調整", expanded=True):
            # 検索モード選択
            search_modes = ["original", "llm_enhanced", "multi_stage"]
            mode_labels = {"original": "原文検索", "llm_enhanced": "LLMクエリ検索", "multi_stage": "多段階OR検索"}
            current_mode_index = search_modes.index(st.session_state.config.search_mode) if st.session_state.config.search_mode in search_modes else 0
            selected_mode = st.selectbox(
                "検索モード",
                search_modes,
                format_func=lambda x: mode_labels[x],
                index=current_mode_index
            )
            st.session_state.config.search_mode = selected_mode

            # 多段階検索パラメータ（multi_stage時のみ表示）
            if selected_mode == "multi_stage":
                st.session_state.config.multi_stage_threshold = st.slider(
                    "しきい値", 0.0, 1.0,
                    st.session_state.config.multi_stage_threshold, 0.05
                )
                st.session_state.config.multi_stage_enable_judgment_support = st.checkbox(
                    "LLM判断支援",
                    value=st.session_state.config.multi_stage_enable_judgment_support
                )

            # 業務分野選択
            business_areas = ["預金", "融資", "外貨", "投信", "住宅ローン", "カード", "保険", "年金", "総則"]
            st.session_state.business_area = st.selectbox(
                "業務分野",
                business_areas,
                index=business_areas.index(st.session_state.business_area)
            )
            st.session_state.config.vector_weight = st.slider("ベクトルの重み", 0.0, 1.0, st.session_state.config.vector_weight, 0.1)
            st.session_state.config.top_k = st.number_input("表示する候補数", min_value=1, max_value=10, value=st.session_state.config.top_k, step=1)
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
                        category = response.get("Search_Category")  # 多段階検索時のみ存在
                        html = format_response_card(idx, response["Similarity"], response["Search_Result_Q"], response["Search_Result_A"], category)
                        st.markdown(html, unsafe_allow_html=True)
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