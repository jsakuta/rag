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
            vector_weight=0.9,
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

def format_response_card(number, similarity, query, answer, category=None,
                         relevance_judgment=None, judgment_reason=None, modification_suggestion=None):
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

    # 関連性判定バッジの色設定
    relevance_colors = {
        '関連あり': '#c8e6c9',
        '要確認': '#fff9c4',
        '関連なし': '#ffcdd2',
        '判断支援無効': '#e0e0e0',
        'エラー': '#ffcdd2'
    }
    relevance_badge = ""
    if relevance_judgment:
        # 関連性判定の最初の単語を取得（「関連あり」「要確認」「関連なし」など）
        judgment_key = relevance_judgment.split()[0] if relevance_judgment else ""
        bg_color = relevance_colors.get(judgment_key, '#e0e0e0')
        escaped_judgment = html.escape(str(relevance_judgment))
        relevance_badge = f" <span style='background-color: {bg_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;'>{escaped_judgment}</span>"

    # LLM分析セクション（関連性判定がある場合のみ表示）
    llm_analysis_section = ""
    show_llm_analysis = relevance_judgment and relevance_judgment not in ['判断支援無効', '']
    if show_llm_analysis:
        reason_text = html.escape(str(judgment_reason)) if judgment_reason else "-"
        suggestion_raw = html.escape(str(modification_suggestion)) if modification_suggestion else ""
        suggestion_text = "なし" if not suggestion_raw or suggestion_raw.strip() in ['-', 'なし'] else suggestion_raw

        llm_analysis_section = f"""<div style="background-color: #f0f7ff; padding: 12px; border-radius: 8px; margin: 8px 0;"><div style="font-weight: 600; margin-bottom: 5px;">LLM分析</div><div>関連性: {html.escape(str(relevance_judgment))}</div><div>根拠: {reason_text}</div><div>修正案: {suggestion_text}</div></div>"""

    return f"""
        <div class="response-card" style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px;
            margin: 10px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #666; margin-bottom: 10px; font-size: 0.95em; padding-bottom: 8px;
                border-bottom: 1px solid #eee;">候補 {number} (類似度: {similarity:.4f}){category_badge}{relevance_badge}
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
                <div style="font-weight: 600; margin-bottom: 5px;">類似質問内容:</div>
                <div style="white-space: pre-wrap;">{query}</div>
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
                <div style="font-weight: 600; margin-bottom: 5px;">回答:</div>
                <div style="white-space: pre-wrap;">{answer}</div>
            </div>
            {llm_analysis_section}
        </div>
    """

def _needs_processor_reinit() -> bool:
    """Processorの再初期化が必要かどうかを判定"""
    if "processor" not in st.session_state:
        return True

    config = st.session_state.config
    return (
        st.session_state.get("last_business_area") != st.session_state.business_area
        or st.session_state.get("last_search_mode") != config.search_mode
        or st.session_state.get("last_judgment_support") != config.multi_stage_enable_judgment_support
    )


def _initialize_processor():
    """Processorを初期化してセッションステートを更新"""
    st.session_state.processor = Processor(st.session_state.config)
    reference_data = st.session_state.processor.reference_handler.load_reference_data()
    st.session_state.processor.searcher.prepare_search(reference_data)
    st.session_state.processor.searcher._select_db_for_business(st.session_state.business_area)
    st.session_state.last_business_area = st.session_state.business_area
    st.session_state.last_search_mode = st.session_state.config.search_mode
    st.session_state.last_judgment_support = st.session_state.config.multi_stage_enable_judgment_support


def process_query(query: str):
    st.session_state.processing_query = True
    try:
        if _needs_processor_reinit():
            _initialize_processor()

        processor = st.session_state.processor
        query_number = len(st.session_state.chat_history) // 2 + 1

        # ログ: 処理開始
        judgment_enabled = st.session_state.config.multi_stage_enable_judgment_support
        logger.info(f"=== 質問 {query_number} の処理開始 ===")
        query_display = f"{query[:80]}..." if len(query) > 80 else query
        logger.info(f"クエリ: {query_display}")
        logger.info(f"検索モード: {st.session_state.config.search_mode} (LLM判断支援: {'有効' if judgment_enabled else '無効'})")

        results = processor.searcher.search(str(query_number), query, "")

        if results:
            logger.info(f"検索結果数: {len(results)}件")
            # 検索結果のログ出力
            for i, result in enumerate(results, 1):
                category = result.get('Search_Category', 'N/A')
                similarity = result.get('Similarity', 0)
                logger.info(f"  結果{i}: 類似度={similarity:.4f}, カテゴリ={category}")

            # 最新の検索結果を保存（LLM分析用）
            st.session_state.last_query = query
            st.session_state.last_results = results

            st.session_state.chat_history.append({"type": "bot", "text": results})
        else:
            logger.info("検索結果: 該当なし")
            st.session_state.last_query = None
            st.session_state.last_results = None
            st.session_state.chat_history.append({"type": "bot", "text": "該当する結果が見つかりませんでした。"})

        logger.info(f"=== 質問 {query_number} の検索完了 ===")

    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        st.error(error_message)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({"type": "bot", "text": error_message})
    finally:
        st.session_state.processing_query = False


def run_llm_analysis():
    """最新の検索結果に対してLLM分析を実行"""
    if "last_results" not in st.session_state or not st.session_state.last_results:
        st.warning("分析対象の検索結果がありません。")
        return

    if "processor" not in st.session_state:
        st.error("Processorが初期化されていません。")
        return

    processor = st.session_state.processor
    if not hasattr(processor, 'judgment_support') or processor.judgment_support is None:
        st.error("JudgmentSupportが初期化されていません。")
        return

    results = st.session_state.last_results
    query = st.session_state.last_query
    judgment_support = processor.judgment_support

    logger.info(f"=== LLM分析開始 ({len(results)}件) ===")

    for i, result in enumerate(results, 1):
        evaluation = judgment_support.evaluate(
            query,
            result.get('Search_Result_Q', ''),
            result.get('Search_Result_A', '')
        )
        result['Relevance_Judgment'] = evaluation['relevance_judgment']
        result['Judgment_Reason'] = evaluation['judgment_reason']
        result['Modification_Suggestion'] = evaluation['modification_suggestion']
        logger.info(f"  結果{i}: → LLM判定: {evaluation['relevance_judgment']}")

    # チャット履歴の最新の結果を更新
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history):
            if msg["type"] == "bot" and isinstance(msg["text"], list):
                msg["text"] = results
                break

    logger.info("=== LLM分析完了 ===")

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

            # 多段階検索ではtop_kは使用しない（しきい値ベースのフィルタリング）
            if selected_mode != "multi_stage":
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
                        # LLM分析結果（多段階検索+LLM判断支援有効時のみ存在）
                        relevance_judgment = response.get("Relevance_Judgment")
                        judgment_reason = response.get("Judgment_Reason")
                        modification_suggestion = response.get("Modification_Suggestion")
                        card_html = format_response_card(
                            idx, response["Similarity"],
                            response["Search_Result_Q"], response["Search_Result_A"],
                            category, relevance_judgment, judgment_reason, modification_suggestion
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.markdown(format_message(msg["text"], False), unsafe_allow_html=True)

        # LLM分析ボタン（多段階検索+LLM判断支援有効+検索結果あり時のみ表示）
        show_llm_button = (
            st.session_state.config.search_mode == "multi_stage"
            and st.session_state.config.multi_stage_enable_judgment_support
            and st.session_state.get("last_results")
            and not any(r.get("Relevance_Judgment") for r in st.session_state.get("last_results", []))
        )
        if show_llm_button:
            if st.button("LLM分析を実行", use_container_width=True, type="primary"):
                with st.spinner("LLM分析を実行中..."):
                    run_llm_analysis()
                st.rerun()

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