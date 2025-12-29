import streamlit as st
import logging
import pandas as pd
from config import SearchConfig
from src.core.processor import ExcelVectorProcessor
from src.utils.utils import setup_logger
import datetime
import os
import re

logger = setup_logger(__name__)

def initialize_session_state():
    """セッションステートの初期化"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    if "config" not in st.session_state:
        st.session_state.config = SearchConfig(
            top_k=3,
            model_name="intfloat/multilingual-e5-base",
            llm_provider="anthropic",
            llm_model="claude-3-sonnet-20240229",
            vector_weight=0.7,
            base_dir="."
        )

def format_message(message, is_user=False):
    """メッセージのスタイルを定義"""
    style = f"""
        <div style="
            display: flex;
            justify-content: {'flex-end' if is_user else 'flex-start'};
            margin: 5px 0;
        ">
            <div style="
                background-color: {'#e6f3ff' if is_user else '#f5f5f5'};
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 80%;
                {'margin-left: auto;' if is_user else ''}
            ">
                {message}
            </div>
        </div>
    """
    return style

def format_response_card(number, similarity, query, answer):
    """応答カードのスタイルを定義"""
    return f"""
        <div class="response-card" style="
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                color: #666;
                margin-bottom: 10px;
                font-size: 0.95em;
                padding-bottom: 8px;
                border-bottom: 1px solid #eee;
            ">
                候補 {number} (類似度: {similarity:.4f})
            </div>
            <div style="
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
            ">
                <div style="font-weight: 600; margin-bottom: 5px;">類似質問内容:</div>
                <div style="white-space: pre-wrap;">{query}</div>
            </div>
            <div style="
                background-color: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin: 8px 0;
            ">
                <div style="font-weight: 600; margin-bottom: 5px;">回答:</div>
                <div style="white-space: pre-wrap;">{answer}</div>
            </div>
        </div>
    """

def process_query(query: str):
    """クエリ処理を実行"""
    st.session_state.processing_query = True
    
    try:
        processor = ExcelVectorProcessor(st.session_state.config)
        _, reference_file = processor.get_latest_files()
        reference_df = pd.read_excel(reference_file)
        processor.reference_df = reference_df  # ここで reference_df を保持
        reference_texts = reference_df['問合せ内容'].fillna('').astype(str).tolist()
        
        cached_data = processor._load_cached_vectors(reference_file)
        if cached_data:
            reference_vectors, cached_texts = cached_data
            if cached_texts != reference_texts:
                reference_vectors = processor.model.encode(reference_texts, normalize_embeddings=True)
                processor._cache_vectors(reference_vectors, reference_texts, reference_file)
        else:
            reference_vectors = processor.model.encode(reference_texts, normalize_embeddings=True)
            processor._cache_vectors(reference_vectors, reference_texts, reference_file)

        query_number = len(st.session_state.chat_history) // 2

        # _process_queryの呼び出しを修正
        results = processor._process_query(
            input_number=str(query_number),
            query_text=query,
            original_answer='',  # UIモードでは原回答は空文字列
            reference_texts=reference_texts,
            reference_vectors=reference_vectors
        )
        
        if results:
            # 検索結果をリストとして保存
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
    """チャット履歴をExcelファイルに保存"""
    try:
        result_rows = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                query = st.session_state.chat_history[i]["text"]
                query_number = i//2 + 1
                bot_response = st.session_state.chat_history[i + 1]["text"]
                
                if isinstance(bot_response, list):
                    first_row = True
                    for response in bot_response:
                        row = {
                            'Input_Number': query_number if first_row else '',
                            'Original_Query': query if first_row else '',
                            'Summarized_Query': response["Summarized_Query"] if first_row else '',
                            'Search_Result_Q': response["Search_Result_Q"],
                            'Search_Result_A': response["Search_Result_A"],
                            'Similarity': f"{response['Similarity']:.4f}",
                            'Vector_Weight': response['Vector_Weight'],
                            'Top_K': response['Top_K']
                        }
                        result_rows.append(row)
                        first_row = False

        result_columns = [
            'Input_Number', 'Original_Query', 'Summarized_Query', 
            'Search_Result_Q', 'Search_Result_A', 'Similarity',
            'Vector_Weight', 'Top_K'
        ]
        result_df = pd.DataFrame(result_rows, columns=result_columns)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        param_summary = f"w{st.session_state.config.vector_weight:.1f}_k{st.session_state.config.top_k}"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, 
            f"chat_log_result_{param_summary}_{timestamp}.xlsx"
        )
        
        with pd.ExcelWriter(output_file, engine='xlsxwriter', 
                           engine_kwargs={'options': {'nan_inf_to_errors': True}}) as writer:
            result_df.to_excel(writer, index=False, sheet_name='Sheet1')
            ExcelVectorProcessor(st.session_state.config)._format_excel(writer, result_df)
            
        st.sidebar.success(f"チャット履歴を保存しました: {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)
        st.sidebar.error("チャット履歴の保存中にエラーが発生しました。")


def run_streamlit_ui():
    """Streamlit UIを起動する"""
    st.set_page_config(
        page_title="質問応答チャットボット",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # テーマ設定をカスタマイズ
    st.markdown(
        """
        <style>
        /* ボタンのホバー時の色を青色に変更 */
        div.stButton > button:hover {
            background-color: #007bff;
            color: white;
        }
        div.stButton > button:focus {
            background-color: #007bff;
            color: white;
        }
        /* チャット履歴保存ボタンの色を変更 */
        [data-baseweb="button"]:has(#save_chat_history_button) {
            background-color: #28a745 !important;
            color: white !important;
         }
         [data-baseweb="button"]:has(#save_chat_history_button):hover {
            background-color: #1e7e34 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    
    initialize_session_state()
    
    # サイドバー設定
    with st.sidebar:
        st.title("設定")
        with st.expander("パラメータ調整", expanded=True):
            st.session_state.config.vector_weight = st.slider(
                "ベクトルの重み",
                0.0, 1.0,
                st.session_state.config.vector_weight,
                0.1
            )
            st.session_state.config.top_k = st.number_input(
                "表示する候補数",
                min_value=1,
                max_value=10,
                value=st.session_state.config.top_k,
                step=1
            )

        if st.button("チャット履歴を保存", use_container_width=True,key = "save_chat_history_button"): # keyを追加
            save_chat_history()

    # メインチャットエリア
    st.title("質問応答チャットボット")
    
    # チャット履歴の表示
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                st.markdown(
                    format_message(msg["text"], True),
                    unsafe_allow_html=True
                )
            else:
                if isinstance(msg["text"], list):  # 検索結果の場合
                    for idx, response in enumerate(msg["text"], 1):  # enumerate with start=1
                        html = format_response_card(
                            idx,  # Input_Number の代わりに idx を使用
                            response["Similarity"],
                            response["Search_Result_Q"],  # Original_Query から Search_Result_Q に変更
                            response["Search_Result_A"]
                        )
                        st.markdown(html, unsafe_allow_html=True)
                else:  # 通常のメッセージまたはエラーメッセージの場合
                    st.markdown(
                        format_message(msg["text"], False),
                        unsafe_allow_html=True
                    )

        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    # 入力フォーム
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input(
            "質問を入力してください",
            key="query",
            label_visibility="collapsed",
            placeholder="質問を入力..."
        )
        
        submit_button = st.form_submit_button(
            "送信",
            use_container_width=True,
            disabled=st.session_state.processing_query
        )

    # フォーム送信時の処理
    if submit_button and query.strip():
        st.session_state.chat_history.append({"type": "user", "text": query})
        process_query(query.strip())
        st.rerun()

    # 処理中の表示
    if st.session_state.processing_query:
        st.markdown('<div class="processing-indicator">処理中...</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    run_streamlit_ui()