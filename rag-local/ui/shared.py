"""共通UI部品 — 回答支援AI（類似回答検索）・運用保守効率化AI（改定影響調査）共用

このモジュールは以下から import される:
- apps/answer-support/ui/chat.py（回答支援UI）
- apps/revision-ops/ui/ops_ui.py（改定影響調査UI）

責務: UIレンダリング部品 + 共通CSS + 共通セッション状態初期化
Processor 等の重い依存は持たない（各UI側で管理する）。
"""
import sys
import os
from pathlib import Path
import html
from typing import Dict, List, Optional, Tuple

import streamlit as st

# rag-local ルートへのパス解決
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SearchConfig, load_settings
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# UI専用設定をYAMLから読み込み
_ui_settings = load_settings("ui")

# 共通CSS
COMMON_STYLES = """
<style>
div.stButton > button:hover { background-color: #007bff; color: white; }
div.stButton > button:focus { background-color: #007bff; color: white; }
[data-baseweb="button"]:has(#save_chat_history_button) { background-color: #28a745 !important; color: white !important; }
[data-baseweb="button"]:has(#save_chat_history_button):hover { background-color: #1e7e34 !important; }
</style>
"""


def apply_common_styles():
    st.markdown(COMMON_STYLES, unsafe_allow_html=True)


def initialize_common_session_state():
    """chat_history, processing_query, config の共通初期化。
    各UIはこの関数を呼んだ後、モード固有の状態を追加する。"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    if "config" not in st.session_state:
        # 必須環境変数のチェック（LLM設定のみ）
        required_env_vars = [
            "DEFAULT_LLM_PROVIDER",
            "DEFAULT_LLM_MODEL",
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"必須環境変数が設定されていません: {', '.join(missing_vars)}")

        # UI専用設定をYAMLから取得
        ui_top_k = _ui_settings.get("top_k", 3)
        ui_vector_weight = _ui_settings.get("vector_weight", 0.9)
        ui_search_mode = _ui_settings.get("search_mode", "original")
        ui_search_type = _ui_settings.get("search_type", "hybrid")

        st.session_state.config = SearchConfig(
            search_type=ui_search_type,
            search_mode=ui_search_mode,
            top_k=ui_top_k,
            llm_provider=os.getenv("DEFAULT_LLM_PROVIDER"),
            llm_model=os.getenv("DEFAULT_LLM_MODEL"),
            vector_weight=ui_vector_weight,
            base_dir=str(PROJECT_ROOT),
        )


def format_message(message: str, is_user: bool = False) -> str:
    """チャットメッセージをHTML形式にフォーマット"""
    escaped_message = html.escape(str(message))
    align = 'flex-end' if is_user else 'flex-start'
    bg_color = '#e6f3ff' if is_user else '#f5f5f5'
    margin = 'margin-left: auto;' if is_user else ''
    return f"""
        <div style="display: flex; justify-content: {align}; margin: 5px 0;">
            <div style="background-color: {bg_color}; padding: 10px 15px;
                border-radius: 15px; max-width: 80%; {margin}">
                {escaped_message}
            </div>
        </div>
    """


def _create_badge(text: str, bg_color: str, text_color: str, bold: bool = True) -> str:
    """共通バッジHTML生成"""
    weight = 'font-weight: bold;' if bold else ''
    return f" <span style='background-color: {bg_color}; color: {text_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; {weight} margin-left: 8px;'>{text}</span>"


def _create_category_badge(category: Optional[str]) -> str:
    """カテゴリバッジを生成"""
    if not category:
        return ""
    styles = {
        'Both': ('#e1bee7', '#6a1b9a', '両方'),
        'Original_Only': ('#FFF2CC', '#856404', '原文のみ'),
        'LLM_Enhanced_Only': ('#DEEBF7', '#1565c0', 'LLMのみ')
    }
    bg_color, text_color, label = styles.get(category, ('#f0f0f0', '#333', category))
    return _create_badge(label, bg_color, text_color)


def _create_correct_badge(is_correct: bool) -> str:
    """正解バッジを生成"""
    if not is_correct:
        return ""
    return _create_badge('★正解', '#fff3cd', '#856404')


def _create_scenario_id_badge(scenario_id: Optional[str]) -> str:
    """シナリオIDバッジを生成"""
    if not scenario_id:
        return ""
    return _create_badge(html.escape(str(scenario_id)), '#e3f2fd', '#1565c0')


def _create_relevance_badge(relevance_judgment: Optional[str]) -> str:
    """関連性判定バッジを生成"""
    if not relevance_judgment:
        return ""
    styles = {
        '関連あり': ('#bbdefb', '#1565c0'),
        '要確認': ('#fff9c4', '#f57f17'),
        '関連なし': ('#ffcdd2', '#c62828'),
        '判断支援無効': ('#e0e0e0', '#616161'),
        'エラー': ('#ffcdd2', '#c62828')
    }
    judgment_key = relevance_judgment.split()[0] if relevance_judgment else ""
    bg_color, text_color = styles.get(judgment_key, ('#e0e0e0', '#333'))
    return _create_badge(html.escape(str(relevance_judgment)), bg_color, text_color)


def _is_valid_llm_judgment(relevance_judgment: Optional[str]) -> bool:
    """LLM判定結果が有効かどうかを判定"""
    if not relevance_judgment:
        return False
    judgment_str = str(relevance_judgment).strip()
    return judgment_str != '' and judgment_str not in ['判断支援無効', 'None']


def _create_llm_analysis_section(
    relevance_judgment: Optional[str],
    judgment_reason: Optional[str]
) -> str:
    """LLM分析セクションのHTMLを生成"""
    if not _is_valid_llm_judgment(relevance_judgment):
        return ""

    reason_text = html.escape(str(judgment_reason)) if judgment_reason else "-"

    return f"""<div style="background-color: #f0f7ff; padding: 12px; border-radius: 8px; margin: 8px 0;"><div style="font-weight: 600; margin-bottom: 5px;">LLM分析</div><div>関連性: {html.escape(str(relevance_judgment))}</div><div>根拠: {reason_text}</div></div>"""


def _clean_excel_artifacts(text: str) -> str:
    """Excel由来のアーティファクト(_x000D_ 等)を除去"""
    return text.replace("_x000D_", "")


def format_response_card(
    number: int,
    similarity: float,
    query: str,
    answer: str,
    category: Optional[str] = None,
    relevance_judgment: Optional[str] = None,
    judgment_reason: Optional[str] = None,
    scenario_id: Optional[str] = None,
    is_correct: bool = False,
) -> str:
    """検索結果カードのHTMLを生成"""
    query = html.escape(_clean_excel_artifacts(str(query)))
    answer = html.escape(_clean_excel_artifacts(str(answer)))

    category_badge = _create_category_badge(category)
    correct_badge = _create_correct_badge(is_correct)
    scenario_id_badge = _create_scenario_id_badge(scenario_id)
    relevance_badge = _create_relevance_badge(relevance_judgment)

    card_bg_color = "#e8f5e9" if is_correct else "white"
    card_border_color = "#81c784" if is_correct else "#e0e0e0"

    base_card = f"""<div class="response-card" style="border: 2px solid {card_border_color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: {card_bg_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><div style="color: #666; margin-bottom: 10px; font-size: 0.95em; padding-bottom: 8px; border-bottom: 1px solid #eee;">候補 {number} (類似度: <strong>{similarity:.4f}</strong>){correct_badge}{scenario_id_badge}{category_badge}{relevance_badge}</div><div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;"><div style="font-weight: 600; margin-bottom: 5px;">類似質問内容:</div><div style="white-space: pre-wrap;">{query}</div></div><div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;"><div style="font-weight: 600; margin-bottom: 5px;">回答:</div><div style="white-space: pre-wrap;">{answer}</div></div>"""

    llm_section = _create_llm_analysis_section(relevance_judgment, judgment_reason)
    return base_card + llm_section + "</div>"


def render_vector_weight_slider(default_value: float, key: Optional[str] = None) -> float:
    """ベクトル重みスライダーを描画し、選択値を返す。
    config の更新は呼び出し側で行うこと（副作用なし）。"""
    vector_weight = st.slider(
        "検索バランス",
        0.0, 1.0,
        default_value, 0.1,
        key=key,
        help="左：ワード重視 / 右：意味重視"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.caption("← ワード重視")
    with col2:
        st.caption("意味重視 →")

    return vector_weight


# 注: save_chat_history() は Processor に依存するため shared.py には含めない。
# 各UI（chat.py, ops_ui.py）にそれぞれ配置する。
