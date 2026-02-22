"""ui/shared.py 共通UI部品の特性テスト

リファクタリング後の shared.py が正しく動作することを検証する。
テスト対象はすべて純粋関数（Streamlit 依存なし）のみ。
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# shared.py のモジュールレベル依存をモックしてからインポート
# 依存チェーン: ui.shared -> streamlit, config (-> src.utils.logger, yaml), dotenv
_mock_st = MagicMock()
sys.modules.setdefault("streamlit", _mock_st)

# dotenv が未インストールの場合のみモック
if "dotenv" not in sys.modules:
    _mock_dotenv = MagicMock()
    _mock_dotenv.load_dotenv = MagicMock()
    sys.modules["dotenv"] = _mock_dotenv

from ui.shared import (
    format_message,
    _create_badge,
    _create_category_badge,
    _create_correct_badge,
    _create_scenario_id_badge,
    _create_relevance_badge,
    _is_valid_llm_judgment,
    _create_llm_analysis_section,
    format_response_card,
)


# ---------------------------------------------------------------------------
# format_message
# ---------------------------------------------------------------------------
class TestFormatMessage:
    """format_message() のテスト"""

    def test_user_message_alignment(self):
        """ユーザーメッセージは右寄せ (flex-end)"""
        result = format_message("hello", is_user=True)
        assert "flex-end" in result

    def test_bot_message_alignment(self):
        """ボットメッセージは左寄せ (flex-start)"""
        result = format_message("hello", is_user=False)
        assert "flex-start" in result

    def test_user_background_color(self):
        """ユーザーメッセージの背景色は #e6f3ff"""
        result = format_message("hello", is_user=True)
        assert "#e6f3ff" in result

    def test_bot_background_color(self):
        """ボットメッセージの背景色は #f5f5f5"""
        result = format_message("hello", is_user=False)
        assert "#f5f5f5" in result

    def test_xss_escape(self):
        """HTML 特殊文字がエスケープされる"""
        result = format_message('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_default_is_bot(self):
        """is_user のデフォルトは False（ボット側）"""
        result = format_message("test")
        assert "flex-start" in result
        assert "#f5f5f5" in result

    def test_contains_message_text(self):
        """出力に元のメッセージテキストが含まれる"""
        result = format_message("特定のテスト文")
        assert "特定のテスト文" in result

    def test_user_margin_left_auto(self):
        """ユーザーメッセージには margin-left: auto が付く"""
        result = format_message("hello", is_user=True)
        assert "margin-left: auto;" in result

    def test_bot_no_margin_left_auto(self):
        """ボットメッセージには margin-left: auto が付かない"""
        result = format_message("hello", is_user=False)
        # margin-left: auto; はボット側には含まれない
        # ただし空文字列なので含まれないことを確認
        assert "margin-left: auto;" not in result


# ---------------------------------------------------------------------------
# _create_badge
# ---------------------------------------------------------------------------
class TestCreateBadge:
    """_create_badge() のテスト"""

    def test_returns_span_html(self):
        """HTML span タグが生成される"""
        result = _create_badge("テスト", "#fff", "#000")
        assert "<span" in result
        assert "</span>" in result

    def test_background_color_applied(self):
        """指定した背景色がスタイルに含まれる"""
        result = _create_badge("テスト", "#abcdef", "#000")
        assert "#abcdef" in result

    def test_text_color_applied(self):
        """指定したテキスト色がスタイルに含まれる"""
        result = _create_badge("テスト", "#fff", "#123456")
        assert "#123456" in result

    def test_bold_default(self):
        """デフォルトで bold が有効"""
        result = _create_badge("テスト", "#fff", "#000")
        assert "font-weight: bold;" in result

    def test_bold_false(self):
        """bold=False のとき font-weight: bold がない"""
        result = _create_badge("テスト", "#fff", "#000", bold=False)
        assert "font-weight: bold;" not in result

    def test_text_content(self):
        """バッジ内にテキストが含まれる"""
        result = _create_badge("ラベル文字", "#fff", "#000")
        assert "ラベル文字" in result


# ---------------------------------------------------------------------------
# _create_category_badge
# ---------------------------------------------------------------------------
class TestCreateCategoryBadge:
    """_create_category_badge() のテスト"""

    def test_none_returns_empty(self):
        """None の場合は空文字列"""
        assert _create_category_badge(None) == ""

    def test_empty_string_returns_empty(self):
        """空文字列の場合は空文字列"""
        assert _create_category_badge("") == ""

    def test_both_label(self):
        """'Both' は '両方' に変換される"""
        result = _create_category_badge("Both")
        assert "両方" in result

    def test_original_only_label(self):
        """'Original_Only' は '原文のみ' に変換される"""
        result = _create_category_badge("Original_Only")
        assert "原文のみ" in result

    def test_llm_enhanced_only_label(self):
        """'LLM_Enhanced_Only' は 'LLMのみ' に変換される"""
        result = _create_category_badge("LLM_Enhanced_Only")
        assert "LLMのみ" in result

    def test_unknown_category_uses_raw_value(self):
        """未知のカテゴリはそのまま表示される"""
        result = _create_category_badge("CustomCategory")
        assert "CustomCategory" in result

    def test_both_colors(self):
        """'Both' のバッジに正しい色が使われる"""
        result = _create_category_badge("Both")
        assert "#e1bee7" in result  # bg
        assert "#6a1b9a" in result  # text


# ---------------------------------------------------------------------------
# _create_correct_badge
# ---------------------------------------------------------------------------
class TestCreateCorrectBadge:
    """_create_correct_badge() のテスト"""

    def test_true_shows_star(self):
        """True のとき ★正解 が表示される"""
        result = _create_correct_badge(True)
        assert "★正解" in result

    def test_false_returns_empty(self):
        """False のとき空文字列を返す"""
        assert _create_correct_badge(False) == ""


# ---------------------------------------------------------------------------
# _create_scenario_id_badge
# ---------------------------------------------------------------------------
class TestCreateScenarioIdBadge:
    """_create_scenario_id_badge() のテスト"""

    def test_none_returns_empty(self):
        """None の場合は空文字列"""
        assert _create_scenario_id_badge(None) == ""

    def test_empty_returns_empty(self):
        """空文字列の場合は空文字列"""
        assert _create_scenario_id_badge("") == ""

    def test_displays_id(self):
        """シナリオID が表示される"""
        result = _create_scenario_id_badge("smile-bot_129")
        assert "smile-bot_129" in result

    def test_xss_escape(self):
        """HTML 特殊文字がエスケープされる"""
        result = _create_scenario_id_badge('<img src=x onerror="alert(1)">')
        assert "<img" not in result
        assert "&lt;img" in result


# ---------------------------------------------------------------------------
# _create_relevance_badge
# ---------------------------------------------------------------------------
class TestCreateRelevanceBadge:
    """_create_relevance_badge() のテスト"""

    def test_none_returns_empty(self):
        """None の場合は空文字列"""
        assert _create_relevance_badge(None) == ""

    def test_empty_returns_empty(self):
        """空文字列の場合は空文字列"""
        assert _create_relevance_badge("") == ""

    def test_relevant_color_blue(self):
        """'関連あり' はブルー系の背景色"""
        result = _create_relevance_badge("関連あり")
        assert "#bbdefb" in result  # bg
        assert "#1565c0" in result  # text

    def test_needs_check_color_yellow(self):
        """'要確認' はイエロー系の背景色"""
        result = _create_relevance_badge("要確認")
        assert "#fff9c4" in result  # bg
        assert "#f57f17" in result  # text

    def test_not_relevant_color_red(self):
        """'関連なし' はレッド系の背景色"""
        result = _create_relevance_badge("関連なし")
        assert "#ffcdd2" in result  # bg
        assert "#c62828" in result  # text

    def test_xss_escape(self):
        """HTML 特殊文字がエスケープされる"""
        result = _create_relevance_badge('<b>関連あり</b>')
        assert "<b>" not in result
        assert "&lt;b&gt;" in result

    def test_judgment_with_extra_text(self):
        """'関連あり (90%信頼度)' のようなテキストでも最初の単語でスタイル判定"""
        result = _create_relevance_badge("関連あり (90%信頼度)")
        assert "#bbdefb" in result  # 関連ありのスタイルが適用される


# ---------------------------------------------------------------------------
# _is_valid_llm_judgment
# ---------------------------------------------------------------------------
class TestIsValidLlmJudgment:
    """_is_valid_llm_judgment() のテスト"""

    def test_none_is_invalid(self):
        """None は無効"""
        assert _is_valid_llm_judgment(None) is False

    def test_empty_string_is_invalid(self):
        """空文字列は無効"""
        assert _is_valid_llm_judgment("") is False

    def test_disabled_is_invalid(self):
        """'判断支援無効' は無効"""
        assert _is_valid_llm_judgment("判断支援無効") is False

    def test_none_string_is_invalid(self):
        """文字列 'None' は無効"""
        assert _is_valid_llm_judgment("None") is False

    def test_relevant_is_valid(self):
        """'関連あり' は有効"""
        assert _is_valid_llm_judgment("関連あり") is True

    def test_not_relevant_is_valid(self):
        """'関連なし' は有効"""
        assert _is_valid_llm_judgment("関連なし") is True

    def test_needs_check_is_valid(self):
        """'要確認' は有効"""
        assert _is_valid_llm_judgment("要確認") is True

    def test_whitespace_only_is_invalid(self):
        """空白のみは無効"""
        assert _is_valid_llm_judgment("   ") is False


# ---------------------------------------------------------------------------
# _create_llm_analysis_section
# ---------------------------------------------------------------------------
class TestCreateLlmAnalysisSection:
    """_create_llm_analysis_section() のテスト"""

    def test_invalid_judgment_returns_empty(self):
        """無効な判定では空文字列を返す"""
        assert _create_llm_analysis_section(None, "理由") == ""

    def test_disabled_judgment_returns_empty(self):
        """'判断支援無効' では空文字列を返す"""
        assert _create_llm_analysis_section("判断支援無効", "理由") == ""

    def test_valid_contains_llm_analysis_header(self):
        """有効な判定では 'LLM分析' が含まれる"""
        result = _create_llm_analysis_section("関連あり", "理由テキスト")
        assert "LLM分析" in result

    def test_valid_contains_judgment(self):
        """有効な判定では判定内容が含まれる"""
        result = _create_llm_analysis_section("関連あり", "理由テキスト")
        assert "関連あり" in result

    def test_valid_contains_reason(self):
        """有効な判定では理由テキストが含まれる"""
        result = _create_llm_analysis_section("関連あり", "キーワードが一致")
        assert "キーワードが一致" in result

    def test_none_reason_shows_dash(self):
        """理由が None のとき '-' が表示される"""
        result = _create_llm_analysis_section("関連あり", None)
        assert "根拠: -" in result

    def test_xss_escape_judgment(self):
        """判定テキストの XSS がエスケープされる"""
        result = _create_llm_analysis_section('<script>alert(1)</script>', "reason")
        # 判定が無効扱いになる（_is_valid_llm_judgment で有効と判定されるため結果に含まれる）
        assert "<script>" not in result

    def test_xss_escape_reason(self):
        """理由テキストの XSS がエスケープされる"""
        result = _create_llm_analysis_section("関連あり", '<img onerror="alert(1)">')
        assert "<img" not in result
        assert "&lt;img" in result


# ---------------------------------------------------------------------------
# format_response_card
# ---------------------------------------------------------------------------
class TestFormatResponseCard:
    """format_response_card() のテスト"""

    def test_similarity_display(self):
        """類似度が小数4桁で表示される"""
        result = format_response_card(1, 0.9123, "質問", "回答")
        assert "0.9123" in result

    def test_query_content(self):
        """質問内容がカードに含まれる"""
        result = format_response_card(1, 0.9, "テスト質問", "テスト回答")
        assert "テスト質問" in result

    def test_answer_content(self):
        """回答内容がカードに含まれる"""
        result = format_response_card(1, 0.9, "テスト質問", "テスト回答")
        assert "テスト回答" in result

    def test_correct_card_green_background(self):
        """正解カードは緑系の背景色"""
        result = format_response_card(1, 0.9, "q", "a", is_correct=True)
        assert "#e8f5e9" in result  # green bg
        assert "#81c784" in result  # green border

    def test_normal_card_white_background(self):
        """通常カードは白背景"""
        result = format_response_card(1, 0.9, "q", "a", is_correct=False)
        assert "background-color: white" in result

    def test_xss_escape_query(self):
        """質問テキストの XSS がエスケープされる"""
        result = format_response_card(1, 0.9, '<script>alert("q")</script>', "a")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_xss_escape_answer(self):
        """回答テキストの XSS がエスケープされる"""
        result = format_response_card(1, 0.9, "q", '<script>alert("a")</script>')
        assert "<script>" not in result

    def test_includes_badges(self):
        """カテゴリ・正解・ID・関連性バッジがすべて含まれる"""
        result = format_response_card(
            1, 0.9, "q", "a",
            category="Both",
            relevance_judgment="関連あり",
            scenario_id="smile-bot_129",
            is_correct=True,
        )
        assert "両方" in result
        assert "★正解" in result
        assert "smile-bot_129" in result
        assert "関連あり" in result

    def test_closes_div(self):
        """最終的な </div> でカードが閉じられる"""
        result = format_response_card(1, 0.9, "q", "a")
        assert result.strip().endswith("</div>")

    def test_llm_section_included_when_valid(self):
        """有効な LLM 判定のとき LLM分析セクションが含まれる"""
        result = format_response_card(
            1, 0.9, "q", "a",
            relevance_judgment="関連あり",
            judgment_reason="キーワード一致",
        )
        assert "LLM分析" in result
        assert "キーワード一致" in result

    def test_llm_section_excluded_when_invalid(self):
        """無効な LLM 判定のとき LLM分析セクションが含まれない"""
        result = format_response_card(
            1, 0.9, "q", "a",
            relevance_judgment="判断支援無効",
            judgment_reason="理由",
        )
        assert "LLM分析" not in result

    def test_number_display(self):
        """候補番号が表示される"""
        result = format_response_card(3, 0.9, "q", "a")
        assert "候補 3" in result
