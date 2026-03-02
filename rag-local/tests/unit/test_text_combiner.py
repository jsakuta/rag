"""TextCombiner のユニットテスト"""
import pytest
from src.core.search.text_combiner import TextCombiner, get_text_combiner


class TestTextCombinerBuild:
    """TextCombiner.build() のテスト"""

    def setup_method(self):
        self.combiner = TextCombiner()

    def test_build_full(self):
        """階層・質問・回答すべてあり"""
        result = self.combiner.build("口座 > 開設", "口座開設方法は？", "窓口へお越しください")
        assert result == "分類: 口座 > 開設 | 質問: 口座開設方法は？ | 回答: 窓口へお越しください"

    def test_build_query_and_answer_only(self):
        """FAQ形式（階層なし）"""
        result = self.combiner.build(query="質問文", answer="回答文")
        assert result == "質問: 質問文 | 回答: 回答文"

    def test_build_empty_strings(self):
        """空文字列は除外"""
        result = self.combiner.build("", "質問のみ", "")
        assert result == "質問: 質問のみ"

    def test_build_whitespace_only(self):
        """空白のみも除外"""
        result = self.combiner.build("  ", "質問", "  ")
        assert result == "質問: 質問"

    def test_build_all_empty(self):
        """全て空"""
        result = self.combiner.build("", "", "")
        assert result == ""

    def test_build_roundtrip_with_parse(self):
        """build → parse のラウンドトリップ"""
        original = self.combiner.build("Lv1 > Lv2", "質問テキスト", "回答テキスト")
        parsed = self.combiner.parse(original)
        assert parsed.hierarchy == "Lv1 > Lv2"
        assert parsed.query == "質問テキスト"
        assert parsed.answer == "回答テキスト"


class TestTextCombinerParse:
    """既存 parse() の動作確認テスト"""

    def setup_method(self):
        self.combiner = TextCombiner()

    def test_parse_full(self):
        result = self.combiner.parse("分類: A > B | 質問: Q | 回答: A")
        assert result.hierarchy == "A > B"
        assert result.query == "Q"
        assert result.answer == "A"

    def test_parse_partial(self):
        result = self.combiner.parse("質問: Q | 回答: A")
        assert result.hierarchy == ""
        assert result.query == "Q"
        assert result.answer == "A"


class TestGetTextCombiner:
    """シングルトンインスタンスのテスト"""

    def test_returns_same_instance(self):
        a = get_text_combiner()
        b = get_text_combiner()
        assert a is b
