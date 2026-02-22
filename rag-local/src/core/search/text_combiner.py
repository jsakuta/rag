# --- src/core/search/text_combiner.py ---
"""テキスト結合ユーティリティ

結合テキストの生成・解析を一元管理。
DRY原則に従い、同じロジックの重複を排除。
"""

from dataclasses import dataclass
from typing import Optional
from src.types.search_types import ParsedCombinedText


@dataclass
class TextCombiner:
    """テキスト結合ユーティリティ

    シナリオ/FAQデータの結合テキスト生成・解析を統一的に行う。

    結合テキストフォーマット:
        「分類: {hierarchy} | 質問: {query} | 回答: {answer}」

    Attributes:
        separator: フィールド区切り文字（デフォルト: " | "）
    """
    separator: str = " | "

    # ラベル定数
    LABEL_HIERARCHY = "分類"
    LABEL_QUERY = "質問"
    LABEL_ANSWER = "回答"

    def parse(self, combined_text: str) -> ParsedCombinedText:
        """結合テキストを解析して構造化データに変換

        Args:
            combined_text: 結合テキスト

        Returns:
            ParsedCombinedText: 解析結果

        Examples:
            >>> combiner = TextCombiner()
            >>> result = combiner.parse('分類: Lv1 > Lv2 | 質問: Q | 回答: A')
            >>> result.hierarchy
            'Lv1 > Lv2'
            >>> result.query
            'Q'
            >>> result.answer
            'A'
        """
        hierarchy = ""
        query = ""
        answer = ""

        # 「|」で分割
        parts = combined_text.split(self.separator)

        for part in parts:
            part = part.strip()
            if part.startswith(f"{self.LABEL_HIERARCHY}: "):
                hierarchy = part[len(f"{self.LABEL_HIERARCHY}: "):].strip()
            elif part.startswith(f"{self.LABEL_QUERY}: "):
                query = part[len(f"{self.LABEL_QUERY}: "):].strip()
            elif part.startswith(f"{self.LABEL_ANSWER}: "):
                answer = part[len(f"{self.LABEL_ANSWER}: "):].strip()

        return ParsedCombinedText(
            hierarchy=hierarchy,
            query=query,
            answer=answer
        )

    def build_display_query(
        self,
        parsed_text: ParsedCombinedText,
        source: str,
        metadata_hierarchy: Optional[str] = None
    ) -> str:
        """表示用のクエリ文字列を構築

        シナリオデータの場合は「階層 > 質問」形式、
        FAQデータの場合は質問のみを返す。

        Args:
            parsed_text: 解析済み結合テキスト
            source: データソース ('scenario' or 'history_data')
            metadata_hierarchy: メタデータからの階層情報（オプション）

        Returns:
            str: 表示用クエリ文字列
        """
        if source == 'scenario':
            # シナリオデータ: 階層構造 + 質問を表示
            hierarchy = metadata_hierarchy or parsed_text.hierarchy
            query = parsed_text.query
            if hierarchy and query:
                return f"{hierarchy} > {query}"
            elif hierarchy:
                return hierarchy
            else:
                return query
        else:
            # FAQデータ: 質問のみ
            return parsed_text.query


# グローバルインスタンス（シングルトンパターン）
_default_combiner: Optional[TextCombiner] = None


def get_text_combiner() -> TextCombiner:
    """デフォルトのTextCombinerインスタンスを取得"""
    global _default_combiner
    if _default_combiner is None:
        _default_combiner = TextCombiner()
    return _default_combiner
