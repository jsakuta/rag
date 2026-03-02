# --- src/types/search_types.py ---
"""検索結果の型定義（TypedDict + dataclass）

このモジュールは検索結果の型安全性を提供します。
- TypedDict: 辞書型の型安全化（SearchResultDict等）
- ParsedCombinedText: 結合テキスト解析結果のデータクラス
- 定数クラス: キー名の一元管理
"""

from dataclasses import dataclass
from typing import TypedDict, Optional, List, Dict, Any, Literal


# ========================================
# TypedDict定義（辞書型の型安全化）
# ========================================

class SearchResultDict(TypedDict, total=False):
    """検索結果の辞書型定義

    total=Falseにより、全キーがオプショナルとなる。
    1位の結果のみInput_Number, Original_Query等が設定される。
    """
    Input_Number: str
    Original_Query: str
    Original_Answer: str
    Search_Query: str
    Search_Result_Q: str
    Search_Result_A: str
    Similarity: float
    Scenario_ID: str
    Sheet_Name: str
    Row_Index: str  # int | str だがExcel出力のためstr
    Vector_Weight: float
    Top_K: int


class MultiStageSearchResultDict(SearchResultDict, total=False):
    """多段階検索結果の辞書型定義（Search_Category付き）"""
    Search_Category: Literal['Both', 'Original_Only', 'LLM_Enhanced_Only']


class VectorSearchResultDict(TypedDict):
    """ベクトルDB検索結果の辞書型定義"""
    id: str
    document: str
    similarity: float
    metadata: Dict[str, Any]


class ParsedCombinedTextDict(TypedDict):
    """結合テキスト解析結果の辞書型定義"""
    hierarchy: str
    query: str
    answer: str


class MetadataDict(TypedDict, total=False):
    """メタデータの辞書型定義"""
    source: Literal['scenario', 'history_data']
    hierarchy: str
    sheet_name: str
    row_index: str  # int | str だがChromaDB制約のためstr


class ReferenceDataDict(TypedDict):
    """参照データの辞書型定義"""
    combined_texts: List[str]
    queries: List[str]
    answers: List[str]
    metadatas: List[MetadataDict]


@dataclass(frozen=True)
class ParsedCombinedText:
    """結合テキスト解析結果のデータクラス"""
    hierarchy: str
    query: str
    answer: str

    def to_dict(self) -> ParsedCombinedTextDict:
        """ParsedCombinedTextDict形式の辞書に変換"""
        return {
            'hierarchy': self.hierarchy,
            'query': self.query,
            'answer': self.answer,
        }

    @classmethod
    def from_dict(cls, data: ParsedCombinedTextDict) -> 'ParsedCombinedText':
        """辞書からParsedCombinedTextを生成"""
        return cls(
            hierarchy=data.get('hierarchy', ''),
            query=data.get('query', ''),
            answer=data.get('answer', ''),
        )


# ========================================
# 定数定義（キー名の一元管理）
# ========================================

class SearchResultKeys:
    """検索結果のキー名定数"""
    INPUT_NUMBER = 'Input_Number'
    ORIGINAL_QUERY = 'Original_Query'
    ORIGINAL_ANSWER = 'Original_Answer'
    SEARCH_QUERY = 'Search_Query'
    SEARCH_RESULT_Q = 'Search_Result_Q'
    SEARCH_RESULT_A = 'Search_Result_A'
    SIMILARITY = 'Similarity'
    SCENARIO_ID = 'Scenario_ID'
    SHEET_NAME = 'Sheet_Name'
    ROW_INDEX = 'Row_Index'
    VECTOR_WEIGHT = 'Vector_Weight'
    TOP_K = 'Top_K'
    SEARCH_CATEGORY = 'Search_Category'
    HIERARCHY = 'Hierarchy'
    LV1_CATEGORY = 'Lv1_Category'  # Lv1カテゴリ（「預金関連」「諸届」など）


class MetadataKeys:
    """メタデータのキー名定数"""
    SOURCE = 'source'
    HIERARCHY = 'hierarchy'
    SHEET_NAME = 'sheet_name'
    ROW_INDEX = 'row_index'
    DATE = 'date'  # Lv1カテゴリ（「預金関連」「諸届」など）


class SourceValues:
    """ソース値の定数"""
    SCENARIO = 'scenario'
    HISTORY_DATA = 'history_data'


class SearchCategoryValues:
    """検索カテゴリ値の定数"""
    BOTH = 'Both'
    ORIGINAL_ONLY = 'Original_Only'
    LLM_ENHANCED_ONLY = 'LLM_Enhanced_Only'
