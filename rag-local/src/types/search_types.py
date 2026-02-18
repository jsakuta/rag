# --- src/types/search_types.py ---
"""検索結果の型定義（TypedDict + dataclass）

このモジュールは検索結果の型安全性を提供します。
- SearchResult: 検索結果の基本型
- MultiStageSearchResult: 多段階検索結果の型（Search_Category付き）
- VectorSearchResult: ベクトルDB検索結果の型
- ParsedCombinedText: 結合テキスト解析結果の型
"""

from dataclasses import dataclass, field
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


# ========================================
# dataclass定義（不変オブジェクトとしての型安全化）
# ========================================

@dataclass(frozen=True)
class SearchResult:
    """検索結果のデータクラス（イミュータブル）

    Attributes:
        search_result_q: 検索結果の質問
        search_result_a: 検索結果の回答
        similarity: 統合類似度スコア（0.0-1.0）
        scenario_id: シナリオID（sheet_name_row_index形式）
        sheet_name: Excelシート名
        row_index: Excel行番号
        vector_weight: 使用したベクトル重み
        top_k: 使用したtop_k値
        input_number: 入力番号（1位のみ設定）
        original_query: 元のクエリ（1位のみ設定）
        original_answer: 元の回答（1位のみ設定）
        search_query: 使用した検索クエリ（1位のみ設定）
    """
    search_result_q: str
    search_result_a: str
    similarity: float
    scenario_id: str
    sheet_name: str
    row_index: str
    vector_weight: float
    top_k: int
    input_number: str = ""
    original_query: str = ""
    original_answer: str = ""
    search_query: str = ""

    def to_dict(self) -> SearchResultDict:
        """SearchResultDict形式の辞書に変換"""
        result: SearchResultDict = {
            'Search_Result_Q': self.search_result_q,
            'Search_Result_A': self.search_result_a,
            'Similarity': self.similarity,
            'Scenario_ID': self.scenario_id,
            'Sheet_Name': self.sheet_name,
            'Row_Index': self.row_index,
            'Vector_Weight': self.vector_weight,
            'Top_K': self.top_k,
        }
        if self.input_number:
            result['Input_Number'] = self.input_number
        if self.original_query:
            result['Original_Query'] = self.original_query
        if self.original_answer:
            result['Original_Answer'] = self.original_answer
        if self.search_query:
            result['Search_Query'] = self.search_query
        return result

    @classmethod
    def from_dict(cls, data: SearchResultDict) -> 'SearchResult':
        """辞書からSearchResultを生成"""
        return cls(
            search_result_q=data.get('Search_Result_Q', ''),
            search_result_a=data.get('Search_Result_A', ''),
            similarity=data.get('Similarity', 0.0),
            scenario_id=data.get('Scenario_ID', ''),
            sheet_name=data.get('Sheet_Name', ''),
            row_index=str(data.get('Row_Index', '')),
            vector_weight=data.get('Vector_Weight', 0.0),
            top_k=data.get('Top_K', 0),
            input_number=data.get('Input_Number', ''),
            original_query=data.get('Original_Query', ''),
            original_answer=data.get('Original_Answer', ''),
            search_query=data.get('Search_Query', ''),
        )


@dataclass(frozen=True)
class MultiStageSearchResult(SearchResult):
    """多段階検索結果のデータクラス（Search_Category付き）"""
    search_category: Literal['Both', 'Original_Only', 'LLM_Enhanced_Only'] = 'Both'

    def to_dict(self) -> MultiStageSearchResultDict:
        """MultiStageSearchResultDict形式の辞書に変換"""
        base_dict = super().to_dict()
        result: MultiStageSearchResultDict = {
            **base_dict,
            'Search_Category': self.search_category,
        }
        return result

    @classmethod
    def from_dict(cls, data: MultiStageSearchResultDict) -> 'MultiStageSearchResult':
        """辞書からMultiStageSearchResultを生成"""
        return cls(
            search_result_q=data.get('Search_Result_Q', ''),
            search_result_a=data.get('Search_Result_A', ''),
            similarity=data.get('Similarity', 0.0),
            scenario_id=data.get('Scenario_ID', ''),
            sheet_name=data.get('Sheet_Name', ''),
            row_index=str(data.get('Row_Index', '')),
            vector_weight=data.get('Vector_Weight', 0.0),
            top_k=data.get('Top_K', 0),
            input_number=data.get('Input_Number', ''),
            original_query=data.get('Original_Query', ''),
            original_answer=data.get('Original_Answer', ''),
            search_query=data.get('Search_Query', ''),
            search_category=data.get('Search_Category', 'Both'),
        )


@dataclass(frozen=True)
class VectorSearchResult:
    """ベクトルDB検索結果のデータクラス"""
    id: str
    document: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> VectorSearchResultDict:
        """VectorSearchResultDict形式の辞書に変換"""
        return {
            'id': self.id,
            'document': self.document,
            'similarity': self.similarity,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: VectorSearchResultDict) -> 'VectorSearchResult':
        """辞書からVectorSearchResultを生成"""
        return cls(
            id=data['id'],
            document=data['document'],
            similarity=data['similarity'],
            metadata=data.get('metadata', {}),
        )


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
