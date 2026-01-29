# --- src/core/search/__init__.py ---
"""検索エンジンモジュール

Searcherクラスの責務を分離したモジュール:
- VectorSearchEngine: ベクトル検索
- KeywordSearchEngine: キーワード検索
- QueryEnhancer: クエリ拡張（LLM使用）
- MultiStageOrchestrator: 多段階検索のオーケストレーション
- TextCombiner: テキスト結合ユーティリティ
"""

from src.core.search.vector_search_engine import VectorSearchEngine
from src.core.search.keyword_search_engine import KeywordSearchEngine
from src.core.search.query_enhancer import QueryEnhancer
from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
from src.core.search.text_combiner import TextCombiner

__all__ = [
    'VectorSearchEngine',
    'KeywordSearchEngine',
    'QueryEnhancer',
    'MultiStageOrchestrator',
    'TextCombiner',
]
