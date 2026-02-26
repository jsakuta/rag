# --- src/core/search/__init__.py ---
"""検索エンジンモジュール

Searcherクラスの責務を分離したモジュール:
- VectorSearchEngine: ベクトル検索
- KeywordSearchEngine: キーワード検索
- QueryEnhancer: クエリ拡張（LLM使用）
- MultiStageOrchestrator: 多段階検索のオーケストレーション
- TextCombiner: テキスト結合ユーティリティ
- SearchStrategy: 検索戦略パターン（Original, LLMEnhanced, MultiStage, KeywordFilter）

遅延インポート: 重い依存パッケージ（chromadb, langchain等）がないテスト環境でも
個別モジュールをインポート可能にする。
"""

__all__ = [
    'VectorSearchEngine',
    'KeywordSearchEngine',
    'QueryEnhancer',
    'MultiStageOrchestrator',
    'TextCombiner',
    'SearchStrategy',
    'OriginalSearchStrategy',
    'LLMEnhancedSearchStrategy',
    'MultiStageSearchStrategy',
    'KeywordFilterSearchStrategy',
    'create_strategy',
    'ChromaDBKeywordSearcher',
    'MatchResult',
]


def __getattr__(name):
    if name == 'VectorSearchEngine':
        from src.core.search.vector_search_engine import VectorSearchEngine
        return VectorSearchEngine
    elif name == 'KeywordSearchEngine':
        from src.core.search.keyword_search_engine import KeywordSearchEngine
        return KeywordSearchEngine
    elif name == 'QueryEnhancer':
        from src.core.search.query_enhancer import QueryEnhancer
        return QueryEnhancer
    elif name == 'MultiStageOrchestrator':
        from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
        return MultiStageOrchestrator
    elif name == 'TextCombiner':
        from src.core.search.text_combiner import TextCombiner
        return TextCombiner
    elif name in ('SearchStrategy', 'OriginalSearchStrategy', 'LLMEnhancedSearchStrategy',
                  'MultiStageSearchStrategy', 'KeywordFilterSearchStrategy', 'create_strategy'):
        import src.core.search.search_strategy as mod
        return getattr(mod, name)
    elif name in ('ChromaDBKeywordSearcher', 'MatchResult'):
        import src.core.search.chromadb_keyword_search as mod
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
