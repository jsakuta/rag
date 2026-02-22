# --- src/core/__init__.py ---
"""コアモジュール

遅延インポート: 依存パッケージ（langchain等）がないテスト環境でも
サブモジュール（search/）を個別にインポート可能にする。
"""

__all__ = [
    'Searcher',
    'Processor',
    'JudgmentSupport',
]


def __getattr__(name):
    if name == 'Searcher':
        from src.core.searcher import Searcher
        return Searcher
    elif name == 'Processor':
        from src.core.processor import Processor
        return Processor
    elif name == 'JudgmentSupport':
        from src.core.judgment_support import JudgmentSupport
        return JudgmentSupport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
