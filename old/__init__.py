# loan_assistant/__init__.py
from .bot import LoanAssistantBot
from .models.metadata import SearchParameters, DocumentMetadata
from .config import (
    DOCS_DIRECTORY, MODEL_PROVIDER, MODEL_NAME,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL, RERANKER_MODEL
)

__version__ = "1.0.0"

__all__ = [
    'LoanAssistantBot',
    'SearchParameters',
    'DocumentMetadata',
    'DOCS_DIRECTORY',
    'MODEL_PROVIDER',
    'MODEL_NAME',
    'EMBEDDING_PROVIDER',
    'EMBEDDING_MODEL',
    'RERANKER_MODEL',
]