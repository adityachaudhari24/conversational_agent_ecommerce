"""Data Retrieval Pipeline.

This module implements advanced RAG techniques for retrieving relevant product information
from the vector database based on user queries. It includes semantic search, MMR diversity,
contextual compression, and query rewriting capabilities.
"""

from .config import RetrievalSettings, ConfigurationLoader
from .exceptions import (
    RetrievalError,
    QueryValidationError,
    EmbeddingError,
    SearchError,
    ConnectionError,
    ConfigurationError,
)
from .cache import ResultCache, CacheConfig, CacheEntry
from .models import (
    DocumentScore,
    RetrievalMetadata,
    EvaluationScores,
    RetrievalResponse,
    create_error_response,
)

__all__ = [
    "RetrievalSettings",
    "ConfigurationLoader",
    "RetrievalError",
    "QueryValidationError",
    "EmbeddingError",
    "SearchError",
    "ConnectionError",
    "ConfigurationError",
    "ResultCache",
    "CacheConfig",
    "CacheEntry",
    "DocumentScore",
    "RetrievalMetadata",
    "EvaluationScores",
    "RetrievalResponse",
    "create_error_response",
]