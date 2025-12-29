"""
Storage Components

Components for storing processed data and embeddings in various backends
(vector databases, file systems, cloud storage).
"""

from .vector_store import VectorStoreManager, VectorStoreConfig

__all__ = [
    "VectorStoreManager",
    "VectorStoreConfig",
]