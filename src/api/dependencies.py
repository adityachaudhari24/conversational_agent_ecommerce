"""Dependency injection for FastAPI."""

from functools import lru_cache
from typing import Optional

from .config import settings
from .services.session_store import SessionStore


# Global inference pipeline instance (initialized on startup)
_inference_pipeline = None


def set_inference_pipeline(pipeline) -> None:
    """Set the global inference pipeline instance.
    
    Called during application startup.
    """
    global _inference_pipeline
    _inference_pipeline = pipeline


def get_inference_pipeline():
    """Get the inference pipeline dependency.
    
    Returns:
        InferencePipeline instance or None if not initialized
    """
    return _inference_pipeline


@lru_cache()
def get_session_store() -> SessionStore:
    """Get the session store dependency.
    
    Returns:
        SessionStore instance (cached)
    """
    return SessionStore(storage_dir=settings.session_dir)
