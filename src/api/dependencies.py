"""Dependency injection for FastAPI."""

from typing import Optional

from src.pipelines.inference.conversation.store import ConversationStore


# Global inference pipeline instance (initialized on startup)
_inference_pipeline = None

# Global conversation store instance (initialized on startup)
_conversation_store: Optional[ConversationStore] = None


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


def set_conversation_store(store: ConversationStore) -> None:
    """Set the global conversation store instance.
    
    Called during application startup.
    
    Args:
        store: ConversationStore instance to use globally
    """
    global _conversation_store
    _conversation_store = store


def get_conversation_store() -> ConversationStore:
    """Get the conversation store dependency.
    
    Returns:
        ConversationStore instance
        
    Raises:
        RuntimeError: If conversation store not initialized
    """
    if _conversation_store is None:
        raise RuntimeError(
            "ConversationStore not initialized. "
            "Call set_conversation_store() during application startup."
        )
    return _conversation_store
