"""Data Inference Pipeline for conversational RAG e-commerce application.

This module provides the inference pipeline that generates conversational responses
using Large Language Models with an agentic workflow powered by LangGraph.

Components:
- LLMClient: Configurable OpenAI LLM client with retry logic
- ConversationManager: In-memory conversation session management
- ResponseGenerator: Prompt construction and response generation
- AgenticWorkflow: LangGraph-based workflow with Router, Retriever, Tool, and Generator nodes
- InferencePipeline: Main orchestrator coordinating all components
"""

from .exceptions import (
    InferenceError,
    ConfigurationError,
    LLMError,
    SessionError,
    StreamingError,
    TimeoutError,
)
from .config import InferenceSettings, get_inference_settings
from .pipeline import InferencePipeline, InferenceConfig, InferenceResult

__all__ = [
    # Exceptions
    "InferenceError",
    "ConfigurationError",
    "LLMError",
    "SessionError",
    "StreamingError",
    "TimeoutError",
    # Configuration
    "InferenceSettings",
    "get_inference_settings",
    # Pipeline
    "InferencePipeline",
    "InferenceConfig",
    "InferenceResult",
]
