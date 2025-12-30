"""Pydantic response models for the inference pipeline."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class InferenceMetadata(BaseModel):
    """Metadata about the inference operation."""
    
    model_used: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    retrieval_used: bool
    tool_used: bool = False
    route_taken: str  # "retrieve", "tool", "respond"


class InferenceResponse(BaseModel):
    """Complete inference response."""
    
    query: str
    response: str
    session_id: str
    metadata: InferenceMetadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamChunk(BaseModel):
    """Single chunk in streaming response."""
    
    content: str
    is_final: bool = False
    error: Optional[str] = None