"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""
    
    query: str = Field(..., min_length=1, max_length=500, description="User message")
    session_id: str = Field(..., min_length=1, description="Session identifier")


class ChatResponse(BaseModel):
    """Response model for non-streaming chat."""
    
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class MessageResponse(BaseModel):
    """Single message in a session."""
    
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")


class SessionResponse(BaseModel):
    """Summary response for a session."""
    
    session_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    preview: str = Field(default="", description="Preview of first user message")


class SessionDetailResponse(BaseModel):
    """Detailed response for a single session."""
    
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]


class SessionListResponse(BaseModel):
    """Response for listing all sessions."""
    
    sessions: List[SessionResponse]
    total: int


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    
    status: str = Field(..., description="Overall health status: healthy or unhealthy")
    services: Dict[str, str] = Field(default_factory=dict, description="Individual service statuses")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
