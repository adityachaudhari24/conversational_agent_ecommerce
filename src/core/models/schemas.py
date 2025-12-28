"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = {
        "from_attributes": True,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
    }


# Document and Chunk Schemas
class DocumentMetadata(BaseSchema):
    """Document metadata schema."""
    
    source: str = Field(..., description="Document source path or URL")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    created_at: Optional[datetime] = Field(None, description="Document creation date")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_type: Optional[str] = Field(None, description="File type/extension")
    language: str = Field(default="en", description="Document language")


class DocumentChunk(BaseSchema):
    """Document chunk schema."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Chunk position in document")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Text embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseSchema):
    """Document schema."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Document ID")
    content: str = Field(..., description="Full document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks")
    processed: bool = Field(default=False, description="Processing status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)


# Chat and Conversation Schemas
class ChatMessage(BaseSchema):
    """Chat message schema."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Message ID")
    session_id: str = Field(..., description="Chat session ID")
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatSession(BaseSchema):
    """Chat session schema."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    title: Optional[str] = Field(None, description="Session title")
    messages: List[ChatMessage] = Field(default_factory=list, description="Session messages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    is_active: bool = Field(default=True, description="Session status")


# Query and Retrieval Schemas
class QueryRequest(BaseSchema):
    """Query request schema."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    include_metadata: bool = Field(default=True, description="Include result metadata")


class RetrievalResult(BaseSchema):
    """Document retrieval result schema."""
    
    chunk_id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    rank: int = Field(..., description="Result ranking")


class QueryResponse(BaseSchema):
    """Query response schema."""
    
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    results: List[RetrievalResult] = Field(default_factory=list, description="Retrieved results")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="LLM model used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Product-specific Schemas (E-commerce)
class ProductInfo(BaseSchema):
    """Product information schema."""
    
    id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    description: Optional[str] = Field(None, description="Product description")
    category: Optional[str] = Field(None, description="Product category")
    brand: Optional[str] = Field(None, description="Product brand")
    price: Optional[float] = Field(None, description="Product price")
    currency: str = Field(default="USD", description="Price currency")
    availability: Optional[str] = Field(None, description="Product availability")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Product rating")
    review_count: Optional[int] = Field(None, ge=0, description="Number of reviews")
    features: List[str] = Field(default_factory=list, description="Product features")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Product specs")
    images: List[str] = Field(default_factory=list, description="Product image URLs")
    url: Optional[str] = Field(None, description="Product URL")


# API Response Schemas
class HealthResponse(BaseSchema):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Uptime in seconds")


class ErrorResponse(BaseSchema):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Pipeline Status Schemas
class PipelineStatus(BaseSchema):
    """Pipeline status schema."""
    
    pipeline_name: str = Field(..., description="Pipeline name")
    status: str = Field(..., description="Pipeline status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: Optional[str] = Field(None, description="Status message")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error: Optional[str] = Field(None, description="Error message if failed")


class IngestionRequest(BaseSchema):
    """Data ingestion request schema."""
    
    source_path: str = Field(..., description="Source data path")
    source_type: str = Field(..., description="Source type (csv, pdf, json, etc.)")
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Chunk size")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Chunk overlap")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    overwrite: bool = Field(default=False, description="Overwrite existing data")