"""
Pydantic response models for the data retrieval pipeline.

This module defines the structured response schemas used by the retrieval pipeline
to ensure consistent API responses and data validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class DocumentScore(BaseModel):
    """Document with its similarity score."""
    
    content: str = Field(..., description="The document content/text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Document metadata including product info"
    )
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Similarity score between 0.0 and 1.0"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Great phone with excellent camera quality...",
                "metadata": {
                    "product_name": "iPhone 14 Pro",
                    "price": 999.99,
                    "rating": 4.5,
                    "source": "customer_review"
                },
                "score": 0.85
            }
        }


class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval operation."""
    
    query_normalized: str = Field(..., description="The normalized query text")
    query_truncated: bool = Field(
        default=False, 
        description="Whether the query was truncated due to length"
    )
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters that were applied to the search"
    )
    documents_fetched: int = Field(
        ..., 
        ge=0, 
        description="Number of documents initially fetched from vector store"
    )
    documents_after_compression: int = Field(
        ..., 
        ge=0, 
        description="Number of documents after contextual compression"
    )
    query_rewritten: bool = Field(
        default=False, 
        description="Whether the query was rewritten for better results"
    )
    rewrite_attempts: int = Field(
        default=0, 
        ge=0, 
        description="Number of query rewrite attempts made"
    )
    cache_hit: bool = Field(
        default=False, 
        description="Whether the result was served from cache"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_normalized": "best smartphone camera quality",
                "query_truncated": False,
                "filters_applied": {"min_rating": 4.0, "max_price": 1000.0},
                "documents_fetched": 20,
                "documents_after_compression": 4,
                "query_rewritten": True,
                "rewrite_attempts": 1,
                "cache_hit": False
            }
        }


class EvaluationScores(BaseModel):
    """RAGAS evaluation scores with enhanced metadata."""
    
    context_precision: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="RAGAS context precision score"
    )
    answer_relevancy: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0,
        description="RAGAS answer relevancy score"
    )
    evaluation_time_ms: Optional[float] = Field(
        None, 
        ge=0.0,
        description="Time taken to compute evaluation metrics in milliseconds"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if evaluation failed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "context_precision": 0.92,
                "answer_relevancy": 0.88,
                "evaluation_time_ms": 1250.5,
                "error": None
            }
        }


class RetrievalResponse(BaseModel):
    """Complete retrieval response with all metadata and evaluation scores."""
    
    query: str = Field(..., description="The original user query")
    documents: List[DocumentScore] = Field(
        default_factory=list,
        description="Retrieved documents with similarity scores"
    )
    formatted_context: str = Field(
        ..., 
        description="Formatted context string ready for LLM consumption"
    )
    metadata: RetrievalMetadata = Field(
        ..., 
        description="Detailed metadata about the retrieval operation"
    )
    evaluation: Optional[EvaluationScores] = Field(
        None,
        description="RAGAS evaluation scores if evaluation is enabled"
    )
    latency_ms: float = Field(
        ..., 
        ge=0.0,
        description="Total retrieval latency in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the retrieval was completed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "best smartphone with good camera",
                "documents": [
                    {
                        "content": "iPhone 14 Pro has excellent camera...",
                        "metadata": {"product_name": "iPhone 14 Pro", "price": 999.99},
                        "score": 0.89
                    }
                ],
                "formatted_context": "Title: iPhone 14 Pro\nPrice: $999.99\nRating: 4.5\nReview: iPhone 14 Pro has excellent camera...",
                "metadata": {
                    "query_normalized": "best smartphone with good camera",
                    "query_truncated": False,
                    "filters_applied": {},
                    "documents_fetched": 20,
                    "documents_after_compression": 4,
                    "query_rewritten": False,
                    "rewrite_attempts": 0,
                    "cache_hit": False
                },
                "evaluation": {
                    "context_precision": 0.92,
                    "answer_relevancy": 0.88,
                    "evaluation_time_ms": 1250.5,
                    "error": None
                },
                "latency_ms": 342.7,
                "timestamp": "2024-01-15T10:30:45.123456"
            }
        }


# Utility function to create a minimal response for error cases
def create_error_response(
    query: str, 
    error_message: str, 
    latency_ms: float = 0.0
) -> RetrievalResponse:
    """Create a minimal RetrievalResponse for error cases."""
    return RetrievalResponse(
        query=query,
        documents=[],
        formatted_context="",
        metadata=RetrievalMetadata(
            query_normalized=query.strip(),
            documents_fetched=0,
            documents_after_compression=0
        ),
        evaluation=EvaluationScores(error=error_message),
        latency_ms=latency_ms
    )