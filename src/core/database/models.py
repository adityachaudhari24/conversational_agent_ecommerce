"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Document(Base):
    """Document model for storing processed documents."""
    
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(200))
    author: Mapped[Optional[str]] = mapped_column(String(100))
    category: Mapped[Optional[str]] = mapped_column(String(100))
    file_type: Mapped[Optional[str]] = mapped_column(String(50))
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    language: Mapped[str] = mapped_column(String(10), default="en")
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )
    
    # Relationship to chunks
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Document chunk model for storing text chunks with embeddings."""
    
    __tablename__ = "document_chunks"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    
    # Relationship to document
    document: Mapped["Document"] = relationship(
        "Document", back_populates="chunks"
    )


class ChatSession(Base):
    """Chat session model for storing conversation sessions."""
    
    __tablename__ = "chat_sessions"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(100))
    title: Mapped[Optional[str]] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )
    
    # Relationship to messages
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    """Chat message model for storing individual messages."""
    
    __tablename__ = "chat_messages"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    
    # Relationship to session
    session: Mapped["ChatSession"] = relationship(
        "ChatSession", back_populates="messages"
    )


class Product(Base):
    """Product model for storing e-commerce product information."""
    
    __tablename__ = "products"
    
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100))
    brand: Mapped[Optional[str]] = mapped_column(String(100))
    price: Mapped[Optional[float]] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(10), default="USD")
    availability: Mapped[Optional[str]] = mapped_column(String(50))
    rating: Mapped[Optional[float]] = mapped_column(Float)
    review_count: Mapped[Optional[int]] = mapped_column(Integer)
    features: Mapped[List[str]] = mapped_column(JSON, default=list)
    specifications: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    images: Mapped[List[str]] = mapped_column(JSON, default=list)
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )


class QueryLog(Base):
    """Query log model for storing user queries and responses."""
    
    __tablename__ = "query_logs"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(36))
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_results: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PipelineRun(Base):
    """Pipeline run model for tracking pipeline executions."""
    
    __tablename__ = "pipeline_runs"
    
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    pipeline_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # running, completed, failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[Optional[str]] = mapped_column(Text)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    results: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration: Mapped[Optional[float]] = mapped_column(Float)  # Duration in seconds