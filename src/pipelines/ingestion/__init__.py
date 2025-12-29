"""
Data Ingestion Pipeline

This module provides components for loading, transforming, and storing
e-commerce product review data for the conversational RAG application.

The pipeline consists of:
- Document Loader: Reads CSV files and validates data
- Text Processor: Transforms raw data into LangChain Documents
- Text Chunker: Splits large documents for optimal embedding
- Embedding Generator: Creates vector embeddings using OpenAI
- Vector Store: Stores embeddings in Pinecone for similarity search
"""

from .exceptions import (
    IngestionError,
    ValidationError,
    ConfigurationError,
    ConnectionError,
    DataQualityError,
)

from .storage import VectorStoreManager, VectorStoreConfig
from .pipeline import (
    IngestionPipeline,
    PipelineConfig,
    create_pipeline_from_settings,
    run_ingestion_pipeline,
)

__all__ = [
    "IngestionError",
    "ValidationError", 
    "ConfigurationError",
    "ConnectionError",
    "DataQualityError",
    "VectorStoreManager",
    "VectorStoreConfig",
    "IngestionPipeline",
    "PipelineConfig",
    "create_pipeline_from_settings",
    "run_ingestion_pipeline",
]