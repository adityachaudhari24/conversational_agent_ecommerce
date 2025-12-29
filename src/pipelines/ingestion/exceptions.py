"""
Custom exceptions for the data ingestion pipeline.

These exceptions provide specific error types for different failure modes
in the ingestion process, enabling better error handling and debugging.
"""

from typing import Dict, List, Any, Optional


class IngestionError(Exception):
    """Base exception for ingestion pipeline errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(IngestionError):
    """
    Raised when data validation fails.
    
    This includes missing required columns, invalid data formats,
    or data that doesn't meet quality standards.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class ConfigurationError(IngestionError):
    """
    Raised when configuration is invalid or missing.
    
    This includes missing environment variables, invalid config files,
    or incompatible configuration values.
    """
    
    def __init__(self, message: str, missing_keys: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []


class ConnectionError(IngestionError):
    """
    Raised when external service connection fails.
    
    This includes failures connecting to APIs (OpenAI, Pinecone),
    databases, or other external services.
    """
    
    def __init__(self, message: str, service: Optional[str] = None):
        super().__init__(message)
        self.service = service


class DataQualityError(IngestionError):
    """
    Raised when data quality falls below acceptable thresholds.
    
    This occurs when too many records fail validation or processing,
    indicating potential issues with the source data.
    """
    
    def __init__(self, message: str, failure_rate: Optional[float] = None):
        super().__init__(message)
        self.failure_rate = failure_rate