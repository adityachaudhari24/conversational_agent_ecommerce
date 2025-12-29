"""Custom exceptions for the retrieval pipeline."""

from typing import Dict, Any, List, Optional


class RetrievalError(Exception):
    """Base exception for retrieval pipeline errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class QueryValidationError(RetrievalError):
    """Raised when query validation fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, validation_errors: Optional[List[str]] = None):
        super().__init__(message)
        self.query = query
        self.validation_errors = validation_errors or []
        self.details = {
            "query": query,
            "validation_errors": self.validation_errors
        }


class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, model: Optional[str] = None):
        super().__init__(message)
        self.query = query
        self.model = model
        self.details = {
            "query": query,
            "model": model
        }


class SearchError(RetrievalError):
    """Raised when vector search fails."""
    
    def __init__(self, message: str, search_params: Optional[Dict[str, Any]] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.search_params = search_params or {}
        self.error_code = error_code
        self.details = {
            "search_params": self.search_params,
            "error_code": error_code
        }


class ConnectionError(RetrievalError):
    """Raised when external service connection fails."""
    
    def __init__(self, message: str, service: Optional[str] = None, endpoint: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.service = service
        self.endpoint = endpoint
        self.status_code = status_code
        self.details = {
            "service": service,
            "endpoint": endpoint,
            "status_code": status_code
        }


class ConfigurationError(RetrievalError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, missing_keys: Optional[List[str]] = None, invalid_values: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.missing_keys = missing_keys or []
        self.invalid_values = invalid_values or {}
        self.details = {
            "missing_keys": self.missing_keys,
            "invalid_values": self.invalid_values
        }