"""Custom exception classes for the application."""

from typing import Any, Dict, Optional


class BaseAppException(Exception):
    """Base exception class for application-specific errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(BaseAppException):
    """Raised when there's a configuration error."""
    pass


class DatabaseError(BaseAppException):
    """Raised when there's a database-related error."""
    pass


class VectorDBError(BaseAppException):
    """Raised when there's a vector database error."""
    pass


class EmbeddingError(BaseAppException):
    """Raised when there's an embedding generation error."""
    pass


class LLMError(BaseAppException):
    """Raised when there's an LLM-related error."""
    pass


class DataProcessingError(BaseAppException):
    """Raised when there's a data processing error."""
    pass


class ValidationError(BaseAppException):
    """Raised when data validation fails."""
    pass


class AuthenticationError(BaseAppException):
    """Raised when authentication fails."""
    pass


class RateLimitError(BaseAppException):
    """Raised when rate limit is exceeded."""
    pass


class DocumentProcessingError(DataProcessingError):
    """Raised when document processing fails."""
    pass


class QueryProcessingError(BaseAppException):
    """Raised when query processing fails."""
    pass


class RetrievalError(BaseAppException):
    """Raised when document retrieval fails."""
    pass


class InferenceError(BaseAppException):
    """Raised when inference pipeline fails."""
    pass