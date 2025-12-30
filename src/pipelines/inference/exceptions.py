"""Custom exception classes for the inference pipeline.

This module defines a hierarchy of exceptions specific to the inference pipeline,
enabling precise error handling and graceful failure recovery.
"""

from typing import Any, Dict, List, Optional


class InferenceError(Exception):
    """Base exception for inference pipeline errors.
    
    All inference-specific exceptions inherit from this class,
    allowing for broad exception handling when needed.
    
    Attributes:
        message: Human-readable error description
        error_code: Optional machine-readable error code
        details: Optional dictionary with additional error context
    """
    
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
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ConfigurationError(InferenceError):
    """Raised when configuration is invalid or missing.
    
    This exception is raised when:
    - Required environment variables are missing
    - API keys are invalid or not provided
    - Configuration values are out of valid range
    - YAML configuration files cannot be loaded
    
    Attributes:
        missing_keys: List of missing configuration keys
    """
    
    def __init__(
        self,
        message: str,
        missing_keys: Optional[List[str]] = None,
        error_code: Optional[str] = "CONFIG_ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.missing_keys = missing_keys or []
        details = details or {}
        if self.missing_keys:
            details["missing_keys"] = self.missing_keys
        super().__init__(message, error_code, details)


class LLMError(InferenceError):
    """Raised when LLM API calls fail.
    
    This exception is raised when:
    - LLM API returns an error response
    - API rate limits are exceeded
    - Network errors occur during API calls
    - All retry attempts are exhausted
    
    Attributes:
        status_code: HTTP status code from the API response
        provider: Name of the LLM provider (e.g., "openai")
        model: Model name that was being used
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_code: Optional[str] = "LLM_ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.status_code = status_code
        self.provider = provider
        self.model = model
        details = details or {}
        if status_code is not None:
            details["status_code"] = status_code
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
        super().__init__(message, error_code, details)


class SessionError(InferenceError):
    """Raised when session operations fail.
    
    This exception is raised when:
    - Session data is corrupted
    - Session operations fail unexpectedly
    
    Note: Session not found is NOT an error - a new session is created automatically.
    
    Attributes:
        session_id: ID of the session that caused the error
    """
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        error_code: Optional[str] = "SESSION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.session_id = session_id
        details = details or {}
        if session_id:
            details["session_id"] = session_id
        super().__init__(message, error_code, details)


class StreamingError(InferenceError):
    """Raised when streaming fails mid-response.
    
    This exception is raised when:
    - Connection is lost during streaming
    - LLM stops generating unexpectedly
    - Stream parsing fails
    
    Attributes:
        partial_response: The response content received before failure
        chunks_received: Number of chunks received before failure
    """
    
    def __init__(
        self,
        message: str,
        partial_response: Optional[str] = None,
        chunks_received: int = 0,
        error_code: Optional[str] = "STREAMING_ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.partial_response = partial_response
        self.chunks_received = chunks_received
        details = details or {}
        if partial_response is not None:
            details["partial_response"] = partial_response
        details["chunks_received"] = chunks_received
        super().__init__(message, error_code, details)


class TimeoutError(InferenceError):
    """Raised when operation exceeds timeout.
    
    This exception is raised when:
    - LLM response takes too long
    - Retrieval operation times out
    - Overall inference exceeds configured timeout
    
    Attributes:
        timeout_seconds: The timeout value that was exceeded
        operation: Name of the operation that timed out
    """
    
    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[int] = None,
        operation: Optional[str] = None,
        error_code: Optional[str] = "TIMEOUT_ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        super().__init__(message, error_code, details)
