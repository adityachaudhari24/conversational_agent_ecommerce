"""Logging utilities for the inference pipeline.

This module provides inference-specific logging configuration that integrates
with the global logging system while adding pipeline-specific context.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from src.utils.logging import (
    get_logger,
    setup_pipeline_logging,
    LoggerMixin,
    ContextFilter,
)


# Default inference pipeline context
INFERENCE_CONTEXT = {
    "pipeline": "inference",
    "component": "inference_pipeline",
}


def get_inference_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """Get a logger for inference pipeline components.
    
    This function creates a logger with inference-specific context
    that integrates with the global logging system.
    
    Args:
        name: Logger name (usually __name__ or component name)
        context: Additional context to add to log records
        
    Returns:
        Logger instance with inference context
    """
    # Merge inference context with provided context
    merged_context = {**INFERENCE_CONTEXT}
    if context:
        merged_context.update(context)
    
    return get_logger(f"pipelines.inference.{name}", merged_context)


def setup_inference_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_json: bool = False
) -> logging.Logger:
    """Set up logging specifically for inference pipeline execution.
    
    This function configures logging for the inference pipeline,
    using the global logging utilities with inference-specific settings.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path (defaults to logs/inference_pipeline.log)
        use_json: Whether to use JSON formatting for file output
        
    Returns:
        Logger instance for the inference pipeline
    """
    if log_file is None:
        log_file = "logs/inference_pipeline.log"
    
    return setup_pipeline_logging(
        pipeline_name="inference",
        log_level=log_level,
        log_file=log_file,
        use_json=use_json
    )


def log_inference_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """Decorator to log inference operations with timing and context.
    
    This decorator logs the start, completion, and any errors during
    inference operations, including execution time.
    
    Args:
        operation_name: Name of the operation being logged
        logger: Logger instance to use (defaults to inference logger)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_logger = logger or get_inference_logger(func.__module__)
            start_time = time.time()
            
            # Log operation start
            op_logger.debug(
                f"Starting {operation_name}",
                extra={"extra_fields": {"operation": operation_name}}
            )
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful completion
                op_logger.info(
                    f"{operation_name} completed in {duration_ms:.2f}ms",
                    extra={"extra_fields": {
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "status": "success"
                    }}
                )
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log failure
                op_logger.error(
                    f"{operation_name} failed after {duration_ms:.2f}ms: {e}",
                    extra={"extra_fields": {
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_type": type(e).__name__
                    }},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_async_inference_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """Decorator to log async inference operations with timing and context.
    
    Similar to log_inference_operation but for async functions.
    
    Args:
        operation_name: Name of the operation being logged
        logger: Logger instance to use (defaults to inference logger)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_logger = logger or get_inference_logger(func.__module__)
            start_time = time.time()
            
            # Log operation start
            op_logger.debug(
                f"Starting async {operation_name}",
                extra={"extra_fields": {"operation": operation_name, "async": True}}
            )
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful completion
                op_logger.info(
                    f"{operation_name} completed in {duration_ms:.2f}ms",
                    extra={"extra_fields": {
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "status": "success",
                        "async": True
                    }}
                )
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log failure
                op_logger.error(
                    f"{operation_name} failed after {duration_ms:.2f}ms: {e}",
                    extra={"extra_fields": {
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "async": True
                    }},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class InferenceLoggerMixin(LoggerMixin):
    """Mixin class to add inference-specific logging to any class.
    
    This mixin extends LoggerMixin with inference pipeline context,
    making it easy to add consistent logging to inference components.
    
    Usage:
        class MyComponent(InferenceLoggerMixin):
            def do_something(self):
                self.logger.info("Doing something")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class with inference context."""
        if not hasattr(self, '_logger'):
            self._logger = get_inference_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}",
                context={"class": self.__class__.__name__}
            )
        return self._logger


def log_token_usage(
    logger: logging.Logger,
    model: str,
    input_tokens: int,
    output_tokens: int,
    operation: str = "inference"
) -> None:
    """Log token usage for cost monitoring.
    
    This function logs token usage in a structured format that can be
    easily parsed for cost analysis and monitoring.
    
    Args:
        logger: Logger instance to use
        model: Model name used for the operation
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        operation: Name of the operation
    """
    total_tokens = input_tokens + output_tokens
    
    logger.info(
        f"Token usage for {operation}: {total_tokens} total "
        f"({input_tokens} input, {output_tokens} output)",
        extra={"extra_fields": {
            "operation": operation,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "metric_type": "token_usage"
        }}
    )


def log_latency(
    logger: logging.Logger,
    operation: str,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log operation latency for performance monitoring.
    
    This function logs latency in a structured format that can be
    easily parsed for performance analysis.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation
        latency_ms: Latency in milliseconds
        metadata: Additional metadata to include
    """
    extra_fields = {
        "operation": operation,
        "latency_ms": latency_ms,
        "metric_type": "latency"
    }
    
    if metadata:
        extra_fields.update(metadata)
    
    logger.info(
        f"{operation} latency: {latency_ms:.2f}ms",
        extra={"extra_fields": extra_fields}
    )
