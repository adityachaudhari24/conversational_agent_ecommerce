"""Logging utilities for the retrieval pipeline."""

import time
from functools import wraps
from typing import Dict, Any, Optional, Callable
import logging

from src.utils.logging import get_logger, log_performance, LoggerMixin


class RetrievalLoggerMixin(LoggerMixin):
    """Enhanced logger mixin for retrieval pipeline components."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_name = self.__class__.__name__
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this retrieval component."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(
                f"src.pipelines.retrieval.{self._component_name.lower()}",
                context={
                    'pipeline': 'retrieval',
                    'component': self._component_name
                }
            )
        return self._logger
    
    def log_operation_start(self, operation: str, **context) -> None:
        """Log the start of an operation with context."""
        self.logger.info(
            f"Starting {operation}",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'operation_status': 'started',
                    **context
                }
            }
        )
    
    def log_operation_success(self, operation: str, duration_ms: float, **context) -> None:
        """Log successful completion of an operation."""
        self.logger.info(
            f"Completed {operation} in {duration_ms:.2f}ms",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'operation_status': 'completed',
                    'duration_ms': duration_ms,
                    **context
                }
            }
        )
    
    def log_operation_error(self, operation: str, error: Exception, duration_ms: float, **context) -> None:
        """Log operation failure with error details."""
        self.logger.error(
            f"Failed {operation} after {duration_ms:.2f}ms: {str(error)}",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'operation_status': 'failed',
                    'duration_ms': duration_ms,
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    **context
                }
            },
            exc_info=True
        )
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metrics",
            extra={
                'extra_fields': {
                    'metrics_type': 'performance',
                    **metrics
                }
            }
        )


def log_retrieval_operation(operation_name: str):
    """Decorator to log retrieval operations with timing and error handling.
    
    Args:
        operation_name: Name of the operation being logged
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Ensure the instance has logging capabilities
            if not hasattr(self, 'logger'):
                logger = get_logger(
                    f"src.pipelines.retrieval.{self.__class__.__name__.lower()}",
                    context={'pipeline': 'retrieval', 'component': self.__class__.__name__}
                )
            else:
                logger = self.logger
            
            start_time = time.time()
            
            # Log operation start
            if hasattr(self, 'log_operation_start'):
                self.log_operation_start(operation_name)
            else:
                logger.info(f"Starting {operation_name}")
            
            try:
                result = func(self, *args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log success
                if hasattr(self, 'log_operation_success'):
                    self.log_operation_success(operation_name, duration_ms)
                else:
                    logger.info(f"Completed {operation_name} in {duration_ms:.2f}ms")
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log error
                if hasattr(self, 'log_operation_error'):
                    self.log_operation_error(operation_name, e, duration_ms)
                else:
                    logger.error(f"Failed {operation_name} after {duration_ms:.2f}ms: {str(e)}", exc_info=True)
                
                raise
        
        return wrapper
    return decorator


class RetrievalMetricsLogger:
    """Specialized logger for retrieval pipeline metrics."""
    
    def __init__(self, component_name: str):
        """Initialize metrics logger.
        
        Args:
            component_name: Name of the component generating metrics
        """
        self.component_name = component_name
        self.logger = get_logger(
            f"src.pipelines.retrieval.metrics",
            context={
                'pipeline': 'retrieval',
                'component': component_name,
                'metrics_logger': True
            }
        )
    
    def log_query_processing(
        self, 
        query: str, 
        processing_time_ms: float,
        truncated: bool = False,
        normalized_length: int = None
    ) -> None:
        """Log query processing metrics."""
        self.logger.info(
            "Query processed",
            extra={
                'extra_fields': {
                    'metric_type': 'query_processing',
                    'component': self.component_name,
                    'query_length': len(query),
                    'normalized_length': normalized_length,
                    'processing_time_ms': processing_time_ms,
                    'truncated': truncated,
                    'query_hash': hash(query) % 10000  # Anonymized query identifier
                }
            }
        )
    
    def log_search_results(
        self, 
        query: str,
        results_count: int,
        search_time_ms: float,
        search_type: str = "similarity",
        filters_applied: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> None:
        """Log vector search results."""
        self.logger.info(
            "Search completed",
            extra={
                'extra_fields': {
                    'metric_type': 'search_results',
                    'component': self.component_name,
                    'query_hash': hash(query) % 10000,
                    'results_count': results_count,
                    'search_time_ms': search_time_ms,
                    'search_type': search_type,
                    'filters_applied': filters_applied or {},
                    'score_threshold': score_threshold
                }
            }
        )
    
    def log_compression_results(
        self, 
        input_count: int,
        output_count: int,
        compression_time_ms: float,
        compression_ratio: Optional[float] = None
    ) -> None:
        """Log contextual compression results."""
        compression_ratio = compression_ratio or (output_count / input_count if input_count > 0 else 0)
        
        self.logger.info(
            "Compression completed",
            extra={
                'extra_fields': {
                    'metric_type': 'compression_results',
                    'component': self.component_name,
                    'input_documents': input_count,
                    'output_documents': output_count,
                    'filtered_count': input_count - output_count,
                    'compression_ratio': compression_ratio,
                    'compression_time_ms': compression_time_ms
                }
            }
        )
    
    def log_cache_operation(
        self, 
        operation: str,  # "hit", "miss", "set", "evict"
        query_hash: str,
        cache_size: Optional[int] = None,
        hit_rate: Optional[float] = None
    ) -> None:
        """Log cache operations."""
        self.logger.info(
            f"Cache {operation}",
            extra={
                'extra_fields': {
                    'metric_type': 'cache_operation',
                    'component': self.component_name,
                    'cache_operation': operation,
                    'query_hash': query_hash,
                    'cache_size': cache_size,
                    'hit_rate': hit_rate
                }
            }
        )
    
    def log_rewrite_operation(
        self,
        original_query: str,
        rewritten_query: str,
        attempt_number: int,
        rewrite_time_ms: float,
        improvement_reason: Optional[str] = None
    ) -> None:
        """Log query rewrite operations."""
        self.logger.info(
            "Query rewritten",
            extra={
                'extra_fields': {
                    'metric_type': 'query_rewrite',
                    'component': self.component_name,
                    'original_query_hash': hash(original_query) % 10000,
                    'rewritten_query_hash': hash(rewritten_query) % 10000,
                    'attempt_number': attempt_number,
                    'rewrite_time_ms': rewrite_time_ms,
                    'improvement_reason': improvement_reason
                }
            }
        )
    
    def log_evaluation_metrics(
        self, 
        metrics: Dict[str, float],
        evaluation_time_ms: float,
        query_hash: Optional[str] = None
    ) -> None:
        """Log RAGAS evaluation metrics."""
        self.logger.info(
            "Evaluation completed",
            extra={
                'extra_fields': {
                    'metric_type': 'evaluation_metrics',
                    'component': self.component_name,
                    'query_hash': query_hash,
                    'evaluation_time_ms': evaluation_time_ms,
                    **{f"eval_{k}": v for k, v in metrics.items()}
                }
            }
        )
    
    def log_pipeline_metrics(
        self,
        total_time_ms: float,
        query_hash: str,
        documents_retrieved: int,
        cache_hit: bool,
        rewrite_attempts: int = 0,
        compression_applied: bool = False
    ) -> None:
        """Log end-to-end pipeline metrics."""
        self.logger.info(
            "Pipeline completed",
            extra={
                'extra_fields': {
                    'metric_type': 'pipeline_metrics',
                    'component': 'RetrievalPipeline',
                    'query_hash': query_hash,
                    'total_time_ms': total_time_ms,
                    'documents_retrieved': documents_retrieved,
                    'cache_hit': cache_hit,
                    'rewrite_attempts': rewrite_attempts,
                    'compression_applied': compression_applied
                }
            }
        )


def setup_retrieval_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging specifically for the retrieval pipeline.
    
    Args:
        log_level: Logging level for the retrieval pipeline
        
    Returns:
        Logger instance for the retrieval pipeline
    """
    from src.utils.logging import setup_pipeline_logging
    
    return setup_pipeline_logging(
        pipeline_name="retrieval",
        log_level=log_level,
        log_file="logs/retrieval_pipeline.log",
        use_json=True
    )