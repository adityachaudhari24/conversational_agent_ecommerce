"""
Pipeline-specific logging utilities and configurations.

This module provides specialized logging functionality for data pipelines,
including performance tracking, stage monitoring, and structured logging
for pipeline execution metrics.
"""

import logging
import logging.handlers
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from .logging import get_logger, LoggerMixin


@dataclass
class PipelineMetrics:
    """Container for pipeline execution metrics."""
    
    pipeline_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get total pipeline duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def total_processed(self) -> int:
        """Get total number of items processed across all stages."""
        return sum(
            stage.get('output_count', 0) 
            for stage in self.stages.values()
        )
    
    def add_stage_metrics(self, stage_name: str, **metrics) -> None:
        """Add metrics for a pipeline stage."""
        self.stages[stage_name] = {
            'timestamp': time.time(),
            **metrics
        }
    
    def add_error(self, stage: str, error: str, **context) -> None:
        """Add an error record."""
        self.errors.append({
            'timestamp': time.time(),
            'stage': stage,
            'error': error,
            **context
        })
    
    def add_warning(self, stage: str, warning: str, **context) -> None:
        """Add a warning record."""
        self.warnings.append({
            'timestamp': time.time(),
            'stage': stage,
            'warning': warning,
            **context
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'pipeline_name': self.pipeline_name,
            'duration': self.duration,
            'total_processed': self.total_processed,
            'stages': self.stages,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }


class PipelineLogger(LoggerMixin):
    """Enhanced logger for pipeline operations with metrics tracking."""
    
    def __init__(self, pipeline_name: str, log_file: Optional[str] = None):
        """
        Initialize pipeline logger.
        
        Args:
            pipeline_name: Name of the pipeline
            log_file: Optional specific log file for this pipeline
        """
        self.pipeline_name = pipeline_name
        self.metrics = PipelineMetrics(pipeline_name)
        
        # Set up pipeline-specific logging context
        self._logger = get_logger(
            f"pipelines.{pipeline_name}",
            context={
                'pipeline': pipeline_name,
                'component': 'pipeline'
            }
        )
        
        # Add file handler if specified
        if log_file:
            self._add_file_handler(log_file)
    
    def _add_file_handler(self, log_file: str) -> None:
        """Add a file handler for pipeline-specific logging."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        self.logger.addHandler(file_handler)
    
    def start_pipeline(self) -> None:
        """Mark the start of pipeline execution."""
        self.metrics.start_time = time.time()
        self.logger.info(f"Starting {self.pipeline_name} pipeline execution")
    
    def end_pipeline(self, success: bool = True) -> None:
        """Mark the end of pipeline execution."""
        self.metrics.end_time = time.time()
        status = "completed successfully" if success else "failed"
        
        self.logger.info(
            f"{self.pipeline_name} pipeline {status} in {self.metrics.duration:.2f}s"
        )
        
        # Log final metrics
        self.log_metrics()
    
    def log_metrics(self) -> None:
        """Log comprehensive pipeline metrics."""
        metrics_dict = self.metrics.to_dict()
        
        self.logger.info("Pipeline Execution Metrics", extra={
            'extra_fields': {
                'metrics': metrics_dict,
                'event_type': 'pipeline_metrics'
            }
        })
    
    @contextmanager
    def stage(self, stage_name: str):
        """Context manager for tracking pipeline stages."""
        start_time = time.time()
        self.logger.info(f"Starting stage: {stage_name}")
        
        try:
            yield self
            duration = time.time() - start_time
            self.logger.info(f"Completed stage: {stage_name} in {duration:.2f}s")
            
            self.metrics.add_stage_metrics(
                stage_name,
                duration=duration,
                status='completed'
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed stage: {stage_name} after {duration:.2f}s - {e}")
            
            self.metrics.add_stage_metrics(
                stage_name,
                duration=duration,
                status='failed',
                error=str(e)
            )
            self.metrics.add_error(stage_name, str(e))
            raise
    
    def log_stage_progress(self, stage_name: str, processed: int, total: int, **kwargs) -> None:
        """Log progress within a stage."""
        percentage = (processed / total * 100) if total > 0 else 0
        
        self.logger.info(
            f"{stage_name}: {processed}/{total} ({percentage:.1f}%)",
            extra={
                'extra_fields': {
                    'stage': stage_name,
                    'processed': processed,
                    'total': total,
                    'percentage': percentage,
                    'event_type': 'stage_progress',
                    **kwargs
                }
            }
        )
    
    def log_data_quality(self, stage_name: str, quality_metrics: Dict[str, Any]) -> None:
        """Log data quality metrics for a stage."""
        self.logger.info(
            f"{stage_name} data quality: {quality_metrics}",
            extra={
                'extra_fields': {
                    'stage': stage_name,
                    'quality_metrics': quality_metrics,
                    'event_type': 'data_quality'
                }
            }
        )
    
    def log_performance(self, operation: str, duration: float, **metrics) -> None:
        """Log performance metrics for specific operations."""
        self.logger.info(
            f"{operation} completed in {duration:.2f}s",
            extra={
                'extra_fields': {
                    'operation': operation,
                    'duration': duration,
                    'event_type': 'performance',
                    **metrics
                }
            }
        )
    
    def log_error(self, stage: str, error: str, **context) -> None:
        """Log an error with context."""
        self.metrics.add_error(stage, error, **context)
        self.logger.error(
            f"{stage}: {error}",
            extra={
                'extra_fields': {
                    'stage': stage,
                    'event_type': 'error',
                    **context
                }
            }
        )
    
    def log_warning(self, stage: str, warning: str, **context) -> None:
        """Log a warning with context."""
        self.metrics.add_warning(stage, warning, **context)
        self.logger.warning(
            f"{stage}: {warning}",
            extra={
                'extra_fields': {
                    'stage': stage,
                    'event_type': 'warning',
                    **context
                }
            }
        )


def create_pipeline_logger(
    pipeline_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> PipelineLogger:
    """
    Create a configured pipeline logger.
    
    Args:
        pipeline_name: Name of the pipeline
        log_level: Logging level
        log_file: Optional specific log file
        
    Returns:
        Configured PipelineLogger instance
    """
    if log_file is None:
        log_file = f"logs/{pipeline_name}_pipeline.log"
    
    logger = PipelineLogger(pipeline_name, log_file)
    logger.logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


# Performance tracking decorator for pipeline functions
def track_performance(pipeline_logger: PipelineLogger, operation_name: str = None):
    """Decorator to track performance of pipeline operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                pipeline_logger.log_performance(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                pipeline_logger.log_error(op_name, str(e), duration=duration)
                raise
        
        return wrapper
    return decorator