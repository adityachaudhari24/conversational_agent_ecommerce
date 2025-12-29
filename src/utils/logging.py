"""Logging configuration and utilities."""

import json
import logging
import logging.handlers
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Dict, Any, Callable

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

console = Console()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.RESET}"
            )
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records."""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        if self.context:
            if not hasattr(record, 'extra_fields'):
                record.extra_fields = {}
            record.extra_fields.update(self.context)
        return True


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    use_rich: bool = True,
    use_json: bool = False,
    context: Dict[str, Any] = None
) -> None:
    """Set up application logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        use_rich: Whether to use Rich handler for console output
        use_json: Whether to use JSON formatting for file output
        context: Additional context to add to all log records
    """
    settings = get_settings()
    
    # Determine log level
    if level is None:
        level = settings.logging.level
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        root_logger.addFilter(context_filter)
    
    # Console handler
    if use_rich and settings.is_development:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setFormatter(
            logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]"
            )
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            ColoredFormatter(settings.logging.format)
        )
    
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or settings.logging.file_path:
        file_path = log_file or settings.logging.file_path
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        
        # Use JSON formatter for production or if explicitly requested
        if use_json or settings.is_production:
            file_handler.setFormatter(
                JSONFormatter(datefmt='%Y-%m-%d %H:%M:%S')
            )
        else:
            file_handler.setFormatter(
                logging.Formatter(settings.logging.format)
            )
        
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    third_party_loggers = [
        "httpx", "httpcore", "urllib3", "chromadb", 
        "sentence_transformers", "openai", "pinecone",
        "langchain", "langchain_core", "langchain_openai"
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str, context: Dict[str, Any] = None) -> logging.Logger:
    """Get a logger instance with the given name and optional context.
    
    Args:
        name: Logger name (usually __name__)
        context: Additional context to add to all log records from this logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        context_filter = ContextFilter(context)
        logger.addFilter(context_filter)
    
    return logger


def log_performance(logger: logging.Logger = None) -> Callable:
    """Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use (defaults to function's module logger)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            start_time = time.time()
            
            try:
                func_logger.debug(f"Starting {func.__name__}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                func_logger.info(f"{func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                func_logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator


def log_method_calls(logger: logging.Logger = None) -> Callable:
    """Decorator to log method entry and exit with parameters.
    
    Args:
        logger: Logger instance to use (defaults to class module logger)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            
            # Log method entry
            func_logger.debug(f"Entering {func.__name__} with args={args[1:]} kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Exiting {func.__name__} successfully")
                return result
            except Exception as e:
                func_logger.debug(f"Exiting {func.__name__} with exception: {e}")
                raise
        
        return wrapper
    return decorator


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}",
                context={'class': self.__class__.__name__}
            )
        return self._logger


def setup_pipeline_logging(
    pipeline_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_json: bool = False
) -> logging.Logger:
    """Set up logging specifically for pipeline execution.
    
    Args:
        pipeline_name: Name of the pipeline (e.g., 'ingestion', 'retrieval')
        log_level: Logging level
        log_file: Optional log file path (defaults to logs/{pipeline_name}.log)
        use_json: Whether to use JSON formatting
        
    Returns:
        Logger instance for the pipeline
    """
    if log_file is None:
        log_file = f"logs/{pipeline_name}.log"
    
    # Set up logging with pipeline context
    context = {
        'pipeline': pipeline_name,
        'component': 'pipeline'
    }
    
    setup_logging(
        level=log_level,
        log_file=log_file,
        use_json=use_json,
        context=context
    )
    
    return get_logger(f"pipelines.{pipeline_name}", context)


# Initialize logging on import
setup_logging()