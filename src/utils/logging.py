"""Logging configuration and utilities."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

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


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    use_rich: bool = True
) -> None:
    """Set up application logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        use_rich: Whether to use Rich handler for console output
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
        file_handler.setFormatter(
            logging.Formatter(settings.logging.format)
        )
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging on import
setup_logging()