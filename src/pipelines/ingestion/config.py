"""
Configuration management for the data ingestion pipeline.

Provides Pydantic-based configuration schemas with environment variable
loading and YAML config file support.
"""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class IngestionSettings(BaseSettings):
    """
    Environment-based configuration for the ingestion pipeline.
    
    Loads configuration from environment variables with sensible defaults.
    Validates required settings and provides clear error messages for
    missing or invalid configuration.
    """
    
    # API Keys (Required)
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    
    # Pinecone Settings
    pinecone_index_name: str = Field(default="ecommerce-products", env="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(default="phone-reviews", env="PINECONE_NAMESPACE")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")
    pinecone_cloud: str = Field(default="aws", env="PINECONE_CLOUD")
    
    # Embedding Settings
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dimension: int = Field(default=3072)
    
    # Chunking Settings
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    # Pipeline Settings
    batch_size: int = Field(default=100)
    abort_threshold: float = Field(default=0.5)
    
    # File Paths
    data_file_path: str = Field(default="data/phones_reviews.csv")
    log_dir: str = Field(default="logs")
    config_file_path: Optional[str] = Field(default=None, env="CONFIG_FILE_PATH")
    
    # Processing Settings
    required_columns: List[str] = Field(default_factory=lambda: [
        "product_name", "description", "price", 
        "rating", "review_title", "review_text"
    ])
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from environment
    )
    
    @field_validator('abort_threshold')
    @classmethod
    def validate_abort_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("abort_threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v):
        if v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v
    
    @field_validator('embedding_dimension')
    @classmethod
    def validate_embedding_dimension(cls, v):
        if v <= 0:
            raise ValueError("embedding_dimension must be positive")
        return v


def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        ConfigurationError: If file doesn't exist or contains invalid YAML
    """
    config_path = Path(file_path)
    
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        if config_data is None:
            return {}
            
        return config_data
        
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in configuration file {file_path}: {e}"
        )
    except Exception as e:
        raise ConfigurationError(
            f"Error reading configuration file {file_path}: {e}"
        )


def create_ingestion_settings(config_file_path: Optional[str] = None) -> IngestionSettings:
    """
    Create IngestionSettings with optional YAML config file override.
    
    Args:
        config_file_path: Optional path to YAML configuration file
        
    Returns:
        Configured IngestionSettings instance
        
    Raises:
        ConfigurationError: If required environment variables are missing
                           or configuration is invalid
    """
    # Load YAML config if provided
    yaml_config = {}
    if config_file_path:
        yaml_config = load_config_from_yaml(config_file_path)
    
    try:
        # Create settings with YAML overrides
        settings = IngestionSettings(**yaml_config)
        return settings
        
    except Exception as e:
        # Check for missing required environment variables
        missing_vars = []
        required_env_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}",
                missing_keys=missing_vars
            )
        
        # Re-raise original error if not missing env vars
        raise ConfigurationError(f"Configuration error: {e}")


def validate_configuration(settings: IngestionSettings) -> None:
    """
    Perform additional validation on configuration settings.
    
    Args:
        settings: IngestionSettings instance to validate
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Validate file paths exist
    data_path = Path(settings.data_file_path)
    if not data_path.parent.exists():
        raise ConfigurationError(
            f"Data directory does not exist: {data_path.parent}"
        )
    
    # Validate log directory can be created
    log_path = Path(settings.log_dir)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ConfigurationError(
            f"Cannot create log directory {settings.log_dir}: {e}"
        )
    
    # Validate chunk settings
    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigurationError(
            "chunk_overlap must be less than chunk_size"
        )