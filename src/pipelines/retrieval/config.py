"""Configuration management for the retrieval pipeline."""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
from pydantic import Field, validator
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class RetrievalSettings(BaseSettings):
    """Environment-based configuration for retrieval pipeline."""
    
    # API Keys (from environment only)
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    
    # Pinecone Settings
    pinecone_index_name: str = Field(default="ecommerce-products", env="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(default="phone-reviews", env="PINECONE_NAMESPACE")
    pinecone_environment: str = Field(default="us-east-1-aws", env="PINECONE_ENVIRONMENT")
    
    # Embedding Settings
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dimension: int = Field(default=3072)
    
    # Search Settings
    top_k: int = Field(default=4, ge=1, le=50)
    fetch_k: int = Field(default=20, ge=1, le=100)
    lambda_mult: float = Field(default=0.7, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    search_type: str = Field(default="mmr")
    
    # Query Settings
    max_query_length: int = Field(default=512, ge=1, le=2048)
    normalize_unicode: bool = Field(default=True)
    
    # Compression Settings
    compression_enabled: bool = Field(default=True)
    relevance_prompt: Optional[str] = Field(default=None)
    
    # Rewriter Settings
    max_rewrite_attempts: int = Field(default=2, ge=0, le=5)
    rewrite_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    rewrite_prompt: Optional[str] = Field(default=None)
    
    # Formatter Settings
    format_template: Optional[str] = Field(default=None)
    format_delimiter: str = Field(default="\n\n---\n\n")
    include_scores: bool = Field(default=False)
    max_context_length: int = Field(default=4000, ge=100, le=10000)
    
    # Cache Settings
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=300, ge=0, le=3600)
    cache_max_size: int = Field(default=1000, ge=1, le=10000)
    
    # Retry Settings
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Evaluation Settings
    enable_evaluation: bool = Field(default=False)
    evaluation_metrics: List[str] = Field(default_factory=lambda: ["context_precision", "answer_relevancy"])
    evaluation_batch_size: int = Field(default=10, ge=1, le=100)
    
    # Metadata Extraction Settings
    metadata_extraction_enabled: bool = Field(default=True)
    metadata_extraction_model: str = Field(default="gpt-3.5-turbo")
    metadata_extraction_timeout: int = Field(default=3, ge=1, le=10)
    
    # Logging Settings
    log_level: str = Field(default="INFO")
    structured_logging: bool = Field(default=True)
    
    @validator("search_type")
    def validate_search_type(cls, v):
        """Validate search type is supported."""
        allowed_types = ["similarity", "mmr"]
        if v not in allowed_types:
            raise ValueError(f"search_type must be one of {allowed_types}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"log_level must be one of {allowed_levels}")
        return v.upper()
    
    @validator("evaluation_metrics")
    def validate_evaluation_metrics(cls, v):
        """Validate evaluation metrics are supported."""
        allowed_metrics = ["context_precision", "answer_relevancy", "faithfulness", "answer_correctness"]
        for metric in v:
            if metric not in allowed_metrics:
                raise ValueError(f"evaluation metric '{metric}' not supported. Allowed: {allowed_metrics}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file


class ConfigurationLoader:
    """Loads configuration from YAML files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.config_path = config_path or "config/retrieval.yaml"
    
    def load_config(self) -> RetrievalSettings:
        """Load configuration from YAML file and environment variables.
        
        Environment variables take precedence over YAML configuration.
        
        Returns:
            RetrievalSettings instance with loaded configuration
            
        Raises:
            ConfigurationError: If configuration validation fails
        """
        yaml_config = self._load_yaml_config()
        
        # Merge YAML config with environment variables
        # Environment variables take precedence
        merged_config = self._flatten_yaml_config(yaml_config)
        
        try:
            return RetrievalSettings(**merged_config)
        except Exception as e:
            missing_keys = self._extract_missing_keys(e)
            invalid_values = self._extract_invalid_values(e)
            raise ConfigurationError(
                f"Configuration validation failed: {str(e)}", 
                missing_keys=missing_keys,
                invalid_values=invalid_values
            )
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dictionary with configuration values from YAML file
            
        Raises:
            ConfigurationError: If YAML file is invalid
        """
        config_file = Path(self.config_path)
        if not config_file.exists():
            return {}  # Use defaults if no config file
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                return content or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML configuration in {self.config_path}: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file {self.config_path}: {str(e)}")
    
    def _flatten_yaml_config(self, yaml_config: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested YAML configuration to match RetrievalSettings field names.
        
        Args:
            yaml_config: Nested YAML configuration dictionary
            
        Returns:
            Flattened configuration dictionary with field names matching RetrievalSettings
        """
        flattened = {}
        
        # Extract metadata_extraction section
        if 'metadata_extraction' in yaml_config:
            metadata_section = yaml_config['metadata_extraction']
            if isinstance(metadata_section, dict):
                if 'enabled' in metadata_section:
                    flattened['metadata_extraction_enabled'] = metadata_section['enabled']
                if 'llm_model' in metadata_section:
                    flattened['metadata_extraction_model'] = metadata_section['llm_model']
                if 'timeout_seconds' in metadata_section:
                    flattened['metadata_extraction_timeout'] = metadata_section['timeout_seconds']
        
        # Keep other top-level fields as-is for backward compatibility
        for key, value in yaml_config.items():
            if key != 'metadata_extraction' and key not in flattened:
                flattened[key] = value
        
        return flattened
    
    def _extract_missing_keys(self, error: Exception) -> List[str]:
        """Extract missing required keys from validation error.
        
        Args:
            error: Pydantic validation error
            
        Returns:
            List of missing field names
        """
        missing_keys = []
        error_str = str(error)
        
        # Parse Pydantic validation error to extract missing fields
        if "field required" in error_str:
            # Extract field names from error message using regex
            matches = re.findall(r"(\w+)\s*\n.*field required", error_str, re.MULTILINE)
            missing_keys.extend(matches)
        
        return missing_keys
    
    def _extract_invalid_values(self, error: Exception) -> Dict[str, Any]:
        """Extract invalid values from validation error.
        
        Args:
            error: Pydantic validation error
            
        Returns:
            Dictionary of field names to error messages
        """
        invalid_values = {}
        error_str = str(error)
        
        # Parse validation errors for invalid values
        # This is a simplified parser - in production you might want to use
        # pydantic's error parsing utilities
        lines = error_str.split('\n')
        current_field = None
        
        for line in lines:
            # Look for field names
            field_match = re.match(r'^(\w+)', line.strip())
            if field_match and not line.strip().startswith('  '):
                current_field = field_match.group(1)
            
            # Look for validation errors
            if current_field and ('ensure this value' in line or 'invalid' in line.lower()):
                invalid_values[current_field] = line.strip()
        
        return invalid_values
    
    def validate_environment(self) -> None:
        """Validate that all required environment variables are set.
        
        Raises:
            ConfigurationError: If required environment variables are missing
        """
        required_env_vars = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            import os
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ConfigurationError(
                f"Missing required environment variables: {', '.join(missing_vars)}",
                missing_keys=missing_vars
            )
    
    def create_sample_config(self, output_path: Optional[str] = None) -> None:
        """Create a sample YAML configuration file.
        
        Args:
            output_path: Path where to create the sample config file
        """
        if output_path is None:
            output_path = self.config_path
        
        sample_config = {
            "# Retrieval Pipeline Configuration": None,
            "search": {
                "top_k": 4,
                "fetch_k": 20,
                "lambda_mult": 0.7,
                "score_threshold": 0.6,
                "search_type": "mmr"
            },
            "query": {
                "max_query_length": 512,
                "normalize_unicode": True
            },
            "embedding": {
                "model": "text-embedding-3-large",
                "dimension": 3072
            },
            "compression": {
                "enabled": True,
                "relevance_prompt": None
            },
            "rewriter": {
                "max_rewrite_attempts": 2,
                "rewrite_threshold": 0.5,
                "rewrite_prompt": None
            },
            "formatter": {
                "template": None,
                "delimiter": "\n\n---\n\n",
                "include_scores": False,
                "max_context_length": 4000
            },
            "cache": {
                "enabled": True,
                "ttl_seconds": 300,
                "max_size": 1000
            },
            "retry": {
                "max_retries": 3,
                "delay_seconds": 1.0
            },
            "evaluation": {
                "enabled": False,
                "metrics": ["context_precision", "answer_relevancy"],
                "batch_size": 10
            },
            "logging": {
                "level": "INFO",
                "structured": True
            }
        }
        
        # Create directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML file with comments
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Retrieval Pipeline Configuration\n")
            f.write("# Environment variables take precedence over these settings\n\n")
            yaml.dump(
                {k: v for k, v in sample_config.items() if not k.startswith("#")},
                f,
                default_flow_style=False,
                indent=2
            )