"""Configuration management for the inference pipeline.

This module provides Pydantic-based configuration classes for all inference
pipeline components, with support for environment variables and YAML config files.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


class LLMConfig(BaseSettings):
    """Configuration for the LLM client.
    
    Attributes:
        provider: LLM provider name (only "openai" supported for MVP)
        model_name: Model identifier (e.g., "gpt-4o-mini", "gpt-4o")
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in the response
        api_key: API key for the provider (loaded from environment)
    """
    
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_LLM_",
        extra="ignore"
    )
    
    provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum response tokens")
    api_key: Optional[str] = Field(default=None, description="API key")
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate that provider is supported."""
        supported = ["openai"]
        if v.lower() not in supported:
            raise ValueError(f"Provider must be one of: {supported}")
        return v.lower()


class ConversationConfig(BaseSettings):
    """Configuration for conversation management.
    
    Attributes:
        max_history_length: Maximum number of messages to retain in history
    """
    
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_CONVERSATION_",
        extra="ignore"
    )
    
    max_history_length: int = Field(
        default=10,
        gt=0,
        description="Maximum messages in conversation history"
    )


class GeneratorConfig(BaseSettings):
    """Configuration for response generation.
    
    Attributes:
        system_prompt: Custom system prompt (uses default if None)
        max_context_tokens: Maximum tokens for context injection
    """
    
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_GENERATOR_",
        extra="ignore"
    )
    
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt"
    )
    max_context_tokens: int = Field(
        default=3000,
        gt=0,
        description="Maximum context tokens"
    )


class WorkflowConfig(BaseSettings):
    """Configuration for the agentic workflow.
    
    Attributes:
        product_keywords: Keywords that trigger retrieval routing
        tool_keywords: Keywords that trigger tool routing
    """
    
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_WORKFLOW_",
        extra="ignore"
    )
    
    product_keywords: List[str] = Field(
        default_factory=lambda: [
            "price", "review", "product", "recommend", "compare",
            "rating", "phone", "buy", "cost", "feature", "spec"
        ],
        description="Keywords for product-related queries"
    )
    tool_keywords: List[str] = Field(
        default_factory=lambda: ["compare"],
        description="Keywords for tool invocation"
    )


class InferenceSettings(BaseSettings):
    """Main configuration for the inference pipeline.
    
    This class aggregates all component configurations and provides
    environment-based configuration with sensible defaults.
    
    Attributes:
        openai_api_key: OpenAI API key (required)
        llm: LLM client configuration
        conversation: Conversation manager configuration
        generator: Response generator configuration
        workflow: Agentic workflow configuration
        enable_streaming: Whether to enable streaming responses
        max_retries: Maximum retry attempts for transient failures
        timeout_seconds: Timeout for inference operations
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    
    # Pipeline settings
    enable_streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Operation timeout in seconds"
    )
    
    def get_api_key(self) -> str:
        """Get the OpenAI API key, raising error if not configured.
        
        Returns:
            The OpenAI API key
            
        Raises:
            ConfigurationError: If API key is not configured
        """
        # Check instance attribute first
        if self.openai_api_key:
            return self.openai_api_key
        
        # Check LLM config
        if self.llm.api_key:
            return self.llm.api_key
        
        # Check environment variable directly
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key
        
        raise ConfigurationError(
            "OpenAI API key is required but not configured",
            missing_keys=["OPENAI_API_KEY"]
        )


# Global settings instance (lazy loaded)
_settings: Optional[InferenceSettings] = None


def get_inference_settings() -> InferenceSettings:
    """Get the inference settings instance.
    
    Returns:
        InferenceSettings instance
    """
    global _settings
    if _settings is None:
        _settings = InferenceSettings()
    return _settings


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    path = Path(config_path)
    
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            error_code="CONFIG_FILE_NOT_FOUND",
            details={"path": str(path.absolute())}
        )
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML configuration: {e}",
            error_code="CONFIG_PARSE_ERROR",
            details={"path": str(path.absolute()), "error": str(e)}
        )
    except IOError as e:
        raise ConfigurationError(
            f"Failed to read configuration file: {e}",
            error_code="CONFIG_READ_ERROR",
            details={"path": str(path.absolute()), "error": str(e)}
        )


def create_settings_from_yaml(
    config_path: Optional[str] = None
) -> InferenceSettings:
    """Create InferenceSettings from a YAML configuration file.
    
    This function loads configuration from a YAML file and merges it
    with environment variables. Environment variables take precedence.
    
    Args:
        config_path: Path to YAML config file. If None, uses default path.
        
    Returns:
        InferenceSettings instance with merged configuration
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Default config path
    if config_path is None:
        config_path = "config/inference.yaml"
    
    # Load YAML config if file exists
    yaml_config: Dict[str, Any] = {}
    if Path(config_path).exists():
        yaml_config = load_config_from_yaml(config_path)
    
    # Build nested config objects
    llm_config = LLMConfig(**yaml_config.get("llm", {}))
    conversation_config = ConversationConfig(**yaml_config.get("conversation", {}))
    generator_config = GeneratorConfig(**yaml_config.get("generator", {}))
    workflow_config = WorkflowConfig(**yaml_config.get("workflow", {}))
    
    # Create main settings with component configs
    settings_kwargs = {
        "llm": llm_config,
        "conversation": conversation_config,
        "generator": generator_config,
        "workflow": workflow_config,
    }
    
    # Add top-level settings from YAML
    for key in ["enable_streaming", "max_retries", "timeout_seconds"]:
        if key in yaml_config:
            settings_kwargs[key] = yaml_config[key]
    
    return InferenceSettings(**settings_kwargs)
