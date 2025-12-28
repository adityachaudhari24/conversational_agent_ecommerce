"""Configuration management using Pydantic Settings."""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/ecommerce_rag",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max connection overflow")


class LLMSettings(BaseSettings):
    """LLM provider configuration settings."""
    
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    default_model: str = Field(default="gpt-3.5-turbo", description="Default LLM model")
    max_tokens: int = Field(default=1000, description="Maximum tokens per response")
    temperature: float = Field(default=0.7, description="LLM temperature setting")


class VectorDBSettings(BaseSettings):
    """Vector database configuration settings."""
    
    provider: str = Field(default="chromadb", description="Vector DB provider")
    collection_name: str = Field(default="ecommerce_products", description="Collection name")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    chunk_size: int = Field(default=500, description="Text chunk size")
    chunk_overlap: int = Field(default=50, description="Text chunk overlap")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    max_results: int = Field(default=10, description="Maximum search results")


class APISettings(BaseSettings):
    """API server configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of worker processes")
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"],
        description="CORS allowed origins"
    )


class StreamlitSettings(BaseSettings):
    """Streamlit frontend configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="Streamlit host")
    port: int = Field(default=8501, description="Streamlit port")
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="FastAPI backend URL"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(default=10485760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup log files")


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=True, description="Debug mode")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    api: APISettings = Field(default_factory=APISettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    api_key: Optional[str] = Field(default=None, description="API authentication key")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() in ("development", "dev", "local")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() in ("production", "prod")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings