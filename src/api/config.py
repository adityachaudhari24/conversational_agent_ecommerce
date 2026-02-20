"""API configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from typing import List


class APISettings(BaseSettings):
    """Configuration for the FastAPI backend."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8009
    debug: bool = False
    
    # CORS settings
    cors_origins: List[str] = ["*"]
    
    # Session storage
    session_dir: str = "data/sessions"
    
    # Optional API key authentication (empty = disabled)
    api_key: str = ""
    
    # Timeouts
    request_timeout: int = 60
    
    class Config:
        env_prefix = "API_"
        env_file = ".env"
        extra = "ignore"


# Global settings instance
settings = APISettings()
