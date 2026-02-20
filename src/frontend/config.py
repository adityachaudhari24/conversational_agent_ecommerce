"""Frontend configuration."""

from pydantic_settings import BaseSettings


class FrontendSettings(BaseSettings):
    """Configuration for the Streamlit frontend."""
    
    # API settings
    api_base_url: str = "http://localhost:8009"
    
    # Chat UI settings
    chat_width: int = 400
    max_input_length: int = 500
    
    # Timeouts
    request_timeout: int = 60
    
    class Config:
        env_prefix = "FRONTEND_"
        env_file = ".env"
        extra = "ignore"


settings = FrontendSettings()
