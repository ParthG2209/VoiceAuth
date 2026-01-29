"""
Configuration management for VoiceAuth API
Uses pydantic-settings for environment variable handling
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # API Security
    api_key: str = "sk_voiceauth_dev_key_12345"
    api_key_header: str = "x-api-key"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Model Configuration
    model_cache_dir: str = "./models"
    use_gpu: bool = False
    
    # Rate Limiting
    rate_limit: int = 60  # requests per minute
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # Supported Languages
    supported_languages: list[str] = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    
    # Audio Constraints
    max_audio_size_mb: int = 10
    max_audio_duration_seconds: int = 60


@lru_cache()
def get_settings() -> Settings:
    """Cache and return settings singleton"""
    return Settings()


# Export settings instance
settings = get_settings()
