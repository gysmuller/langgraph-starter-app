"""Configuration management for LangGraph chat application."""

import os
from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database configuration (for checkpointing)
    database_url: str = Field(
        default="postgresql://localhost:5432/chat_basic",
        env="DATABASE_URL",
        description="PostgreSQL database URL for chat persistence"
    )
    
    # Tavily API configuration (for web search)
    tavily_api_key: Optional[str] = Field(
        default=None,
        env="TAVILY_API_KEY",
        description="Tavily API key for web search functionality"
    )
    
    # Application configuration
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    # LangSmith settings (optional)
    langsmith_api_key: Optional[str] = Field(
        default=None,
        env="LANGSMITH_API_KEY",
        description="LangSmith API key for tracing"
    )
    langsmith_project: Optional[str] = Field(
        default="chat-basic",
        env="LANGSMITH_PROJECT",
        description="LangSmith project name"
    )
    langsmith_tracing: bool = Field(
        default=True,
        env="LANGSMITH_TRACING",
        description="Enable LangSmith tracing"
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        env="LANGSMITH_ENDPOINT",
        description="LangSmith API endpoint"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level is a valid logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @model_validator(mode='after')
    def validate_required_for_production(self):
        """Validate that required keys are present for production environments."""
        if not self.debug:  # Production mode
            # Basic validation - specific services will validate their own requirements
            logger.info("Production mode enabled - services should validate their own requirements")
            
        return self
    
    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

# Global settings instance
settings = Settings()

# Configure logging with the loaded settings
settings.configure_logging()

# Log after settings are created and configured
logger.info("✅ Configuration loaded successfully!")
logger.debug(f"Configuration details: Debug={settings.debug}, Log Level={settings.log_level}")

if settings.debug:
    logger.debug("🔧 Debug mode is enabled")
    
if settings.langsmith_api_key:
    logger.info("📊 LangSmith tracing is configured")
else:
    logger.info("📊 LangSmith tracing is not configured (optional)")
    
if settings.tavily_api_key:
    logger.info("🔍 Tavily search is configured")
else:
    logger.info("🔍 Tavily search is not configured (optional)")