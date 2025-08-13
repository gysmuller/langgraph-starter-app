"""Agent-specific configuration for chat_basic agent."""

import os
import logging
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AgentConfig(BaseSettings):
    """Configuration specific to the chat_basic agent."""
    
    # OpenAI LLM configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY",
        description="OpenAI API key for LLM access"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        env="OPENAI_MODEL", 
        description="Default OpenAI model to use"
    )
    chat_model: Optional[str] = Field(
        default=None,
        env="CHAT_MODEL",
        description="Chat-specific model override (defaults to openai_model)"
    )
    
    # Chat agent specific settings
    max_search_results: int = Field(
        default=5,
        env="MAX_SEARCH_RESULTS",
        description="Maximum number of search results to return"
    )
    search_depth: str = Field(
        default="basic",
        env="SEARCH_DEPTH",
        description="Search depth for web searches (basic or advanced)"
    )
    
    @field_validator('search_depth')
    @classmethod
    def validate_search_depth(cls, v):
        """Validate search depth is valid."""
        valid_depths = ['basic', 'advanced']
        if v not in valid_depths:
            raise ValueError(f'search_depth must be one of {valid_depths}')
        return v
    
    @property
    def effective_chat_model(self) -> str:
        """Get the effective chat model (chat_model if set, otherwise openai_model)."""
        return self.chat_model or self.openai_model
    
    def validate_llm_config(self) -> None:
        """Validate LLM configuration is sufficient."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM functionality")
    
    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        """Get a configured ChatOpenAI model instance."""
        self.validate_llm_config()
        
        # Set defaults that can be overridden by kwargs
        config = {
            'model': kwargs.pop('model', self.effective_chat_model),
            'api_key': self.openai_api_key,
        }
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        return ChatOpenAI(**config)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Create agent-specific configuration instance
agent_config = AgentConfig()

# Log configuration status
logger.info(f"✅ Agent configuration loaded! Model: {agent_config.effective_chat_model}")

if agent_config.openai_api_key:
    logger.info("🤖 OpenAI LLM is configured")
else:
    logger.warning("⚠️  OpenAI LLM is not configured - agent will fail without API key")
