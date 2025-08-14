"""Agent-specific configuration for research multi-agent graph."""

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


class ResearchAgentConfig(BaseSettings):
    """Configuration specific to the research multi-agent graph."""
    
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
    
    # Research-specific settings
    max_research_iterations: int = Field(
        default=5,
        env="MAX_RESEARCH_ITERATIONS",
        description="Maximum number of research iterations per topic"
    )
    max_search_results: int = Field(
        default=5,
        env="MAX_SEARCH_RESULTS",
        description="Maximum number of search results per query"
    )
    research_depth: str = Field(
        default="comprehensive",
        env="RESEARCH_DEPTH",
        description="Research depth: 'quick', 'standard', or 'comprehensive'"
    )
    
    # Report generation settings
    output_format: str = Field(
        default="markdown",
        env="OUTPUT_FORMAT",
        description="Output format for reports: 'markdown' or 'plain_text'"
    )
    include_sources: bool = Field(
        default=True,
        env="INCLUDE_SOURCES",
        description="Whether to include source citations in reports"
    )
    
    @field_validator('research_depth')
    @classmethod
    def validate_research_depth(cls, v):
        """Validate research depth is valid."""
        valid_depths = ['quick', 'standard', 'comprehensive']
        if v not in valid_depths:
            raise ValueError(f'research_depth must be one of {valid_depths}')
        return v
    
    @field_validator('output_format')
    @classmethod  
    def validate_output_format(cls, v):
        """Validate output format is valid."""
        valid_formats = ['markdown', 'plain_text']
        if v not in valid_formats:
            raise ValueError(f'output_format must be one of {valid_formats}')
        return v
    
    def validate_llm_config(self) -> None:
        """Validate LLM configuration is sufficient."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM functionality")
    
    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        """Get a configured ChatOpenAI model instance."""
        self.validate_llm_config()
        
        # Set defaults that can be overridden by kwargs
        config = {
            'model': kwargs.pop('model', self.openai_model),
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


# Create research agent configuration instance
research_agent_config = ResearchAgentConfig()

# Log configuration status
logger.info(f"✅ Research agent configuration loaded! Model: {research_agent_config.openai_model}")

if research_agent_config.openai_api_key:
    logger.info("🤖 OpenAI LLM is configured for research")
else:
    logger.warning("⚠️  OpenAI LLM is not configured - research agents will fail without API key")
