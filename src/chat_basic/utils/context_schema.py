"""Context schema for LangGraph runtime context."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatBasicContextSchema:
    """Runtime context schema for chat_basic agent."""
    # Search configuration
    max_search_results: Optional[int] = None
    search_depth: Optional[str] = None
