"""Tools for the chat_basic graph - Tavily Search only"""

import logging
from typing import Any, List
from langgraph.runtime import get_runtime
from src.shared.config import settings
from src.chat_basic.utils.agent_config import agent_config
from src.chat_basic.utils.context_schema import ChatBasicContextSchema

# Try to import the new tavily package, fall back to community version
try:
    from langchain_tavily import TavilySearch as TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

async def prepare_basic_tools() -> List[Any]:
    """
    Prepare tools for the chat_basic agent - only Tavily Search.
    
    Returns:
        List containing only the Tavily search tool
    """
    # Get runtime context
    try:
        runtime = get_runtime(ChatBasicContextSchema)
        context = runtime.context
        max_results = context.max_search_results if context.max_search_results is not None else agent_config.max_search_results
    except Exception:
        # Fallback to default if no context available (e.g., during initialization)
        max_results = agent_config.max_search_results
    
    # Check if Tavily API key is available
    tavily_api_key = settings.tavily_api_key or "dummy_key_for_testing"
    
    # Initialize Tavily Search Tool with configuration
    try:
        # Try with the new TavilySearch API first
        tavily_tool = TavilySearchResults(
            max_results=max_results,
            api_key=tavily_api_key
        )
    except Exception as e:
        # If Tavily fails, create a mock tool for testing
        logger.warning(f"Failed to create Tavily tool: {e}. Creating mock tool for testing.")
        from langchain_core.tools import Tool
        
        def mock_search(query: str) -> str:
            return f"Mock search result for: {query} (Tavily API key not configured)"
        
        tavily_tool = Tool(
            name="tavily_search_results_json",
            description="Search the web for current information",
            func=mock_search
        )
    
    return [tavily_tool] 