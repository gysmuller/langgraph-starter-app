"""Tools for the research multi-agent graph - TavilySearch with enhanced configuration"""

import logging
from typing import Any, List
from langgraph.runtime import get_runtime
from src.shared.config import settings
from src.research_multi_agent.utils.agent_config import research_agent_config
from src.research_multi_agent.utils.context_schema import ResearchMultiAgentContextSchema

# Try to import the new tavily package, fall back to community version
try:
    from langchain_tavily import TavilySearch as TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)


async def prepare_research_tools() -> List[Any]:
    """
    Prepare tools for the research agent with enhanced configuration.
    
    Returns:
        List containing the enhanced Tavily search tool for research
    """
    # Get runtime context
    try:
        runtime = get_runtime(ResearchMultiAgentContextSchema)
        context = runtime.context
        max_results = context.max_search_results_per_query if context.max_search_results_per_query is not None else research_agent_config.max_search_results
    except Exception:
        # Fallback to default if no context available (e.g., during initialization)
        max_results = research_agent_config.max_search_results
    
    # Check if Tavily API key is available
    tavily_api_key = settings.tavily_api_key or "dummy_key_for_testing"
    
    # Initialize Tavily Search Tool with research-specific configuration
    try:
        # Enhanced configuration for research use cases
        tavily_tool = TavilySearchResults(
            name="tavily_search",
            description=(
                "Search for comprehensive research information on any topic. "
                "Returns detailed results with content, sources, and AI-generated summaries. "
                "Use this tool to gather information for research plans and reports."
            ),
            max_results=max_results,  # More results for comprehensive research
            api_key=tavily_api_key,
            # Additional parameters for research
            include_raw_content=True,  # Get full content for detailed analysis
            include_answer=True,  # Get AI-generated answer for quick insights
        )
    except Exception as e:
        # If Tavily fails, create a mock tool for testing
        logger.warning(f"Failed to create Tavily research tool: {e}. Creating mock tool for testing.")
        from langchain_core.tools import Tool
        
        def mock_research_search(query: str) -> str:
            return f"""
Mock research result for: {query}

Summary: This is a mock research result since Tavily API key is not configured.

Content: Detailed information about {query} would appear here in a real search.

Sources:
- https://example.com/source1
- https://example.com/source2

Note: Configure TAVILY_API_KEY environment variable for real search results.
"""
        
        tavily_tool = Tool(
            name="tavily_search",
            description="Search for comprehensive research information (mock mode)",
            func=mock_research_search
        )
    
    return [tavily_tool]


async def prepare_planner_tools() -> List[Any]:
    """
    Prepare tools for the planning agent.
    Currently, the planner doesn't need external tools - it uses LLM reasoning only.
    
    Returns:
        Empty list - planner agent works with LLM reasoning only
    """
    return []


async def prepare_writer_tools() -> List[Any]:
    """
    Prepare tools for the report writer agent.
    Currently, the writer doesn't need external tools - it synthesizes from state.
    
    Returns:
        Empty list - writer agent works with LLM reasoning and state data only
    """
    return []
