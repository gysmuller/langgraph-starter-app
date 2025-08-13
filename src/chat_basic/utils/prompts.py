"""Simple prompt utilities for chat_basic agent."""

from pathlib import Path
from typing import List
from langchain_core.messages import AnyMessage
from langgraph.runtime import get_runtime
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.chat_basic.utils.agent_config import agent_config
from src.chat_basic.utils.context_schema import ChatBasicContextSchema

# Load system prompt template at module import to avoid blocking
prompt_file = Path(__file__).parent / "prompts" / "chat_basic_system_prompt.txt"
with open(prompt_file, 'r', encoding='utf-8') as f:
    _SYSTEM_PROMPT_TEMPLATE = f.read()


def get_system_prompt(state: AgentState) -> List[AnyMessage]:
    """Get system prompt with context values and return messages."""
    # Get context values or use defaults
    try:
        runtime = get_runtime(ChatBasicContextSchema)
        context = runtime.context
        max_results = context.max_search_results or agent_config.max_search_results
        depth = context.search_depth or agent_config.search_depth  
    except Exception:
        max_results = agent_config.max_search_results
        depth = agent_config.search_depth
    
    # Format prompt with values
    content = _SYSTEM_PROMPT_TEMPLATE.format(
        max_search_results=max_results,
        search_depth=depth
    )
    
    return [{"role": "system", "content": content}] + state["messages"]
