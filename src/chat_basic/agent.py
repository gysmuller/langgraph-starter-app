"""
Basic Finance Chat Agent

This module provides a StateGraph-based chat agent with simplified tool access (Tavily Search only).
"""

from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig

from src.chat_basic.utils.tools import prepare_basic_tools
from src.chat_basic.utils.agent_config import agent_config
from src.chat_basic.utils.prompts import get_system_prompt
from src.chat_basic.utils.context_schema import ChatBasicContextSchema

async def create_chat_basic_graph(config: RunnableConfig = None):
    """
    Create a StateGraph-based basic chat agent with Tavily Search only.
    
    Args:
        config: Optional RunnableConfig containing checkpointer and interrupt_before
        
    Returns:
        A compiled StateGraph for basic chatting with web search capability
    """
    # Extract checkpointer and interrupt_before from config if provided
    checkpointer = getattr(config, 'checkpointer', None) if config else None
    interrupt_before = getattr(config, 'interrupt_before', None) if config else None
    
    try:
        # Get model with error handling
        model = agent_config.get_chat_model()
        if model is None:
            raise ValueError("Failed to initialize chat model")
            
        # Prepare tools
        tools = await prepare_basic_tools()
        if not tools:
            print("[WARNING] No tools prepared - agent will run without external tools")
        
        # Get the system prompt function
        system_prompt_func = get_system_prompt
        
        # Create agent using prebuilt create_react_agent
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt_func,
            state_schema=AgentState,
            context_schema=ChatBasicContextSchema,
            checkpointer=checkpointer,
            interrupt_before=interrupt_before
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to create agent: {e}")
        raise
    
    # If checkpointer is provided and has setup method, call it
    if checkpointer is not None:
        try:
            if hasattr(checkpointer, 'setup'):
                await checkpointer.setup()
            if interrupt_before:
                print(f"[INFO] Chat agent created with checkpointer and interrupts: {interrupt_before}")
            else:
                print("[INFO] Chat agent created with checkpointer")
        except Exception as e:
            print(f"[WARNING] Checkpointer setup failed: {e}")
            print("[INFO] Agent will continue without persistent state")
    else:
        print("[INFO] Chat agent created without checkpointer")
    
    return agent 