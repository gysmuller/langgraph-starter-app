"""
Main graph definition for the research multi-agent system.

This module orchestrates the three-agent research workflow:
1. Planning Agent - Creates research plans
2. Research Agent - Executes research using TavilySearch  
3. Report Writer Agent - Synthesizes findings into reports
"""

import logging
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from src.research_multi_agent.state import ResearchMultiAgentState
from src.research_multi_agent.agents.planner import call_planner_agent
from src.research_multi_agent.agents.researcher import call_research_agent
from src.research_multi_agent.agents.writer import call_writer_agent

logger = logging.getLogger(__name__)


async def create_research_multi_agent_graph(config: RunnableConfig = None):
    """
    Create the multi-agent research graph.
    
    Args:
        config: Optional RunnableConfig containing checkpointer and interrupt_before
        
    Returns:
        A compiled StateGraph for the research workflow
    """
    # Extract checkpointer and interrupt_before from config if provided
    checkpointer = getattr(config, 'checkpointer', None) if config else None
    interrupt_before = getattr(config, 'interrupt_before', None) if config else None
    
    try:
        logger.info("Creating research multi-agent graph...")
        
        # Initialize the StateGraph with our custom state
        builder = StateGraph(ResearchMultiAgentState)
        
        # Add agent nodes
        builder.add_node("planner", call_planner_agent)
        builder.add_node("researcher", call_research_agent) 
        builder.add_node("writer", call_writer_agent)
        
        # Define the deterministic flow
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "researcher")
        builder.add_edge("researcher", "writer")
        builder.add_edge("writer", END)
        
        # Compile the graph with optional checkpointer
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before
        )
        
        logger.info("Research multi-agent graph created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create research multi-agent graph: {e}")
        raise
    
    # Handle checkpointer setup if provided
    if checkpointer is not None:
        try:
            if hasattr(checkpointer, 'setup'):
                await checkpointer.setup()
            if interrupt_before:
                logger.info(f"Research multi-agent graph created with checkpointer and interrupts: {interrupt_before}")
            else:
                logger.info("Research multi-agent graph created with checkpointer")
        except Exception as e:
            logger.warning(f"Checkpointer setup failed: {e}")
            logger.info("Graph will continue without persistent state")
    else:
        logger.info("Research multi-agent graph created without checkpointer")
    
    return graph


# Alternative flow with conditional research iterations
async def create_research_multi_agent_graph_with_loops(config: RunnableConfig = None):
    """
    Create the research multi-agent graph with conditional loops for research iterations.
    
    This version allows the research agent to loop back to itself for multiple iterations
    before proceeding to the writer.
    
    Args:
        config: Optional RunnableConfig containing checkpointer and interrupt_before
        
    Returns:
        A compiled StateGraph for the research workflow with loops
    """
    # Extract checkpointer and interrupt_before from config if provided
    checkpointer = getattr(config, 'checkpointer', None) if config else None
    interrupt_before = getattr(config, 'interrupt_before', None) if config else None
    
    try:
        logger.info("Creating research multi-agent graph with conditional loops...")
        
        # Initialize the StateGraph with our custom state
        builder = StateGraph(ResearchMultiAgentState)
        
        # Add agent nodes
        builder.add_node("planner", call_planner_agent)
        builder.add_node("researcher", call_research_agent) 
        builder.add_node("writer", call_writer_agent)
        
        # Define the flow with conditional logic
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "researcher")
        
        # Add conditional edge for research loops
        builder.add_conditional_edges(
            "researcher",
            _should_continue_research_decision,
            {
                "continue": "researcher",  # Loop back to researcher
                "write": "writer"         # Proceed to writer
            }
        )
        
        builder.add_edge("writer", END)
        
        # Compile the graph with optional checkpointer
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before
        )
        
        logger.info("Research multi-agent graph with loops created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create research multi-agent graph with loops: {e}")
        raise
    
    # Handle checkpointer setup if provided
    if checkpointer is not None:
        try:
            if hasattr(checkpointer, 'setup'):
                await checkpointer.setup()
            if interrupt_before:
                logger.info(f"Research multi-agent graph with loops created with checkpointer and interrupts: {interrupt_before}")
            else:
                logger.info("Research multi-agent graph with loops created with checkpointer")
        except Exception as e:
            logger.warning(f"Checkpointer setup failed: {e}")
            logger.info("Graph will continue without persistent state")
    else:
        logger.info("Research multi-agent graph with loops created without checkpointer")
    
    return graph


def _should_continue_research_decision(state: ResearchMultiAgentState) -> str:
    """
    Decision function for conditional research loops.
    
    Args:
        state: Current state of the research workflow
        
    Returns:
        "continue" to loop back to researcher, "write" to proceed to writer
    """
    # Check if we've reached maximum iterations
    if state.research_iterations >= state.max_research_iterations:
        logger.info(f"Maximum research iterations ({state.max_research_iterations}) reached, proceeding to writer")
        return "write"
    
    # Check if we have sufficient research results
    if not state.research_plan:
        logger.warning("No research plan available, proceeding to writer")
        return "write"
    
    # Simple heuristic: continue if we don't have enough results per subtopic
    min_results_per_subtopic = 2
    required_results = len(state.research_plan.subtopics) * min_results_per_subtopic
    current_results = len(state.research_results)
    
    if current_results < required_results:
        logger.info(f"Need more research: {current_results}/{required_results} results, continuing research")
        return "continue"
    
    # Check the current agent field from state (set by researcher)
    if hasattr(state, 'current_agent') and state.current_agent == "writer":
        logger.info("Research agent indicated completion, proceeding to writer")
        return "write"
    
    # Default: continue researching if we haven't hit limits
    logger.info("Continuing research for more comprehensive coverage")
    return "continue"


# Example usage functions for deployment
async def create_research_graph_with_persistence(interrupt_before: list = None):
    """
    Create research graph with PostgreSQL checkpointing for production use.
    
    Args:
        interrupt_before: Optional list of nodes to interrupt before (e.g., ["researcher"])
        
    Returns:
        Configured research graph with persistence
    """
    from src.shared.checkpoint import get_postgres_checkpointer
    
    async with get_postgres_checkpointer() as checkpointer:
        config = RunnableConfig(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before
        )
        graph = await create_research_multi_agent_graph(config)
        return graph


async def create_research_graph_simple():
    """
    Create research graph without persistence for development/testing.
    
    Returns:
        Simple research graph without checkpointing
    """
    return await create_research_multi_agent_graph()
