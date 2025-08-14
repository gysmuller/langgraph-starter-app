"""
Research Agent for the research multi-agent graph.

This agent executes the research plan using TavilySearch tool.
"""

import logging
import json
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from src.research_multi_agent.state import ResearchMultiAgentState, ResearchResult
from src.research_multi_agent.utils.agent_config import research_agent_config
from src.research_multi_agent.utils.prompts import get_researcher_system_prompt_async
from src.research_multi_agent.utils.tools import prepare_research_tools

logger = logging.getLogger(__name__)


async def call_research_agent(state: ResearchMultiAgentState) -> Dict[str, Any]:
    """
    Research agent node that executes the research plan.
    
    Input: state.research_plan (from planning agent)
    Output: Updates state.research_results with findings
    """
    try:
        # Handle both dict and state object formats
        research_plan = state.get("research_plan") if isinstance(state, dict) else state.research_plan
        research_results = state.get("research_results", []) if isinstance(state, dict) else state.research_results
        research_iterations = state.get("research_iterations", 0) if isinstance(state, dict) else state.research_iterations
        max_research_iterations = state.get("max_research_iterations", 5) if isinstance(state, dict) else state.max_research_iterations
        
        # Check if we have a research plan
        if not research_plan:
            raise ValueError("No research plan available for research execution")
        
        # Check iteration limits
        if research_iterations >= max_research_iterations:
            logger.warning(f"Maximum research iterations ({max_research_iterations}) reached")
            summary_message = AIMessage(
                content=f"Research completed. Maximum iterations ({max_research_iterations}) reached. "
                       f"Gathered {len(research_results)} research results.",
                name="researcher"
            )
            return {
                "messages": [summary_message],
                "research_iterations": research_iterations,
                "current_agent": "writer"
            }
        
        logger.info(f"Starting research iteration {research_iterations + 1}/{max_research_iterations}")
        
        # Get model and tools
        model = research_agent_config.get_chat_model()
        tools = await prepare_research_tools()
        
        if not tools:
            raise ValueError("No research tools available")
        
        # Create ReAct agent for research
        system_prompt = await get_researcher_system_prompt_async()
        research_agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=system_prompt
        )
        
        # Create research instructions based on the plan (pass a mock state object)
        mock_state = type('MockState', (), {
            'research_plan': research_plan,
            'research_results': research_results,
            'research_iterations': research_iterations,
            'max_research_iterations': max_research_iterations
        })()
        research_instructions = _create_research_instructions(mock_state)
        
        # Execute research using the ReAct agent
        research_response = await research_agent.ainvoke({
            "messages": [{"role": "user", "content": research_instructions}]
        })
        
        # Extract new research results from the agent's response
        new_results = _extract_research_results_from_response(research_response, research_plan.main_topic)
        
        # Combine with existing results
        all_results = list(research_results) + new_results
        
        # Create response message
        research_message = AIMessage(
            content=f"Research iteration {research_iterations + 1} completed. "
                   f"Found {len(new_results)} new results. "
                   f"Total results: {len(all_results)}",
            name="researcher"
        )
        
        # Determine if we should continue researching or move to writing
        mock_state.research_results = all_results
        should_continue = _should_continue_research(mock_state, all_results)
        next_agent = "researcher" if should_continue else "writer"
        
        if not should_continue:
            final_message = AIMessage(
                content=f"Research phase completed. Gathered {len(all_results)} total research results across "
                       f"{len(research_plan.subtopics)} subtopics. Ready for report writing.",
                name="researcher"
            )
            messages = [research_message, final_message]
        else:
            messages = [research_message]
        
        return {
            "messages": messages,
            "research_results": all_results,
            "research_iterations": research_iterations + 1,
            "current_agent": next_agent
        }
        
    except Exception as e:
        logger.error(f"Error in research agent: {e}")
        error_message = AIMessage(
            content=f"Error during research: {str(e)}",
            name="researcher"
        )
        
        return {
            "messages": [error_message],
            "research_results": research_results,  # Keep existing results
            "research_iterations": research_iterations + 1,
            "current_agent": "writer"  # Move to writer even on error
        }


def _create_research_instructions(state: ResearchMultiAgentState) -> str:
    """Create research instructions based on the current state and plan."""
    plan = state.research_plan
    existing_results_count = len(state.research_results)
    iteration = state.research_iterations + 1
    
    # Determine what to focus on this iteration
    if iteration == 1:
        # First iteration: focus on main topic and first few subtopics
        focus_areas = plan.subtopics[:3] if len(plan.subtopics) > 3 else plan.subtopics
        focus_questions = plan.research_questions[:5] if len(plan.research_questions) > 5 else plan.research_questions
    else:
        # Later iterations: focus on remaining subtopics and questions
        subtopics_per_iteration = max(2, len(plan.subtopics) // state.max_research_iterations)
        start_idx = (iteration - 1) * subtopics_per_iteration
        focus_areas = plan.subtopics[start_idx:start_idx + subtopics_per_iteration]
        
        questions_per_iteration = max(3, len(plan.research_questions) // state.max_research_iterations)
        start_idx = (iteration - 1) * questions_per_iteration
        focus_questions = plan.research_questions[start_idx:start_idx + questions_per_iteration]
    
    instructions = f"""
Research Plan Execution - Iteration {iteration}

Main Topic: {plan.main_topic}

Focus Areas for This Iteration:
{chr(10).join(f"- {area}" for area in focus_areas)}

Key Questions to Research:
{chr(10).join(f"- {question}" for question in focus_questions)}

Current Progress: {existing_results_count} research results already gathered.

Instructions:
1. Use the tavily_search tool to research each focus area systematically
2. Search for recent, authoritative information
3. Gather comprehensive information that answers the key questions
4. Focus on factual, well-documented information from credible sources
5. Note any conflicting viewpoints or perspectives

Please begin researching these focus areas now.
"""
    
    return instructions


def _extract_research_results_from_response(response: dict, main_topic: str) -> List[ResearchResult]:
    """Extract research results from the agent's response."""
    results = []
    
    try:
        # Get the messages from the response
        messages = response.get("messages", [])
        
        # Look for tool calls and results in the messages
        for message in messages:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # This is a message with tool calls
                for tool_call in message.tool_calls:
                    if tool_call.get('name') == 'tavily_search':
                        query = tool_call.get('args', {}).get('query', '')
                        if query:
                            # Find the corresponding tool response
                            result_content = _find_tool_result(messages, tool_call.get('id'))
                            if result_content:
                                results.append(_create_research_result(query, result_content))
            
            # Also check for function calls in content (alternative format)
            if hasattr(message, 'content') and 'tavily_search' in str(message.content):
                # Try to extract search information from content
                extracted_results = _extract_from_content(message.content, main_topic)
                results.extend(extracted_results)
    
    except Exception as e:
        logger.warning(f"Error extracting research results: {e}")
        # Fallback: create a basic result from the response
        if response and 'messages' in response:
            content = str(response['messages'][-1].content if response['messages'] else "Research completed")
            results.append(ResearchResult(
                query=f"Research on {main_topic}",
                content=content[:1000],  # Limit content length
                source="Generated from research agent",
                relevance_score=0.8
            ))
    
    return results


def _find_tool_result(messages: list, tool_call_id: str) -> str:
    """Find the tool result message for a given tool call ID."""
    for message in messages:
        if (hasattr(message, 'tool_call_id') and 
            message.tool_call_id == tool_call_id):
            return str(message.content)
    return ""


def _create_research_result(query: str, content: str) -> ResearchResult:
    """Create a ResearchResult from search query and content."""
    # Try to parse Tavily response format
    try:
        if isinstance(content, str):
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list) and data:
                    # Take the first result
                    first_result = data[0]
                    return ResearchResult(
                        query=query,
                        content=first_result.get('content', content)[:2000],
                        source=first_result.get('url', 'Unknown source'),
                        relevance_score=first_result.get('score', 0.8)
                    )
                elif isinstance(data, dict):
                    return ResearchResult(
                        query=query,
                        content=data.get('content', content)[:2000],
                        source=data.get('url', 'Unknown source'),
                        relevance_score=data.get('score', 0.8)
                    )
            except json.JSONDecodeError:
                pass
        
        # Fallback: use content as-is
        return ResearchResult(
            query=query,
            content=str(content)[:2000],
            source="Web search result",
            relevance_score=0.8
        )
    
    except Exception as e:
        logger.warning(f"Error creating research result: {e}")
        return ResearchResult(
            query=query,
            content=f"Research result for: {query}",
            source="Research agent",
            relevance_score=0.5
        )


def _extract_from_content(content: str, main_topic: str) -> List[ResearchResult]:
    """Extract research results from message content."""
    # This is a simple fallback extraction method
    # In production, you might want more sophisticated parsing
    
    results = []
    
    # Look for patterns that might indicate search results
    if len(content) > 100:  # Only process substantial content
        results.append(ResearchResult(
            query=f"Research on {main_topic}",
            content=content[:2000],
            source="Research agent analysis",
            relevance_score=0.7
        ))
    
    return results


def _should_continue_research(state: ResearchMultiAgentState, current_results: List[ResearchResult]) -> bool:
    """Determine if research should continue or move to writing."""
    
    # Don't continue if we've reached max iterations
    if state.research_iterations + 1 >= state.max_research_iterations:
        return False
    
    # Continue if we don't have enough results yet
    min_results_needed = len(state.research_plan.subtopics) * 2  # At least 2 results per subtopic
    if len(current_results) < min_results_needed:
        return True
    
    # Continue if we haven't covered all subtopics yet
    # This is a simple heuristic - in production you might want more sophisticated logic
    results_per_iteration = len(current_results) // (state.research_iterations + 1)
    expected_coverage = results_per_iteration * (state.research_iterations + 1)
    
    if expected_coverage < len(state.research_plan.subtopics):
        return True
    
    # Otherwise, we have sufficient coverage
    return False
