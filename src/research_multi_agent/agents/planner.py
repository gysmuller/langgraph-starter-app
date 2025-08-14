"""
Planning Agent for the research multi-agent graph.

This agent analyzes user input and creates a structured research plan.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from src.research_multi_agent.state import ResearchMultiAgentState, ResearchPlan
from src.research_multi_agent.utils.agent_config import research_agent_config
from src.research_multi_agent.utils.prompts import get_planner_system_prompt_async
from src.research_multi_agent.utils.tools import prepare_planner_tools

logger = logging.getLogger(__name__)


async def call_planner_agent(state: ResearchMultiAgentState) -> Dict[str, Any]:
    """
    Planning agent node that analyzes user input and creates a research plan.
    
    Input: state.messages (user query)
    Output: Updates state.research_plan with structured plan
    """
    try:
        # Handle both dict and state object formats
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        
        # Get the last user message
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user")]
        if not user_messages:
            raise ValueError("No user message found for research planning")
        
        # Extract content from message (handle both dict and object formats)
        last_message = user_messages[-1]
        if isinstance(last_message, dict):
            last_user_message = last_message.get("content", "")
        else:
            last_user_message = last_message.content
        logger.info(f"Planning research for query: {last_user_message}")
        
        # Get model and tools
        model = research_agent_config.get_chat_model()
        tools = await prepare_planner_tools()
        
        # Create the planning prompt
        system_prompt = await get_planner_system_prompt_async()
        
        planning_prompt = f"""
{system_prompt}

User Query: {last_user_message}

Please analyze this query and create a comprehensive research plan. Be specific and thorough.
Respond with a structured plan that includes:
1. Main topic identification
2. Key subtopics to research  
3. Specific research questions

Format your response as a clear, structured plan.
"""
        
        # Generate the research plan using the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User Query: {last_user_message}\n\nPlease create a comprehensive research plan for this query."}
        ]
        
        response = await model.ainvoke(messages)
        plan_content = response.content
        
        # Parse the response to extract structured plan components
        # For now, we'll create a simple parsing approach
        # In production, you might want to use structured output or more sophisticated parsing
        
        try:
            # Extract main topic, subtopics, and questions from the plan
            main_topic = _extract_main_topic(plan_content, last_user_message)
            subtopics = _extract_subtopics(plan_content)
            research_questions = _extract_research_questions(plan_content)
            
            # Create structured research plan
            research_plan = ResearchPlan(
                main_topic=main_topic,
                subtopics=subtopics,
                research_questions=research_questions
            )
            
            logger.info(f"Research plan created with {len(subtopics)} subtopics and {len(research_questions)} questions")
            
        except Exception as e:
            logger.warning(f"Failed to parse structured plan, using fallback approach: {e}")
            # Fallback: create a basic plan from the user query
            research_plan = ResearchPlan(
                main_topic=last_user_message,
                subtopics=[f"Research aspect of: {last_user_message}"],
                research_questions=[f"What information is available about: {last_user_message}?"]
            )
        
        # Add the plan response to messages
        plan_message = AIMessage(
            content=f"Research plan created:\n\n{plan_content}",
            name="planner"
        )
        
        return {
            "messages": [plan_message],
            "research_plan": research_plan,
            "current_agent": "researcher"
        }
        
    except Exception as e:
        logger.error(f"Error in planning agent: {e}")
        error_message = AIMessage(
            content=f"Error creating research plan: {str(e)}",
            name="planner"
        )
        
        # Create a minimal fallback plan
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, 'messages', [])
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                user_query = last_msg.get("content", "Unknown query")
            else:
                user_query = getattr(last_msg, 'content', "Unknown query")
        else:
            user_query = "Unknown query"
        fallback_plan = ResearchPlan(
            main_topic=user_query,
            subtopics=[f"General research on: {user_query}"],
            research_questions=[f"What information is available about: {user_query}?"]
        )
        
        return {
            "messages": [error_message],
            "research_plan": fallback_plan,
            "current_agent": "researcher"
        }


def _extract_main_topic(content: str, fallback: str) -> str:
    """Extract the main topic from the plan content."""
    lines = content.split('\n')
    
    # Look for lines that might contain the main topic
    for line in lines:
        if 'main topic' in line.lower() or 'topic:' in line.lower():
            # Extract the topic after the colon or indicator
            if ':' in line:
                topic = line.split(':', 1)[1].strip()
                if topic:
                    return topic
    
    # Fallback to user query
    return fallback


def _extract_subtopics(content: str) -> list[str]:
    """Extract subtopics from the plan content."""
    subtopics = []
    lines = content.split('\n')
    
    in_subtopics_section = False
    for line in lines:
        line = line.strip()
        
        # Check if we're entering the subtopics section
        if 'subtopic' in line.lower() and ':' in line:
            in_subtopics_section = True
            continue
        
        # Check if we've moved to a different section
        if in_subtopics_section and line and line.endswith(':') and 'question' in line.lower():
            break
        
        # Extract subtopic items
        if in_subtopics_section and line:
            # Clean up bullet points and numbering
            cleaned = line.lstrip('- •*123456789. ').strip()
            if cleaned and len(cleaned) > 3:  # Avoid very short/empty lines
                subtopics.append(cleaned)
    
    # Fallback if no subtopics found
    if not subtopics:
        subtopics = ["General research topic"]
    
    return subtopics[:7]  # Limit to reasonable number


def _extract_research_questions(content: str) -> list[str]:
    """Extract research questions from the plan content."""
    questions = []
    lines = content.split('\n')
    
    in_questions_section = False
    for line in lines:
        line = line.strip()
        
        # Check if we're entering the questions section
        if 'question' in line.lower() and ':' in line:
            in_questions_section = True
            continue
        
        # Check if we've moved to a different section
        if in_questions_section and line and line.endswith(':') and 'question' not in line.lower():
            break
        
        # Extract question items
        if in_questions_section and line:
            # Clean up bullet points and numbering
            cleaned = line.lstrip('- •*123456789. ').strip()
            if cleaned and len(cleaned) > 10:  # Questions should be reasonably long
                # Ensure it ends with a question mark
                if not cleaned.endswith('?'):
                    cleaned += '?'
                questions.append(cleaned)
    
    # Fallback if no questions found
    if not questions:
        questions = ["What are the key aspects of this topic?"]
    
    return questions[:10]  # Limit to reasonable number
