"""
Report Writer Agent for the research multi-agent graph.

This agent synthesizes research findings into a comprehensive report.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from src.research_multi_agent.state import ResearchMultiAgentState
from src.research_multi_agent.utils.agent_config import research_agent_config
from src.research_multi_agent.utils.prompts import get_writer_system_prompt_async
from src.research_multi_agent.utils.tools import prepare_writer_tools

logger = logging.getLogger(__name__)


async def call_writer_agent(state: ResearchMultiAgentState) -> Dict[str, Any]:
    """
    Report writer agent node that creates the final report.
    
    Input: state.messages (original query) + state.research_results
    Output: Updates state.final_report with synthesized report
    """
    try:
        # Handle both dict and state object formats
        messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        research_plan = state.get("research_plan") if isinstance(state, dict) else state.research_plan
        research_results = state.get("research_results", []) if isinstance(state, dict) else state.research_results
        
        # Get the original user query
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user")]
        if not user_messages:
            raise ValueError("No user query found for report writing")
        
        # Extract content from message (handle both dict and object formats)
        first_message = user_messages[0]
        if isinstance(first_message, dict):
            original_query = first_message.get("content", "")
        else:
            original_query = first_message.content
        
        # Check if we have research data
        if not research_plan:
            raise ValueError("No research plan available for report writing")
        
        if not research_results:
            logger.warning("No research results available, creating report with available information")
        
        logger.info(f"Writing report for query: {original_query}")
        logger.info(f"Using {len(research_results)} research results")
        
        # Get model and tools
        model = research_agent_config.get_chat_model()
        tools = await prepare_writer_tools()
        
        # Create the report writing prompt
        system_prompt = await get_writer_system_prompt_async()
        
        # Create a mock state object for helper functions
        mock_state = type('MockState', (), {
            'research_plan': research_plan,
            'research_results': research_results,
            'messages': messages,
            'research_iterations': state.get("research_iterations", 1) if isinstance(state, dict) else getattr(state, 'research_iterations', 1)
        })()
        
        # Prepare research summary for the writer
        research_summary = _prepare_research_summary(mock_state)
        
        writing_prompt = f"""
{system_prompt}

Original User Query: {original_query}

Research Plan Summary:
- Main Topic: {research_plan.main_topic}
- Subtopics: {', '.join(research_plan.subtopics)}
- Research Questions: {len(research_plan.research_questions)} questions investigated

Research Findings:
{research_summary}

Please create a comprehensive report that addresses the user's original query. Structure the report professionally with clear sections, include all relevant findings, and provide proper source citations.
"""
        
        # Generate the report using the LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": writing_prompt}
        ]
        
        response = await model.ainvoke(messages)
        final_report = response.content
        
        # Enhance the report with additional formatting
        enhanced_report = _enhance_report_formatting(final_report, mock_state)
        
        logger.info(f"Report generated successfully ({len(enhanced_report)} characters)")
        
        # Create response message
        report_message = AIMessage(
            content=f"Research report completed for: {original_query}\n\n{enhanced_report}",
            name="writer"
        )
        
        return {
            "messages": [report_message],
            "final_report": enhanced_report,
            "current_agent": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error in report writer agent: {e}")
        
        # Create a fallback state object for error case
        try:
            messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, 'messages', [])
            research_plan = state.get("research_plan") if isinstance(state, dict) else getattr(state, 'research_plan', None)
            research_results = state.get("research_results", []) if isinstance(state, dict) else getattr(state, 'research_results', [])
            
            fallback_state = type('FallbackState', (), {
                'research_plan': research_plan,
                'research_results': research_results,
                'messages': messages,
                'research_iterations': state.get("research_iterations", 1) if isinstance(state, dict) else getattr(state, 'research_iterations', 1)
            })()
        except:
            # Ultimate fallback
            fallback_state = type('FallbackState', (), {
                'research_plan': None,
                'research_results': [],
                'messages': [],
                'research_iterations': 1
            })()
        
        # Create a fallback report
        fallback_report = _create_fallback_report(fallback_state, str(e))
        
        error_message = AIMessage(
            content=f"Report generation encountered an error, but a summary report was created:\n\n{fallback_report}",
            name="writer"
        )
        
        return {
            "messages": [error_message],
            "final_report": fallback_report,
            "current_agent": "completed"
        }


def _prepare_research_summary(state: ResearchMultiAgentState) -> str:
    """Prepare a summary of research findings for the report writer."""
    if not state.research_results:
        return "No research results available."
    
    summary_parts = []
    
    # Group results by subtopic if possible
    subtopic_results = {}
    other_results = []
    
    for result in state.research_results:
        # Try to match result to subtopics
        matched_subtopic = None
        for subtopic in state.research_plan.subtopics:
            if any(word.lower() in result.query.lower() for word in subtopic.split()[:3]):
                matched_subtopic = subtopic
                break
        
        if matched_subtopic:
            if matched_subtopic not in subtopic_results:
                subtopic_results[matched_subtopic] = []
            subtopic_results[matched_subtopic].append(result)
        else:
            other_results.append(result)
    
    # Format grouped results
    for subtopic, results in subtopic_results.items():
        summary_parts.append(f"\n--- {subtopic} ---")
        for i, result in enumerate(results[:3], 1):  # Limit to 3 results per subtopic
            summary_parts.append(f"{i}. Query: {result.query}")
            summary_parts.append(f"   Content: {result.content[:500]}...")
            summary_parts.append(f"   Source: {result.source}")
            summary_parts.append("")
    
    # Add other results
    if other_results:
        summary_parts.append("\n--- Additional Research ---")
        for i, result in enumerate(other_results[:5], 1):  # Limit to 5 additional results
            summary_parts.append(f"{i}. Query: {result.query}")
            summary_parts.append(f"   Content: {result.content[:300]}...")
            summary_parts.append(f"   Source: {result.source}")
            summary_parts.append("")
    
    return "\n".join(summary_parts)


def _enhance_report_formatting(report: str, state) -> str:
    """Enhance the report with additional formatting and metadata."""
    
    # Get values safely from state
    main_topic = getattr(state.research_plan, 'main_topic', 'Unknown Topic') if state.research_plan else 'Unknown Topic'
    research_results = getattr(state, 'research_results', [])
    research_iterations = getattr(state, 'research_iterations', 1)
    
    # Add metadata header
    metadata = f"""# Research Report

**Topic:** {main_topic}
**Research Date:** {_get_current_date()}
**Sources Consulted:** {len(research_results)} research sources
**Research Iterations:** {research_iterations}

---

"""
    
    # Add source appendix if not already included
    if research_results and "sources:" not in report.lower():
        sources_section = "\n\n## Sources\n\n"
        unique_sources = set()
        
        for i, result in enumerate(research_results, 1):
            if result.source not in unique_sources and result.source != "Unknown source":
                sources_section += f"{i}. {result.source}\n"
                unique_sources.add(result.source)
        
        report += sources_section
    
    return metadata + report


def _create_fallback_report(state, error_msg: str) -> str:
    """Create a basic fallback report when the main report generation fails."""
    
    # Get the original query safely
    messages = getattr(state, 'messages', [])
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user")]
    
    if user_messages:
        first_msg = user_messages[0]
        if isinstance(first_msg, dict):
            original_query = first_msg.get("content", "Unknown query")
        else:
            original_query = getattr(first_msg, 'content', "Unknown query")
    else:
        original_query = "Unknown query"
    
    research_plan = getattr(state, 'research_plan', None)
    main_topic = getattr(research_plan, 'main_topic', original_query) if research_plan else original_query
    
    fallback_report = f"""# Research Report - Summary

**Topic:** {main_topic}
**Status:** Partial report due to processing error

## Summary

This is a summary report generated after encountering an error during full report generation.

**Original Query:** {original_query}

"""
    
    if research_plan:
        subtopics = getattr(research_plan, 'subtopics', [])
        questions = getattr(research_plan, 'research_questions', [])
        
        fallback_report += f"""## Research Plan

**Main Topic:** {research_plan.main_topic}

**Subtopics Investigated:**
{chr(10).join(f"- {subtopic}" for subtopic in subtopics)}

**Key Research Questions:**
{chr(10).join(f"- {question}" for question in questions[:5])}

"""
    
    research_results = getattr(state, 'research_results', [])
    research_iterations = getattr(state, 'research_iterations', 1)
    
    if research_results:
        fallback_report += f"""## Research Findings Summary

**Total Sources Consulted:** {len(research_results)}
**Research Iterations Completed:** {research_iterations}

**Key Findings:**
"""
        
        # Add brief summary of first few results
        for i, result in enumerate(research_results[:3], 1):
            fallback_report += f"""
{i}. **{result.query}**
   - {result.content[:200]}...
   - Source: {result.source}
"""
    
    fallback_report += f"""
## Note

This report was generated in summary mode due to a processing error: {error_msg}

For a complete analysis, please retry the research request.
"""
    
    return fallback_report


def _get_current_date() -> str:
    """Get the current date for report metadata."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
