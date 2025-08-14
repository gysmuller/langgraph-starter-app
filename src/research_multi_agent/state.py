"""
State schema for the research multi-agent graph.

This module defines the shared state that flows through all agents in the research workflow.
"""

from typing import List, Optional
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Structured research plan created by the planner"""
    main_topic: str = Field(description="Main research topic")
    subtopics: List[str] = Field(description="List of subtopics to research")
    research_questions: List[str] = Field(description="Specific questions to answer")
    

class ResearchResult(BaseModel):
    """Individual research result from a search"""
    query: str = Field(description="Search query used")
    content: str = Field(description="Research content found")
    source: str = Field(description="Source URL")
    relevance_score: float = Field(description="Relevance score", default=1.0)


class ResearchMultiAgentState(MessagesState):
    """
    Shared state for the entire multi-agent research graph.
    This single state flows through all agents in sequence.
    
    Extends MessagesState, so the conversation history (messages) is preserved
    throughout the entire workflow, allowing agents to access the original
    user query and any intermediate messages.
    """
    # Planning phase outputs
    research_plan: Optional[ResearchPlan] = None
    
    # Research phase outputs
    research_results: List[ResearchResult] = []
    research_iterations: int = 0    # Track research iterations
    max_research_iterations: int = 5  # Configurable limit
    
    # Report writing phase outputs
    final_report: Optional[str] = None
    
    # Workflow tracking
    current_agent: str = "planner"  # Track active agent for debugging
