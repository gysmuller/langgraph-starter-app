"""
Context schema for the research multi-agent graph.

Defines the configuration context that can be passed to customize the research workflow.
"""

from pydantic import BaseModel, Field


class ResearchMultiAgentContextSchema(BaseModel):
    """Context schema for research multi-agent graph"""
    
    research_depth: str = Field(
        default="comprehensive",
        description="Depth of research: 'quick', 'standard', or 'comprehensive'"
    )
    
    output_format: str = Field(
        default="markdown", 
        description="Output format for the report: 'markdown' or 'plain_text'"
    )
    
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations in the report"
    )
    
    max_research_iterations: int = Field(
        default=5,
        description="Maximum number of research iterations per topic"
    )
    
    max_search_results_per_query: int = Field(
        default=5,
        description="Maximum number of search results per individual query"
    )
