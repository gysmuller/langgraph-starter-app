"""
Research Multi-Agent Graph

This module provides a multi-agent research system with three specialized agents:
1. Planning Agent - Creates research plans from user input
2. Research Agent - Executes research using TavilySearch
3. Report Writer Agent - Synthesizes research into comprehensive reports
"""

from .graph import create_research_multi_agent_graph

__all__ = ["create_research_multi_agent_graph"]
