# Research Graph Implementation Plan

## Overview
This document outlines the plan for implementing a new multi-agent research graph with three specialized agents:
1. **Planning Agent** - Creates a research plan from user input
2. **Research Agent** - Executes the research plan using TavilySearch
3. **Report Writer Agent** - Generates a final report based on research results

### State Flow Architecture
The graph uses a single shared state that flows through all agents:

```
User Input → [ResearchMultiAgentState]
                    ↓
            [Planning Agent]
                    ↓ (adds research_plan to state)
            [Research Agent]
                    ↓ (adds research_results to state)
            [Report Writer Agent]
                    ↓ (adds final_report to state)
              Final Output
```

Each agent reads from and writes to the same state instance, ensuring seamless data flow and coordination.

## Directory Structure

Following the LangGraph platform best practices, the new graph will be structured as follows:

```
src/
├── chat_basic/              # (existing basic chat agent)
│   ├── __init__.py
│   ├── agent.py
│   └── utils/
│       ├── __init__.py
│       ├── agent_config.py
│       ├── context_schema.py
│       ├── prompts.py
│       ├── tools.py
│       └── prompts/
│           └── chat_basic_system_prompt.txt
│
└── research_multi_agent/    # (new research graph)
    ├── __init__.py
    ├── graph.py            # Main graph definition and orchestration
    ├── state.py            # State schema definition
    └── agents/             # Individual agent implementations
    │   ├── __init__.py
    │   ├── planner.py      # Planning agent
    │   ├── researcher.py   # Research agent  
    │   └── writer.py       # Report writer agent
    └── utils/
        ├── __init__.py
        ├── agent_config.py # Agent configuration (reuse/extend from chat_basic)
        ├── context_schema.py # Context schema for the research graph
        ├── tools.py        # Tool definitions (including TavilySearch)
        └── prompts/        # System prompts for each agent
            ├── planner_prompt.txt
            ├── researcher_prompt.txt
            └── writer_prompt.txt
```

## Implementation Details

### 1. State Schema (`state.py`)

All agents in the graph share a single state instance that flows through the entire workflow. This enables seamless information passing and coordination between agents.

```python
from typing import List, Optional, TypedDict
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
    relevance_score: float = Field(description="Relevance score")

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
```

### 2. Agent Implementations

Each agent is implemented as a node function that receives the shared state, performs its specific task, and returns a partial state update. LangGraph automatically merges these updates into the existing state, preserving all other fields.

#### Planning Agent (`agents/planner.py`)
```python
async def call_planner_agent(state: ResearchMultiAgentState) -> dict:
    """
    Planning agent node that analyzes user input and creates a research plan.
    
    Input: state.messages (user query)
    Output: Updates state.research_plan with structured plan
    """
    # Analyze the last user message to create research plan
    # Use LLM to generate ResearchPlan object
    # Return updated state with research_plan field populated
    return {"research_plan": plan, "current_agent": "researcher"}
```

#### Research Agent (`agents/researcher.py`) 
```python
async def call_research_agent(state: ResearchMultiAgentState) -> dict:
    """
    Research agent node that executes the research plan.
    
    Input: state.research_plan (from planning agent)
    Output: Updates state.research_results with findings
    """
    # Read research plan from state
    # Execute searches using TavilySearch tool
    # Accumulate results in state.research_results
    # Increment state.research_iterations
    # Return updated state with research results
    return {
        "research_results": results,
        "research_iterations": state.research_iterations + 1,
        "current_agent": "writer"
    }
```

#### Report Writer Agent (`agents/writer.py`)
```python
async def call_writer_agent(state: ResearchMultiAgentState) -> dict:
    """
    Report writer agent node that creates the final report.
    
    Input: state.messages (original query) + state.research_results
    Output: Updates state.final_report with synthesized report
    """
    # Read original query and research results from state
    # Synthesize information into comprehensive report
    # Format with proper structure and citations
    # Return updated state with final_report
    return {"final_report": report, "current_agent": "completed"}
```

### 3. Main Graph (`graph.py`)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig
from typing import Literal
# Import agent node functions
from src.research_multi_agent.agents.planner import call_planner_agent
from src.research_multi_agent.agents.researcher import call_research_agent
from src.research_multi_agent.agents.writer import call_writer_agent
from src.research_multi_agent.state import ResearchMultiAgentState

async def create_research_multi_agent_graph(config: RunnableConfig = None):
    """Create the multi-agent research graph"""
    
    # Extract checkpointer and interrupt_before from config if provided
    checkpointer = getattr(config, 'checkpointer', None) if config else None
    interrupt_before = getattr(config, 'interrupt_before', None) if config else None
    
    try:
        # Initialize the StateGraph
        builder = StateGraph(ResearchMultiAgentState)
        
        # Add agent nodes
        builder.add_node("planner", call_planner_agent)
        builder.add_node("researcher", call_research_agent) 
        builder.add_node("writer", call_writer_agent)
        
        # Define the flow
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "researcher")
        builder.add_edge("researcher", "writer")
        builder.add_edge("writer", END)
        
        # Compile with optional checkpointer
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to create research multi-agent graph: {e}")
        raise
    
    # If checkpointer is provided and has setup method, call it
    if checkpointer is not None:
        try:
            if hasattr(checkpointer, 'setup'):
                await checkpointer.setup()
            if interrupt_before:
                print(f"[INFO] Research multi-agent graph created with checkpointer and interrupts: {interrupt_before}")
            else:
                print("[INFO] Research multi-agent graph created with checkpointer")
        except Exception as e:
            print(f"[WARNING] Checkpointer setup failed: {e}")
            print("[INFO] Graph will continue without persistent state")
    else:
        print("[INFO] Research multi-agent graph created without checkpointer")
    
    return graph
```

### 4. Tool Configuration (`utils/tools.py`)

Extend the existing tools module to include TavilySearch configuration specifically for the research agent:

```python
async def prepare_research_tools():
    """Prepare tools for the research agent"""
    tools = []
    
    # TavilySearch with research-specific configuration
    tavily_tool = TavilySearchResults(
        name="tavily_search",
        description="Search for research information",
        max_results=5,  # More results for comprehensive research
        include_raw_content=True,  # Get full content
        include_answer=True  # Get AI-generated answer
    )
    tools.append(tavily_tool)
    
    return tools
```

### 5. Context Schema (`utils/context_schema.py`)

```python
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
```

### 6. System Prompts

Each agent will have its own specialized prompt in the `utils/prompts/` directory:

- **planner_prompt.txt**: Instructions for creating comprehensive research plans
- **researcher_prompt.txt**: Instructions for systematic research execution
- **writer_prompt.txt**: Instructions for synthesizing research into reports

### 7. Update langgraph.json

Add the new graph to the configuration:

```json
{
    "dependencies": [
        "langchain_openai",
        "./src"
    ],
    "graphs": {
        "chat_basic": "./src/chat_basic/agent.py:create_chat_basic_graph",
        "research_multi_agent": "./src/research_multi_agent/graph.py:create_research_multi_agent_graph"
    },
    "env": "./.env"
}
```

### 8. Checkpointing Integration

The research graph will use the shared PostgreSQL checkpointer from `src/shared/checkpoint.py` for persistent state management:

```python
# Example usage in deployment/API integration
from langchain_core.runnables import RunnableConfig
from src.shared.checkpoint import get_postgres_checkpointer
from src.research_multi_agent.graph import create_research_multi_agent_graph

async def create_research_graph_with_persistence():
    """Create research graph with PostgreSQL checkpointing"""
    async with get_postgres_checkpointer() as checkpointer:
        config = RunnableConfig(
            checkpointer=checkpointer,
            interrupt_before=["researcher"]  # Optional: human-in-the-loop before research
        )
        graph = await create_research_multi_agent_graph(config)
        return graph
```

This enables:
- Persistent storage of research state across sessions
- Recovery from interruptions during long research tasks
- Human-in-the-loop capabilities (e.g., approve research plan)
- Audit trail of research steps

## Key Design Decisions

1. **Single Shared State**: All agents operate on a single `ResearchMultiAgentState` instance that flows through the entire graph. This design:
   - Eliminates state synchronization issues
   - Provides clear data flow visibility
   - Enables easy debugging and monitoring
   - Supports checkpointing the entire workflow state

2. **Explicit Flow Pattern**: Using a deterministic flow (planner → researcher → writer) rather than dynamic routing for clarity and predictability.

3. **State-Based Communication**: Agents communicate exclusively through the shared state:
   - Planning agent adds `research_plan` to state
   - Research agent reads the plan and adds `research_results`
   - Writer agent reads both to create `final_report`

4. **Modular Agent Design**: Each agent is self-contained with its own prompts and logic, but all share the same state schema.

5. **Research Iteration Control**: The research agent has a configurable iteration limit to prevent infinite loops.

6. **Tool Isolation**: Only the research agent has access to TavilySearch, maintaining clear separation of concerns.

7. **Reusable Components**: Leveraging existing utilities from chat_basic where appropriate (agent_config, base tools).

8. **Persistent State Management**: Using the shared PostgreSQL checkpointer for production deployments, enabling recovery and human-in-the-loop workflows.

## Implementation Steps

1. Create the directory structure
2. Implement state.py with the state schema
3. Create placeholder files for each agent
4. Implement the planning agent
5. Implement the research agent with TavilySearch integration
6. Implement the report writer agent
7. Create the main graph orchestration (with checkpointer support)
8. Write system prompts for each agent
9. Update langgraph.json
10. Test the complete workflow (with and without checkpointing)
11. Test persistence and recovery scenarios

## Testing Strategy

1. Unit tests for each agent's logic
2. Integration test for the full graph flow
3. Test with various research topics
4. Validate research iteration limits
5. Test error handling and edge cases

## Future Enhancements

1. Add human-in-the-loop for research plan approval
2. Implement parallel research for multiple subtopics
3. Add specialized research tools beyond TavilySearch
4. Implement citation validation
5. Add report formatting options (PDF, HTML)
6. Implement caching for repeated searches
