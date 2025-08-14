"""
System prompts for the research multi-agent graph.

This module provides functions to load and format system prompts for each agent.
"""

import asyncio
from pathlib import Path


def _load_prompt_file_sync(filename: str) -> str:
    """Load a prompt file from the prompts directory synchronously."""
    prompts_dir = Path(__file__).parent / "prompts"
    file_path = prompts_dir / filename
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading prompt file {file_path}: {e}")


async def _load_prompt_file_async(filename: str) -> str:
    """Load a prompt file from the prompts directory asynchronously using asyncio.to_thread."""
    return await asyncio.to_thread(_load_prompt_file_sync, filename)


async def get_planner_system_prompt_async() -> str:
    """Get the system prompt for the planning agent (async)."""
    return await _load_prompt_file_async("planner_prompt.txt")


async def get_researcher_system_prompt_async() -> str:
    """Get the system prompt for the research agent (async)."""
    return await _load_prompt_file_async("researcher_prompt.txt")


async def get_writer_system_prompt_async() -> str:
    """Get the system prompt for the report writer agent (async)."""
    return await _load_prompt_file_async("writer_prompt.txt")


def get_planner_system_prompt() -> str:
    """Get the system prompt for the planning agent (sync)."""
    return _load_prompt_file_sync("planner_prompt.txt")


def get_researcher_system_prompt() -> str:
    """Get the system prompt for the research agent (sync)."""
    return _load_prompt_file_sync("researcher_prompt.txt")


def get_writer_system_prompt() -> str:
    """Get the system prompt for the report writer agent (sync)."""
    return _load_prompt_file_sync("writer_prompt.txt")
