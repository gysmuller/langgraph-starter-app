"""
Implementation of PostgreSQL-based checkpointing for LangGraph workflows.
This module provides checkpointing functionality for persistent storage of workflow state.
"""

from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver as BaseAsyncPostgresSaver
from src.shared.config import settings

@asynccontextmanager
async def get_postgres_checkpointer():
    """
    Context manager for getting a PostgreSQL checkpointer following the official LangGraph pattern.
    
    Automatically initializes database tables on first use for seamless setup.
    
    Environment variables:
        DATABASE_URL: PostgreSQL connection string (from settings)
    """
    db_url = settings.database_url
    if not db_url:
        raise ValueError("DATABASE_URL is not configured in settings")
    
    # Fix for Heroku PostgreSQL - convert postgres:// to postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    # Use the official LangGraph context manager pattern
    async with BaseAsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        print(f"[INFO] Using AsyncPostgresSaver with official context manager pattern")
        
        # Auto-initialize database tables if they don't exist
        try:
            await checkpointer.setup()
            print(f"[INFO] Database tables initialized successfully")
        except Exception as e:
            # Tables might already exist, which is fine
            print(f"[DEBUG] Database setup note: {e}")
        
        yield checkpointer

# Re-export the official AsyncPostgresSaver for convenience
AsyncPostgresSaver = BaseAsyncPostgresSaver
