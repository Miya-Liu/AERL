"""Database service module for Supabase connection management and agent operations.

This module provides:
- DBConnection: Thread-safe singleton for async Supabase client management
- SyncDBConnection: Thread-safe singleton for sync Supabase client management
- create_task: Helper function to create task records in the database
- get_llm_messages: Helper function to fetch LLM messages for a task
- add_message: Helper function to add a message to the database
- AgentService: Service class for agent CRUD operations
- AgentFilters: Dataclass for agent query filters
- PaginationParams: Dataclass for pagination parameters
- TPFC_CONFIG: Builtin configuration for TPFC agent
- Agent schemas: Pydantic models for agent requests/responses
- AgentLoader: Unified agent loading service
- get_agent_loader: Get the global AgentLoader instance
"""

from .agent_loader import AgentConfig, AgentData, AgentLoader, get_agent_loader
from .agent_service import AgentFilters, AgentService, get_agent_service
from .builtin import TPFC_CONFIG
from .connection import DBConnection, SyncDBConnection
from .messages import add_message, get_llm_messages
from .pagination import PaginationParams
from .schemas import AgentCreateRequest, AgentResponse, AgentUpdateRequest
from .tasks import TaskStatus, create_task

__all__ = [
    "DBConnection",
    "SyncDBConnection",
    "create_task",
    "get_llm_messages",
    "add_message",
    "TaskStatus",
    "TPFC_CONFIG",
    "AgentCreateRequest",
    "AgentResponse",
    "AgentUpdateRequest",
    "PaginationParams",
    "AgentService",
    "AgentFilters",
    "get_agent_service",
    "AgentConfig",
    "AgentData",
    "AgentLoader",
    "get_agent_loader",
]
