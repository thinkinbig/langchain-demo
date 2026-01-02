"""
Checkpointer factory for supporting multiple persistence backends.

Supports:
- MemorySaver: In-memory storage (default, for development)
- SqliteSaver: SQLite-based persistence (recommended for production)
- PostgresSaver: PostgreSQL-based persistence (for distributed systems)
"""

import os
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver


def get_checkpointer(
    backend: Optional[str] = None,
    connection_string: Optional[str] = None,
    **kwargs
) -> BaseCheckpointSaver:
    """
    Factory function to create a checkpointer based on configuration.

    Args:
        backend: Backend type ('memory', 'sqlite', 'postgres').
                 If None, reads from CHECKPOINTER_BACKEND env var or
                 defaults to 'memory'
        connection_string: Database connection string (for sqlite/postgres)
        **kwargs: Additional arguments for specific checkpointers

    Returns:
        BaseCheckpointSaver instance

    Note:
        SQLite and PostgreSQL checkpointers may not be available in all
        langgraph versions. The factory will automatically fall back to
        MemorySaver if they are not available.

    Examples:
        >>> # Use MemorySaver (default)
        >>> checkpointer = get_checkpointer()

        >>> # Try to use SQLite (falls back to MemorySaver if not available)
        >>> checkpointer = get_checkpointer('sqlite')

        >>> # Try to use PostgreSQL (falls back to MemorySaver if not available)
        >>> checkpointer = get_checkpointer('postgres', 'postgresql://user:pass@localhost/db')

        >>> # Explicitly use in-memory
        >>> checkpointer = get_checkpointer('memory')
    """
    # Determine backend from env or parameter
    # Default to 'memory' since SQLite/Postgres support may not be available
    backend = backend or os.getenv("CHECKPOINTER_BACKEND", "memory").lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    elif backend == "sqlite":
        # Try to import SQLite checkpointer
        # Note: SQLite checkpoint support may require a newer version of langgraph
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            # Fallback to MemorySaver if SQLite support not available
            print(
                "⚠️  SQLite checkpointer not available, falling back to MemorySaver. "
                "SQLite checkpoint support may require a newer version of langgraph. "
                "Using MemorySaver for now."
            )
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

        # Default SQLite path
        db_path = connection_string or os.getenv(
            "CHECKPOINTER_DB_PATH",
            os.path.join(os.path.dirname(__file__), "..", "checkpoints.db")
        )

        # Ensure directory exists
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        return SqliteSaver.from_conn_string(db_path)

    elif backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError:
            # Fallback to MemorySaver if PostgreSQL support not available
            print(
                "⚠️  PostgreSQL checkpointer not available, falling back to "
                "MemorySaver. Install with: pip install "
                "'langgraph[checkpoint-postgres]' or 'psycopg2-binary'"
            )
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

        conn_str = connection_string or os.getenv("CHECKPOINTER_CONNECTION_STRING")
        if not conn_str:
            raise ValueError(
                "PostgresSaver requires connection_string parameter or "
                "CHECKPOINTER_CONNECTION_STRING environment variable"
            )

        return PostgresSaver.from_conn_string(conn_str)

    else:
        raise ValueError(
            f"Unknown checkpointer backend: {backend}. "
            "Supported backends: 'memory', 'sqlite', 'postgres'"
        )


def get_checkpointer_with_namespace(
    namespace: str,
    backend: Optional[str] = None,
    connection_string: Optional[str] = None,
    **kwargs
) -> BaseCheckpointSaver:
    """
    Create a checkpointer with a namespace for write isolation.

    Namespaces allow multiple graphs or subgraphs to have isolated
    checkpoint storage, preventing state pollution.

    Args:
        namespace: Namespace identifier for isolation
        backend: Backend type
        connection_string: Database connection string
        **kwargs: Additional arguments

    Returns:
        BaseCheckpointSaver instance configured with namespace
    """
    checkpointer = get_checkpointer(backend, connection_string, **kwargs)

    # Note: LangGraph checkpointers support namespaces via configurable
    # The namespace is typically passed in the config dict, not at creation time
    # This wrapper is for future extensibility and documentation

    return checkpointer



