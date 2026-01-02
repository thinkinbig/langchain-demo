"""
LangMem integration for native LangGraph memory management.

Uses LangMem's built-in tools and managers instead of reimplementing functionality.
"""

from typing import Any, Optional, Tuple

from langgraph.store.base import BaseStore
from langmem import (
    create_manage_memory_tool,
    create_memory_store_manager,
    create_search_memory_tool,
)  # type: ignore[import-untyped]
from memory.long_term_memory import LongTermMemory


def get_langmem_tools(
    store: BaseStore, namespace: tuple = ("memories",)
) -> Tuple[Any, Any]:
    """
    Get LangMem tools for memory management and search.

    These tools can be added to agent tool lists to enable memory capabilities.

    Args:
        store: LangGraph store instance (e.g., from checkpointer or InMemoryStore)
        namespace: Namespace tuple for memory storage (default: ("memories",))
                   Can use ("memories", "{user_id}") for per-user scoping

    Returns:
        Tuple of (manage_tool, search_tool) that can be added to agent tools

    Example:
        >>> from langgraph.store.memory import InMemoryStore
        >>> store = InMemoryStore(
        ...     index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        ... )
        >>> manage_tool, search_tool = get_langmem_tools(store)
        >>> agent = create_react_agent(
        ...     "anthropic:claude-3-5-sonnet-latest",
        ...     tools=[manage_tool, search_tool]
        ... )
    """
    manage_tool = create_manage_memory_tool(namespace=namespace)
    search_tool = create_search_memory_tool(namespace=namespace)
    return manage_tool, search_tool


def get_langmem_manager(store: BaseStore) -> Any:
    """
    Get LangMem background memory manager for automatic extraction and consolidation.

    This manager automatically extracts, consolidates, and updates memories
    from conversations in the background.

    Args:
        store: LangGraph store instance

    Returns:
        Memory store manager instance that can process conversations

    Example:
        >>> from langgraph.store.memory import InMemoryStore
        >>> store = InMemoryStore(
        ...     index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        ... )
        >>> manager = get_langmem_manager(store)
        >>> manager.invoke({"messages": conversation_messages})
    """
    return create_memory_store_manager(store=store)


class LongTermMemoryLangMemBridge:
    """
    Bridge between LongTermMemory and LangMem for unified access.

    Allows using both application-scoped (LongTermMemory) and
    agent-scoped (LangMem via tools) memories through a single interface.
    """

    def __init__(
        self,
        long_term_memory: Optional[LongTermMemory] = None,
        store: Optional[BaseStore] = None,
        namespace: tuple = ("memories",),
    ):
        """
        Initialize bridge.

        Args:
            long_term_memory: LongTermMemory instance for app-scoped memories
            store: LangGraph store for LangMem integration
            namespace: Namespace tuple for LangMem memory storage
        """
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.store = store
        self.namespace = namespace

        # Get LangMem tools if store is provided
        if self.store:
            try:
                self.manage_tool, self.search_tool = get_langmem_tools(
                    self.store, self.namespace
                )
                self.memory_manager = get_langmem_manager(self.store)
            except Exception:
                self.manage_tool = None
                self.search_tool = None
                self.memory_manager = None
        else:
            self.manage_tool = None
            self.search_tool = None
            self.memory_manager = None

    def get_tools(self) -> list:
        """
        Get LangMem tools for use in agent tool lists.

        Returns:
            List of tools [manage_tool, search_tool] or empty list if not available
        """
        if self.manage_tool and self.search_tool:
            return [self.manage_tool, self.search_tool]
        return []

    def process_conversation(self, messages: list) -> dict:
        """
        Process conversation through LangMem's background manager.

        Args:
            messages: List of conversation messages

        Returns:
            Result from memory manager processing
        """
        if self.memory_manager:
            return self.memory_manager.invoke({"messages": messages})
        return {}

    def store_app_memory(
        self,
        content: str,
        priority: float = 1.0,
        ttl_days: Optional[int] = None,
        tags: Optional[list] = None,
        **kwargs
    ) -> str:
        """
        Store memory in application-scoped LongTermMemory.

        Args:
            content: Memory content
            priority: Priority score
            ttl_days: Time-to-live in days
            tags: List of tags
            **kwargs: Additional arguments

        Returns:
            Memory ID
        """
        return self.long_term_memory.store_memory(
            content=content,
            metadata=kwargs.get("metadata"),
            priority=priority,
            ttl_days=ttl_days,
            tags=tags,
        )

    def retrieve_app_memory(
        self,
        query: str,
        k: int = 5,
        min_relevance: float = 0.5,
        **kwargs
    ) -> list:
        """
        Retrieve memories from application-scoped LongTermMemory.

        Args:
            query: Search query
            k: Number of results
            min_relevance: Minimum relevance score
            **kwargs: Additional retrieval arguments

        Returns:
            List of (Document, relevance_score) tuples
        """
        return self.long_term_memory.retrieve_memories(
            query=query,
            k=k,
            min_relevance=min_relevance,
            filter_by_tags=kwargs.get("filter_by_tags"),
            filter_by_priority=kwargs.get("filter_by_priority"),
            exclude_expired=kwargs.get("exclude_expired", True),
        )
