"""
Findings Memory Management with Strategy Pattern.

This module provides a flexible findings storage and retrieval system using
the Strategy pattern to support multiple retrieval strategies (LangMem, keyword
matching, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore


class FindingsRetrievalStrategy(ABC):
    """
    Abstract base class for findings retrieval strategies.

    Defines the interface that all retrieval strategies must implement.
    """

    @abstractmethod
    def store_findings(self, findings: List[dict]) -> bool:
        """
        Store findings using this strategy.

        Args:
            findings: List of finding dictionaries with keys: task, summary,
                     sources, etc.

        Returns:
            True if storage successful, False otherwise
        """
        pass

    @abstractmethod
    def retrieve_relevant_findings(
        self,
        query_context: str,
        top_k: int = 5
    ) -> str:
        """
        Retrieve relevant findings based on query context.

        Args:
            query_context: Query string combining query and previous step output
            top_k: Number of top findings to retrieve

        Returns:
            Formatted string of relevant findings, or empty string if retrieval
            fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the strategy is available and ready to use.

        Returns:
            True if strategy is ready, False otherwise
        """
        pass


class LangMemRetrievalStrategy(FindingsRetrievalStrategy):
    """
    LangMem-based semantic retrieval strategy.

    Uses LangMem tools for semantic search and storage.
    Falls back gracefully if LangMem is not available.
    """

    def __init__(self, thread_id: Optional[str] = None):
        """
        Initialize LangMem retrieval strategy.

        Args:
            thread_id: Optional thread ID for namespace isolation
        """
        self.thread_id = thread_id or "default"
        self.namespace = ("synthesizer_findings", self.thread_id)
        self.store: Optional[BaseStore] = None
        self.manage_tool = None
        self.search_tool = None
        self.memory_manager = None
        self._initialized = False
        self._findings_cache: List[dict] = []

    def _initialize(self) -> bool:
        """
        Initialize the LangMem store and tools.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Create InMemoryStore with embedding configuration
            self.store = InMemoryStore(
                index={
                    "dims": 1536,
                    "embed": "openai:text-embedding-3-small",
                }
            )

            # Lazy import LangMem to avoid blocking on import
            try:
                from langmem import (
                    create_manage_memory_tool,
                    create_memory_store_manager,
                    create_search_memory_tool,
                )  # type: ignore[import-untyped]

                # Create memory store manager
                # Note: Some versions require model parameter
                self.memory_manager = None
                try:
                    # First try with model parameter (newer API)
                    from llm.factory import get_llm_by_model_choice
                    model = get_llm_by_model_choice("turbo")
                    self.memory_manager = create_memory_store_manager(
                        store=self.store,
                        model=model
                    )
                except TypeError:
                    # If TypeError (wrong signature), try without model (older API)
                    try:
                        self.memory_manager = create_memory_store_manager(
                            store=self.store
                        )
                    except Exception as e:
                        print(f"  ⚠️  Failed to create memory manager (old API): {e}")
                        self.memory_manager = None
                except Exception as e:
                    # Other exceptions (e.g., ImportError for llm.factory)
                    print(f"  ⚠️  Failed to create memory manager: {e}")
                    # Try without model as fallback
                    try:
                        self.memory_manager = create_memory_store_manager(
                            store=self.store
                        )
                    except Exception as e2:
                        print(f"  ⚠️  Failed to create memory manager (fallback): {e2}")
                        self.memory_manager = None

                # Create LangMem tools
                try:
                    self.manage_tool = create_manage_memory_tool(
                        namespace=self.namespace
                    )
                    self.search_tool = create_search_memory_tool(
                        namespace=self.namespace
                    )
                except Exception as e:
                    print(f"  ⚠️  Failed to create LangMem tools: {e}")
                    self.manage_tool = None
                    self.search_tool = None
            except ImportError as e:
                print(f"  ⚠️  LangMem not available: {e}")
                self.memory_manager = None
                self.manage_tool = None
                self.search_tool = None

            self._initialized = True
            return True
        except Exception as e:
            print(f"  ⚠️  Failed to initialize LangMem strategy: {e}")
            self.store = None
            self.manage_tool = None
            self.search_tool = None
            return False

    def store_findings(self, findings: List[dict]) -> bool:
        """
        Store findings using LangMem.

        Args:
            findings: List of finding dictionaries

        Returns:
            True if storage successful, False otherwise
        """
        # Always cache findings for fallback
        self._findings_cache = findings.copy()

        if not self._initialize():
            # If initialization fails, we still have the cache
            return True

        if not findings:
            return True

        try:
            # Store each finding as a separate memory
            for i, finding in enumerate(findings):
                task = finding.get('task', 'Unknown')
                summary = finding.get('summary', 'No summary')
                sources = finding.get('sources', [])
                citations = finding.get('extracted_citations', [])

                # Format finding content for storage
                content_parts = [f"Task: {task}", f"Summary: {summary}"]

                if sources:
                    source_list = ", ".join([
                        s.get('title', 'Unknown')[:60]
                        for s in sources[:5]
                    ])
                    if len(sources) > 5:
                        source_list += f" (+{len(sources) - 5} more)"
                    content_parts.append(f"Sources: {source_list}")

                if citations:
                    citation_titles = [
                        c.get('title', 'Unknown')[:60]
                        for c in citations[:3]
                    ]
                    if citation_titles:
                        content_parts.append(
                            f"Mentioned Papers: {', '.join(citation_titles)}"
                        )
                        if len(citations) > 3:
                            content_parts.append(
                                f" (+{len(citations) - 3} more)"
                            )

                content = "\n".join(content_parts)
                memory_id = f"finding_{i}_{self.thread_id}"
                stored = False

                # Try using memory manager first
                if self.memory_manager:
                    try:
                        from langchain_core.messages import HumanMessage
                        self.memory_manager.invoke({
                            "messages": [HumanMessage(content=content)]
                        })
                        stored = True
                    except Exception:
                        pass

                # Fallback to direct store API
                if not stored and self.store:
                    try:
                        key = self.namespace + (memory_id,)
                        self.store.put(key, {
                            "content": content,
                            "memory_id": memory_id,
                            "task": task,
                            "summary": summary,
                        })
                        stored = True
                    except Exception as e:
                        # Last resort: try tool
                        if self.manage_tool:
                            try:
                                self.manage_tool.invoke({
                                    "content": content,
                                    "memory_id": memory_id,
                                })
                                stored = True
                            except Exception as e2:
                                print(f"  ⚠️  Failed to store finding {i}: {e2}")
                                continue
                        else:
                            print(f"  ⚠️  Failed to store finding {i}: {e}")
                            continue

            return True
        except Exception as e:
            print(f"  ⚠️  Failed to store findings: {e}")
            return False

    def retrieve_relevant_findings(
        self,
        query_context: str,
        top_k: int = 5,
        min_relevance: float = 0.3
    ) -> str:
        """
        Retrieve relevant findings using LangMem semantic search.

        Falls back to keyword-based retrieval if LangMem fails.

        Args:
            query_context: Query string
            top_k: Number of findings to retrieve
            min_relevance: Minimum relevance score (unused for now)

        Returns:
            Formatted string of relevant findings
        """
        if not self._initialize():
            # Fallback to keyword retrieval if not initialized
            return self._keyword_fallback(query_context, top_k)

        try:
            result = None

            # Try search tool first
            if self.search_tool:
                try:
                    result = self.search_tool.invoke({
                        "query": query_context,
                        "limit": top_k,
                    })
                except Exception:
                    pass

            # Try store's search method if available
            if result is None and self.store and hasattr(self.store, 'search'):
                try:
                    result = self.store.search(
                        query=query_context,
                        namespace=self.namespace,
                        limit=top_k
                    )
                except Exception:
                    pass

            # Parse result
            if result is None:
                # Fallback to keyword retrieval
                return self._keyword_fallback(query_context, top_k)

            memories = []
            if isinstance(result, list):
                memories = result
            elif isinstance(result, dict):
                memories = result.get("memories", [])
                if not memories and "results" in result:
                    memories = result["results"]

            if not memories:
                # Fallback to keyword retrieval
                return self._keyword_fallback(query_context, top_k)

            # Format retrieved findings
            formatted_findings = []
            for i, memory in enumerate(memories, 1):
                if isinstance(memory, dict):
                    content = memory.get("content", "")
                    lines = content.split("\n")
                    task_line = next(
                        (line for line in lines if line.startswith("Task:")),
                        ""
                    )
                    summary_line = next(
                        (line for line in lines if line.startswith("Summary:")),
                        ""
                    )

                    if task_line and summary_line:
                        task = task_line.replace("Task:", "").strip()
                        summary = summary_line.replace("Summary:", "").strip()
                        formatted_findings.append(
                            f"{i}. Task: {task}\n   Summary: {summary}"
                        )
                    else:
                        formatted_findings.append(f"{i}. {content}")
                elif isinstance(memory, str):
                    formatted_findings.append(f"{i}. {memory}")

            if not formatted_findings:
                # Fallback to keyword retrieval
                return self._keyword_fallback(query_context, top_k)

            return "\n\n".join(formatted_findings)

        except Exception as e:
            print(f"  ⚠️  Failed to retrieve findings: {e}")
            # Fallback to keyword retrieval
            return self._keyword_fallback(query_context, top_k)

    def _keyword_fallback(
        self,
        query_context: str,
        top_k: int = 5
    ) -> str:
        """
        Fallback keyword-based retrieval when LangMem fails.

        Args:
            query_context: Query string
            top_k: Number of findings to return

        Returns:
            Formatted string of relevant findings
        """
        if not self._findings_cache:
            return ""

        # Extract keywords from query context
        query_lower = query_context.lower()
        keywords = set()
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can'
        }
        for word in query_lower.split():
            word = word.strip('.,!?;:()[]{}"\'-')
            if len(word) > 3 and word not in common_words:
                keywords.add(word)

        # Score findings by keyword matches
        scored_findings = []
        for finding in self._findings_cache:
            task = finding.get('task', '').lower()
            summary = finding.get('summary', '').lower()
            finding_text = f"{task} {summary}"

            score = sum(1 for keyword in keywords if keyword in finding_text)
            if score > 0:
                scored_findings.append((score, finding))

        # Sort by score and take top_k
        scored_findings.sort(key=lambda x: x[0], reverse=True)
        top_findings = [f for _, f in scored_findings[:top_k]]

        if not top_findings:
            return ""

        # Format findings
        formatted = []
        for i, finding in enumerate(top_findings, 1):
            task = finding.get('task', 'Unknown')
            summary = finding.get('summary', 'No summary')
            formatted.append(f"{i}. Task: {task}\n   Summary: {summary}")

        return "\n\n".join(formatted)

    def is_available(self) -> bool:
        """
        Check if LangMem strategy is available.

        Returns:
            True if available, False otherwise
        """
        return self._initialized and self.store is not None


class KeywordRetrievalStrategy(FindingsRetrievalStrategy):
    """
    Keyword-based retrieval strategy.

    Simple keyword matching fallback when LangMem is not available.
    """

    def __init__(self):
        """Initialize keyword retrieval strategy."""
        self._findings_cache: List[dict] = []

    def store_findings(self, findings: List[dict]) -> bool:
        """
        Store findings in memory cache.

        Args:
            findings: List of finding dictionaries

        Returns:
            Always returns True (simple in-memory storage)
        """
        self._findings_cache = findings.copy()
        return True

    def retrieve_relevant_findings(
        self,
        query_context: str,
        top_k: int = 5,
        min_relevance: float = 0.3
    ) -> str:
        """
        Retrieve relevant findings using keyword matching.

        Args:
            query_context: Query string
            top_k: Number of findings to return
            min_relevance: Unused (for interface compatibility)

        Returns:
            Formatted string of relevant findings
        """
        if not self._findings_cache:
            return ""

        # Extract keywords from query context
        query_lower = query_context.lower()
        keywords = set()
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
            'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is',
            'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can'
        }
        for word in query_lower.split():
            word = word.strip('.,!?;:()[]{}"\'-')
            if len(word) > 3 and word not in common_words:
                keywords.add(word)

        # Score findings by keyword matches
        scored_findings = []
        for finding in self._findings_cache:
            task = finding.get('task', '').lower()
            summary = finding.get('summary', '').lower()
            finding_text = f"{task} {summary}"

            score = sum(1 for keyword in keywords if keyword in finding_text)
            if score > 0:
                scored_findings.append((score, finding))

        # Sort by score and take top_k
        scored_findings.sort(key=lambda x: x[0], reverse=True)
        top_findings = [f for _, f in scored_findings[:top_k]]

        if not top_findings:
            return ""

        # Format findings
        formatted = []
        for i, finding in enumerate(top_findings, 1):
            task = finding.get('task', 'Unknown')
            summary = finding.get('summary', 'No summary')
            formatted.append(f"{i}. Task: {task}\n   Summary: {summary}")

        return "\n\n".join(formatted)

    def is_available(self) -> bool:
        """
        Check if keyword strategy is available.

        Returns:
            True if we have cached findings, False otherwise
        """
        return bool(self._findings_cache)


class FindingsMemoryManager:
    """
    Manager for findings storage and retrieval using Strategy pattern.

    Delegates all operations to the configured retrieval strategy.
    """

    def __init__(
        self,
        strategy: FindingsRetrievalStrategy,
        thread_id: Optional[str] = None
    ):
        """
        Initialize FindingsMemoryManager with a strategy.

        Args:
            strategy: Retrieval strategy to use
            thread_id: Optional thread ID for namespace isolation
        """
        self.strategy = strategy
        self.thread_id = thread_id or "default"

    def store_findings(self, findings: List[dict]) -> bool:
        """
        Store findings using the configured strategy.

        Args:
            findings: List of finding dictionaries

        Returns:
            True if storage successful, False otherwise
        """
        return self.strategy.store_findings(findings)

    def retrieve_relevant_findings(
        self,
        query_context: str,
        top_k: int = 5,
        min_relevance: float = 0.3
    ) -> str:
        """
        Retrieve relevant findings using the configured strategy.

        Args:
            query_context: Query string combining query and previous step output
            top_k: Number of top findings to retrieve
            min_relevance: Minimum relevance score (0.0-1.0)

        Returns:
            Formatted string of relevant findings, or empty string if retrieval
            fails
        """
        return self.strategy.retrieve_relevant_findings(
            query_context, top_k, min_relevance
        )

    def is_available(self) -> bool:
        """
        Check if the manager is available and ready to use.

        Returns:
            True if manager is ready, False otherwise
        """
        return self.strategy.is_available()


def create_findings_memory_manager(
    thread_id: Optional[str] = None,
    strategy_type: str = "auto"
) -> FindingsMemoryManager:
    """
    Factory function to create FindingsMemoryManager with appropriate strategy.

    Args:
        thread_id: Optional thread ID for namespace isolation
        strategy_type: Strategy type - "auto", "langmem", or "keyword"
                     - "auto": Try LangMem first, fallback to keyword
                     - "langmem": Force LangMem strategy
                     - "keyword": Force keyword strategy

    Returns:
        FindingsMemoryManager instance with configured strategy
    """
    if strategy_type == "auto":
        # Try LangMem first, fallback to keyword
        try:
            strategy = LangMemRetrievalStrategy(thread_id)
            if strategy.is_available():
                return FindingsMemoryManager(strategy, thread_id)
        except Exception as e:
            print(f"  ⚠️  LangMem strategy not available: {e}")

        # Fallback to keyword
        strategy = KeywordRetrievalStrategy()
    elif strategy_type == "langmem":
        strategy = LangMemRetrievalStrategy(thread_id)
    else:  # keyword
        strategy = KeywordRetrievalStrategy()

    return FindingsMemoryManager(strategy, thread_id)

