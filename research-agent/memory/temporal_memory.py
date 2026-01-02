"""
Temporal memory wrapper with enhanced temporal annotations.

Extends LongTermMemory with temporal-aware operations following Zep/Graphiti patterns.
Provides soft deletion, version history, and temporal-aware retrieval.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from memory.long_term_memory import LongTermMemory


class TemporalMemory(LongTermMemory):
    """
    Temporal-aware memory manager extending LongTermMemory.

    Adds temporal annotations for soft deletion, version tracking, and
    temporal-aware memory retrieval following Zep/Graphiti patterns.
    """

    def store_memory_with_temporal(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 1.0,
        ttl_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
        supersedes: Optional[List[str]] = None
    ) -> str:
        """
        Store a memory with automatic temporal tracking and conflict handling.

        Args:
            content: Memory content to store
            metadata: Additional metadata
            priority: Priority score
            ttl_days: Time-to-live in days
            tags: List of tags for categorization
            supersedes: List of memory IDs this memory supersedes

        Returns:
            Memory ID
        """
        # Prepare temporal metadata
        temporal_metadata = metadata or {}

        if supersedes:
            temporal_metadata["supersedes"] = supersedes
            # Invalidate superseded memories
            for old_id in supersedes:
                self.invalidate_memory(old_id, reason="Superseded by new memory")
                # Update superseded memory to point to this one
                old_doc = self.get_memory_by_id(old_id)
                if old_doc:
                    old_doc.metadata["superseded_by"] = None  # Will be set after we get the new ID
                    self.vector_store.add_documents([old_doc], ids=[old_id])

        # Store the new memory
        memory_id = super().store_memory(
            content=content,
            metadata=temporal_metadata,
            priority=priority,
            ttl_days=ttl_days,
            tags=tags
        )

        # Update superseded memories to point to this new memory
        if supersedes:
            for old_id in supersedes:
                old_doc = self.get_memory_by_id(old_id)
                if old_doc:
                    old_doc.metadata["superseded_by"] = memory_id
                    self.vector_store.add_documents([old_doc], ids=[old_id])

        return memory_id

    def retrieve_valid_memories(
        self,
        query: str,
        k: int = 5,
        min_relevance: float = 0.5,
        filter_by_tags: Optional[List[str]] = None,
        filter_by_priority: Optional[float] = None,
        include_invalid: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve only valid memories (temporal-aware retrieval).

        Args:
            query: Search query
            k: Number of memories to retrieve
            min_relevance: Minimum relevance score
            filter_by_tags: Only retrieve memories with these tags
            filter_by_priority: Minimum priority score
            include_invalid: If True, include invalidated memories

        Returns:
            List of (Document, relevance_score) tuples
        """
        # Use parent's retrieve_memories with exclude_expired=True (which filters by is_valid)
        results = self.retrieve_memories(
            query=query,
            k=k * 2 if include_invalid else k,  # Get more if including invalid
            min_relevance=min_relevance,
            filter_by_tags=filter_by_tags,
            filter_by_priority=filter_by_priority,
            exclude_expired=not include_invalid  # exclude_expired filters by is_valid
        )

        if include_invalid:
            # If including invalid, we need to manually filter
            # The parent method already filters by is_valid when exclude_expired=True
            # So we need to do a separate search for invalid memories
            all_results = self.vector_store.similarity_search_with_score(query, k=k * 3)
            invalid_results = []

            for doc, score in all_results:
                if not doc.metadata.get("is_valid", True):
                    relevance = 1.0 / (1.0 + score) if score > 0 else 1.0
                    if relevance >= min_relevance:
                        invalid_results.append((doc, relevance))

            # Combine and sort
            all_combined = results + invalid_results
            all_combined.sort(key=lambda x: x[1], reverse=True)
            return all_combined[:k]

        return results[:k]

    def get_memory_timeline(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float, str]]:
        """
        Retrieve memory timeline showing how memories evolved over time.

        Args:
            query: Search query
            k: Number of memories to retrieve

        Returns:
            List of (Document, relevance_score, status) tuples where status is
            'valid', 'invalidated', or 'superseded'
        """
        # Get all memories matching query (including invalid)
        all_results = self.vector_store.similarity_search_with_score(query, k=k * 2)

        timeline = []
        for doc, score in all_results:
            relevance = 1.0 / (1.0 + score) if score > 0 else 1.0
            metadata = doc.metadata

            # Determine status
            if not metadata.get("is_valid", True):
                if metadata.get("superseded_by"):
                    status = "superseded"
                else:
                    status = "invalidated"
            else:
                status = "valid"

            timeline.append((doc, relevance, status))

        # Sort by stored_at (temporal order)
        timeline.sort(key=lambda x: x[0].metadata.get("stored_at", ""))

        return timeline[:k]

    def consolidate_memories(
        self,
        memory_ids: List[str],
        merge_strategy: str = "latest"
    ) -> Optional[str]:
        """
        Consolidate multiple memories into one.

        Args:
            memory_ids: List of memory IDs to consolidate
            merge_strategy: Strategy for merging ('latest', 'highest_priority', 'llm_reasoning')

        Returns:
            ID of consolidated memory, or None if consolidation failed
        """
        if not memory_ids:
            return None

        if len(memory_ids) == 1:
            return memory_ids[0]

        # Get all memories
        memories = []
        for mem_id in memory_ids:
            doc = self.get_memory_by_id(mem_id)
            if doc:
                memories.append(doc)

        if not memories:
            return None

        # Choose merge strategy
        if merge_strategy == "latest":
            # Use the most recent memory
            memories.sort(key=lambda d: d.metadata.get("stored_at", ""), reverse=True)
            consolidated = memories[0]
        elif merge_strategy == "highest_priority":
            # Use memory with highest priority
            memories.sort(key=lambda d: d.metadata.get("priority", 0.0), reverse=True)
            consolidated = memories[0]
        else:
            # Default to latest
            memories.sort(key=lambda d: d.metadata.get("stored_at", ""), reverse=True)
            consolidated = memories[0]

        # Create consolidated content
        # For now, just use the selected memory's content
        # In future, could use LLM to merge content
        consolidated_content = consolidated.page_content

        # Store new consolidated memory
        consolidated_id = self.store_memory_with_temporal(
            content=consolidated_content,
            metadata={
                "consolidated_from": memory_ids,
                "merge_strategy": merge_strategy,
                "consolidated_at": datetime.now().isoformat()
            },
            priority=max(m.metadata.get("priority", 1.0) for m in memories),
            supersedes=memory_ids
        )

        return consolidated_id

