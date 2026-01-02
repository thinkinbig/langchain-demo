"""
Long-term memory module for semantic memory retrieval and management.

Provides persistent memory storage with semantic search capabilities,
memory prioritization, and expiration policies.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class LongTermMemory:
    """
    Long-term memory manager with semantic retrieval.

    Stores and retrieves memories using vector embeddings for semantic search.
    Supports memory prioritization, expiration, and relevance-based retrieval.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "long_term_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
        default_ttl_days: int = 30
    ):
        """
        Initialize long-term memory manager.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the Chroma collection
            embedding_model: HuggingFace embedding model name
            default_ttl_days: Default time-to-live for memories in days
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.persist_dir = persist_dir or os.path.join(base_dir, "memory_db")
        self.collection_name = collection_name
        self.default_ttl_days = default_ttl_days

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 1.0,
        ttl_days: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Store a memory with metadata and optional expiration.

        Args:
            content: Memory content to store
            metadata: Additional metadata (e.g., source, timestamp, context)
            priority: Priority score (higher = more important, default: 1.0)
            ttl_days: Time-to-live in days (None = use default)
            tags: List of tags for categorization

        Returns:
            Memory ID
        """
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty")

        # Prepare metadata
        memory_metadata = metadata or {}
        memory_metadata.update({
            "stored_at": datetime.now().isoformat(),
            "priority": priority,
            "expires_at": (
                (datetime.now() + timedelta(days=ttl_days or self.default_ttl_days))
                .isoformat()
            ),
            "tags": ",".join(tags) if tags else "",
        })

        # Create document
        doc = Document(
            page_content=content,
            metadata=memory_metadata
        )

        # Store in vector database
        ids = self.vector_store.add_documents([doc])
        return ids[0] if ids else ""

    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        min_relevance: float = 0.5,
        filter_by_tags: Optional[List[str]] = None,
        filter_by_priority: Optional[float] = None,
        exclude_expired: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: Search query
            k: Number of memories to retrieve
            min_relevance: Minimum relevance score (0.0-1.0)
            filter_by_tags: Only retrieve memories with these tags
            filter_by_priority: Minimum priority score
            exclude_expired: Whether to exclude expired memories

        Returns:
            List of (Document, relevance_score) tuples
        """
        if not query or not query.strip():
            return []

        # Perform similarity search
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k * 2  # Retrieve more to filter
        )

        # Filter and process results
        filtered_results = []
        now = datetime.now()

        for doc, score in results:
            # Convert distance to relevance (lower distance = higher relevance)
            # Chroma uses L2 distance, so we convert to similarity score
            relevance = 1.0 / (1.0 + score) if score > 0 else 1.0

            if relevance < min_relevance:
                continue

            metadata = doc.metadata

            # Check expiration
            if exclude_expired:
                expires_at_str = metadata.get("expires_at")
                if expires_at_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if expires_at < now:
                            continue
                    except (ValueError, TypeError):
                        pass  # Invalid date, include anyway

            # Check priority filter
            if filter_by_priority is not None:
                priority = metadata.get("priority", 0.0)
                if priority < filter_by_priority:
                    continue

            # Check tag filter
            if filter_by_tags:
                doc_tags = metadata.get("tags", "").split(",")
                doc_tags = [t.strip() for t in doc_tags if t.strip()]
                if not any(tag in doc_tags for tag in filter_by_tags):
                    continue

            filtered_results.append((doc, relevance))

            if len(filtered_results) >= k:
                break

        # Sort by relevance (descending)
        filtered_results.sort(key=lambda x: x[1], reverse=True)

        return filtered_results[:k]

    def get_memory_by_id(self, memory_id: str) -> Optional[Document]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Document if found, None otherwise
        """
        try:
            results = self.vector_store.get(ids=[memory_id])
            if results and results.get("documents"):
                metadata = (
                    results.get("metadatas", [{}])[0]
                    if results.get("metadatas")
                    else {}
                )
                return Document(
                    page_content=results["documents"][0],
                    metadata=metadata
                )
        except Exception:
            pass
        return None

    def update_memory_priority(self, memory_id: str, new_priority: float) -> bool:
        """
        Update the priority of a memory.

        Args:
            memory_id: Memory ID
            new_priority: New priority score

        Returns:
            True if updated, False if not found
        """
        # Note: Chroma doesn't support direct metadata updates easily
        # This would require re-adding the document with updated metadata
        # For now, this is a placeholder for future implementation
        return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            True if deleted, False if not found
        """
        try:
            self.vector_store.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def cleanup_expired_memories(self) -> int:
        """
        Remove expired memories from storage.

        Returns:
            Number of memories deleted
        """
        # Get all memories (this is expensive, but necessary for cleanup)
        # In production, you might want to use a scheduled job
        all_docs = self.vector_store.get()

        if not all_docs or not all_docs.get("ids"):
            return 0

        expired_ids = []
        now = datetime.now()

        for i, metadata in enumerate(all_docs.get("metadatas", [])):
            expires_at_str = metadata.get("expires_at")
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if expires_at < now:
                        expired_ids.append(all_docs["ids"][i])
                except (ValueError, TypeError):
                    pass

        if expired_ids:
            self.vector_store.delete(ids=expired_ids)

        return len(expired_ids)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Returns:
            Dictionary with memory statistics
        """
        try:
            all_docs = self.vector_store.get()
            if not all_docs or not all_docs.get("ids"):
                return {
                    "total_memories": 0,
                    "expired_memories": 0,
                    "active_memories": 0
                }

            total = len(all_docs["ids"])
            now = datetime.now()
            expired = 0

            for metadata in all_docs.get("metadatas", []):
                expires_at_str = metadata.get("expires_at")
                if expires_at_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if expires_at < now:
                            expired += 1
                    except (ValueError, TypeError):
                        pass

            return {
                "total_memories": total,
                "expired_memories": expired,
                "active_memories": total - expired
            }
        except Exception:
            return {
                "total_memories": 0,
                "expired_memories": 0,
                "active_memories": 0
            }


# Global singleton instance
_memory_instance: Optional[LongTermMemory] = None


def get_long_term_memory() -> LongTermMemory:
    """
    Get or create the global long-term memory instance.

    Returns:
        LongTermMemory instance
    """
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = LongTermMemory()
    return _memory_instance


