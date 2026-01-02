"""
Background cleanup tasks for memory management.

Automatically cleans up expired memories and consolidates similar entries.
"""

import asyncio
from typing import Optional

from memory.long_term_memory import LongTermMemory
from memory.temporal_memory import TemporalMemory


async def cleanup_expired_memories(memory: Optional[LongTermMemory] = None) -> int:
    """
    Clean up expired memories from storage.

    Args:
        memory: LongTermMemory instance (creates new if None)

    Returns:
        Number of memories cleaned up
    """
    if memory is None:
        memory = LongTermMemory()

    try:
        count = memory.cleanup_expired_memories()
        return count
    except Exception as e:
        print(f"Error cleaning up expired memories: {e}")
        return 0


async def consolidate_similar_memories(
    memory: Optional[TemporalMemory] = None,
    similarity_threshold: float = 0.9
) -> int:
    """
    Consolidate similar memories to reduce redundancy.

    Args:
        memory: TemporalMemory instance (creates new if None)
        similarity_threshold: Minimum similarity to consolidate (0.0-1.0)

    Returns:
        Number of memories consolidated
    """
    if memory is None:
        memory = TemporalMemory()

    try:
        # Get all memories
        all_memories = memory.vector_store.get()
        if not all_memories or not all_memories.get("ids"):
            return 0

        consolidated_count = 0
        processed_ids = set()

        # Use ChromaDB's query method directly to get IDs with similarity scores
        # This allows us to find semantically similar memories with their IDs
        for i, doc_content in enumerate(all_memories.get("documents", [])):
            memory_id = all_memories["ids"][i]
            if memory_id in processed_ids:
                continue

            # Use ChromaDB's query method to get IDs directly
            # This performs semantic search and returns IDs, not just Documents
            try:
                # Get embedding for the current memory content
                query_embedding = memory.embeddings.embed_query(doc_content)

                # Query ChromaDB directly to get IDs with distances
                query_results = memory.vector_store._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,  # Get more candidates
                    include=["metadatas", "documents", "distances"]
                )

                # Extract IDs and convert distances to relevance scores
                similar_ids = []
                if query_results and query_results.get("ids"):
                    result_ids = query_results["ids"][0]  # First query result
                    distances = query_results.get("distances", [[]])[0]

                    for result_id, distance in zip(
                        result_ids, distances, strict=False
                    ):
                        # Convert distance to relevance
                        # (lower distance = higher relevance)
                        relevance = (
                            1.0 / (1.0 + distance) if distance > 0 else 1.0
                        )

                        # Only include if relevance meets threshold
                        if (
                            relevance >= similarity_threshold
                            and result_id != memory_id
                            and result_id not in processed_ids
                        ):
                            similar_ids.append(result_id)

                # If we found similar memories, consolidate them
                if similar_ids:
                    try:
                        # Include the current memory in the consolidation
                        all_to_consolidate = [memory_id] + similar_ids
                        memory.consolidate_memories(
                            all_to_consolidate, merge_strategy="latest"
                        )
                        consolidated_count += len(similar_ids)
                        # Mark all as processed
                        processed_ids.add(memory_id)
                        processed_ids.update(similar_ids)
                    except Exception:
                        pass  # Skip if consolidation fails

            except Exception as e:
                # Fallback: if direct query fails, skip this memory
                print(f"  ⚠️  Failed to query similar memories for {memory_id[:8]}: {e}")
                continue

        return consolidated_count
    except Exception as e:
        print(f"Error consolidating memories: {e}")
        return 0


async def run_cleanup_tasks(
    cleanup_expired: bool = True,
    consolidate: bool = True,
    interval_seconds: int = 3600
):
    """
    Run cleanup tasks periodically in the background.

    Args:
        cleanup_expired: Whether to clean up expired memories
        consolidate: Whether to consolidate similar memories
        interval_seconds: Interval between cleanup runs (default: 1 hour)
    """
    while True:
        try:
            if cleanup_expired:
                count = await cleanup_expired_memories()
                if count > 0:
                    print(f"🧹 Cleaned up {count} expired memories")

            if consolidate:
                count = await consolidate_similar_memories()
                if count > 0:
                    print(f"🔄 Consolidated {count} similar memories")

        except Exception as e:
            print(f"Error in cleanup task: {e}")

        await asyncio.sleep(interval_seconds)

