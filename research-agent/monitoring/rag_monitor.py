"""
RAG Monitoring Module.

Tracks queries, retrieval results, similarity scores, and performance metrics.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from monitoring.storage import MonitoringStorage


class RAGMonitor:
    """Monitor for RAG operations."""

    def __init__(self, storage: Optional[MonitoringStorage] = None):
        """
        Initialize RAG monitor.

        Args:
            storage: Monitoring storage backend
        """
        self.storage = storage or MonitoringStorage()
        self.current_trace: Optional[Dict[str, Any]] = None

    def start_trace(self, query: str, k: int = 4) -> str:
        """
        Start a new retrieval trace.

        Args:
            query: Search query
            k: Number of results requested

        Returns:
            Trace ID
        """
        import uuid
        trace_id = str(uuid.uuid4())

        self.current_trace = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "k": k,
            "start_time": time.time(),
            "vector_results": [],
            "ppr_results": None,
            "final_results": [],
            "performance": {}
        }

        return trace_id

    def record_vector_search(
        self,
        results: List[Document],
        scores: Optional[List[float]] = None,
        method: str = "similarity_search"
    ):
        """
        Record vector search results.

        Args:
            results: Retrieved documents
            scores: Similarity scores (if available)
            method: Search method used
        """
        if not self.current_trace:
            return

        vector_results = []
        for i, doc in enumerate(results):
            result_data = {
                "rank": i + 1,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "metadata": doc.metadata
            }

            if scores and i < len(scores):
                result_data["score"] = scores[i]
                result_data["distance"] = scores[i]  # For compatibility

            vector_results.append(result_data)

        self.current_trace["vector_results"] = vector_results
        self.current_trace["vector_method"] = method

    def record_ppr_retrieval(
        self,
        query_entities: List[str],
        top_nodes: List[tuple],
        document_sources: List[str],
        ppr_context: str
    ):
        """
        Record PPR-based graph retrieval results.

        Args:
            query_entities: Extracted entities from query
            top_nodes: List of (node_id, ppr_score) tuples
            document_sources: Document sources retrieved via PPR
            ppr_context: Formatted PPR context string
        """
        if not self.current_trace:
            return

        self.current_trace["ppr_results"] = {
            "query_entities": query_entities,
            "top_nodes": [
                {"node_id": node_id, "ppr_score": score}
                for node_id, score in top_nodes
            ],
            "document_sources": document_sources,
            "context_preview": ppr_context[:500] + "..." if len(ppr_context) > 500 else ppr_context
        }

    def record_final_results(
        self,
        results: List[Document],
        sources: List[str],
        context_str: str
    ):
        """
        Record final merged and reranked results.

        Args:
            results: Final document results
            sources: Source identifiers
            context_str: Final context string
        """
        if not self.current_trace:
            return

        self.current_trace["final_results"] = [
            {
                "rank": i + 1,
                "source": doc.metadata.get("source", "unknown"),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for i, doc in enumerate(results)
        ]
        self.current_trace["final_sources"] = sources
        self.current_trace["context_length"] = len(context_str)

    def record_performance(
        self,
        vector_search_time: float,
        ppr_time: Optional[float] = None,
        rerank_time: Optional[float] = None,
        total_time: Optional[float] = None
    ):
        """
        Record performance metrics.

        Args:
            vector_search_time: Time for vector search (seconds)
            ppr_time: Time for PPR retrieval (seconds)
            rerank_time: Time for reranking (seconds)
            total_time: Total retrieval time (seconds)
        """
        if not self.current_trace:
            return

        if total_time is None:
            total_time = time.time() - self.current_trace["start_time"]

        self.current_trace["performance"] = {
            "vector_search_time": vector_search_time,
            "ppr_time": ppr_time,
            "rerank_time": rerank_time,
            "total_time": total_time
        }

    def end_trace(self) -> Optional[str]:
        """
        End current trace and save to storage.

        Returns:
            Path to saved trace file, or None if no trace
        """
        if not self.current_trace:
            return None

        # Calculate total time if not set
        if "total_time" not in self.current_trace.get("performance", {}):
            total_time = time.time() - self.current_trace["start_time"]
            self.current_trace["performance"]["total_time"] = total_time

        # Save trace
        filepath = self.storage.save_rag_trace(self.current_trace)

        # Clear current trace
        trace_id = self.current_trace["trace_id"]
        self.current_trace = None

        return filepath

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trace by ID.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace data or None
        """
        return self.storage.load_trace(trace_id, trace_type="rag")

    def query_traces(
        self,
        query: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query traces by criteria.

        Args:
            query: Query string to search for
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results

        Returns:
            List of trace data
        """
        return self.storage.query_traces(
            trace_type="rag",
            query=query,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

