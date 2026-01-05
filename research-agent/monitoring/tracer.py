"""
Unified Tracer Interface.

Integrates RAG and KG monitoring with LangSmith support.
"""

from typing import Any, Dict, List, Optional

try:
    from monitoring.kg_monitor import KGMonitor
    from monitoring.rag_monitor import RAGMonitor
    from monitoring.storage import MonitoringStorage
except ImportError:
    # Handle case where monitoring modules might not be available
    KGMonitor = None
    RAGMonitor = None
    MonitoringStorage = None


class UnifiedTracer:
    """Unified interface for RAG and KG tracing."""

    def __init__(
        self,
        storage: Optional[MonitoringStorage] = None,
        enable_langsmith: bool = False
    ):
        """
        Initialize unified tracer.

        Args:
            storage: Monitoring storage backend
            enable_langsmith: Whether to enable LangSmith integration
        """
        self.storage = storage or MonitoringStorage()
        self.rag_monitor = RAGMonitor(self.storage)
        self.kg_monitor = KGMonitor(self.storage)
        self.enable_langsmith = enable_langsmith

        # Check for LangSmith
        if enable_langsmith:
            try:
                import os
                if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
                    self.langsmith_enabled = True
                else:
                    self.langsmith_enabled = False
            except Exception:
                self.langsmith_enabled = False
        else:
            self.langsmith_enabled = False

    def trace_rag_retrieval(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> "RAGTraceContext":
        """
        Start tracing a RAG retrieval operation.

        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional parameters

        Returns:
            RAGTraceContext for managing the trace
        """
        trace_id = self.rag_monitor.start_trace(query, k)
        return RAGTraceContext(self.rag_monitor, trace_id)

    def trace_kg_operation(
        self,
        operation: str,
        **kwargs
    ) -> "KGTraceContext":
        """
        Start tracing a KG operation.

        Args:
            operation: Operation type
            **kwargs: Additional parameters

        Returns:
            KGTraceContext for managing the trace
        """
        trace_id = self.kg_monitor.start_trace(operation, **kwargs)
        return KGTraceContext(self.kg_monitor, trace_id)

    def query_rag_traces(
        self,
        query: Optional[str] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query RAG traces."""
        return self.rag_monitor.query_traces(
            query=query,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    def query_kg_traces(
        self,
        operation: Optional[str] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query KG traces."""
        return self.kg_monitor.query_traces(
            operation=operation,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    def get_rag_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get RAG trace by ID."""
        return self.rag_monitor.get_trace(trace_id)

    def get_kg_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get KG trace by ID."""
        return self.kg_monitor.get_trace(trace_id)


class RAGTraceContext:
    """Context manager for RAG traces."""

    def __init__(self, monitor: RAGMonitor, trace_id: str):
        self.monitor = monitor
        self.trace_id = trace_id

    def __enter__(self):
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_trace()
        return False


class KGTraceContext:
    """Context manager for KG traces."""

    def __init__(self, monitor: KGMonitor, trace_id: str):
        self.monitor = monitor
        self.trace_id = trace_id

    def __enter__(self):
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_trace()
        return False


# Global tracer instance
_tracer: Optional[UnifiedTracer] = None


def get_tracer() -> UnifiedTracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        from config import settings
        _tracer = UnifiedTracer(
            enable_langsmith=settings.ENABLE_MONITORING
        )
    return _tracer

