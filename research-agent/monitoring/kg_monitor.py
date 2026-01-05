"""
KG Monitoring Module.

Tracks entity extraction, graph updates, PPR calculations, and node-document mappings.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from monitoring.storage import MonitoringStorage


class KGMonitor:
    """Monitor for Knowledge Graph operations."""

    def __init__(self, storage: Optional[MonitoringStorage] = None):
        """
        Initialize KG monitor.

        Args:
            storage: Monitoring storage backend
        """
        self.storage = storage or MonitoringStorage()
        self.current_trace: Optional[Dict[str, Any]] = None

    def start_trace(self, operation: str, **kwargs) -> str:
        """
        Start a new KG operation trace.

        Args:
            operation: Operation type ("index", "ppr", "entity_extraction", etc.)
            **kwargs: Additional operation-specific parameters

        Returns:
            Trace ID
        """
        import uuid
        trace_id = str(uuid.uuid4())

        self.current_trace = {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "start_time": time.time(),
            "parameters": kwargs,
            "nodes_added": [],
            "edges_added": [],
            "entities_extracted": [],
            "ppr_calculation": None,
            "performance": {}
        }

        return trace_id

    def record_entity_extraction(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        extraction_time: float
    ):
        """
        Record entity extraction operation.

        Args:
            text: Input text
            entities: Extracted entities (nodes and edges)
            extraction_time: Time taken for extraction
        """
        if not self.current_trace:
            return

        self.current_trace["entities_extracted"] = {
            "text_preview": text[:500] + "..." if len(text) > 500 else text,
            "nodes": entities.get("nodes", []),
            "edges": entities.get("edges", []),
            "extraction_time": extraction_time
        }

    def record_graph_update(
        self,
        nodes_added: List[Dict[str, Any]],
        edges_added: List[Dict[str, Any]],
        document_source: str
    ):
        """
        Record graph update operation.

        Args:
            nodes_added: List of added nodes
            edges_added: List of added edges
            document_source: Source document identifier
        """
        if not self.current_trace:
            return

        self.current_trace["nodes_added"] = nodes_added
        self.current_trace["edges_added"] = edges_added
        self.current_trace["document_source"] = document_source

    def record_ppr_calculation(
        self,
        query: str,
        seed_nodes: List[str],
        top_nodes: List[tuple],
        ppr_scores: Dict[str, float],
        calculation_time: float,
        alpha: float = 0.85
    ):
        """
        Record PPR calculation.

        Args:
            query: Original query
            seed_nodes: Seed nodes for PPR
            top_nodes: Top nodes by PPR score
            ppr_scores: Full PPR score dictionary
            calculation_time: Time taken for calculation
            alpha: Damping factor used
        """
        if not self.current_trace:
            return

        self.current_trace["ppr_calculation"] = {
            "query": query,
            "seed_nodes": seed_nodes,
            "top_nodes": [
                {"node_id": node_id, "ppr_score": score}
                for node_id, score in top_nodes[:20]  # Limit to top 20
            ],
            "ppr_scores_count": len(ppr_scores),
            "calculation_time": calculation_time,
            "alpha": alpha
        }

    def record_node_activation_path(
        self,
        seed_nodes: List[str],
        activation_paths: List[Dict[str, Any]]
    ):
        """
        Record node activation paths from PPR.

        Args:
            seed_nodes: Starting seed nodes
            activation_paths: List of activation paths with scores
        """
        if not self.current_trace:
            return

        if "ppr_calculation" not in self.current_trace:
            self.current_trace["ppr_calculation"] = {}

        self.current_trace["ppr_calculation"]["activation_paths"] = activation_paths

    def record_performance(
        self,
        operation_time: float,
        nodes_processed: int = 0,
        edges_processed: int = 0
    ):
        """
        Record performance metrics.

        Args:
            operation_time: Time taken for operation
            nodes_processed: Number of nodes processed
            edges_processed: Number of edges processed
        """
        if not self.current_trace:
            return

        self.current_trace["performance"] = {
            "operation_time": operation_time,
            "nodes_processed": nodes_processed,
            "edges_processed": edges_processed
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
        if "operation_time" not in self.current_trace.get("performance", {}):
            total_time = time.time() - self.current_trace["start_time"]
            self.current_trace["performance"]["operation_time"] = total_time

        # Save trace
        filepath = self.storage.save_kg_trace(self.current_trace)

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
        return self.storage.load_trace(trace_id, trace_type="kg")

    def query_traces(
        self,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query traces by criteria.

        Args:
            operation: Operation type filter
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results

        Returns:
            List of trace data
        """
        traces = self.storage.query_traces(
            trace_type="kg",
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        # Filter by operation if specified
        if operation:
            traces = [t for t in traces if t.get("operation") == operation]

        return traces

