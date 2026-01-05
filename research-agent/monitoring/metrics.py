"""
Performance Metrics Collection.

Collects and aggregates performance metrics for RAG and KG operations.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from monitoring.storage import MonitoringStorage


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, storage: Optional[MonitoringStorage] = None):
        """
        Initialize metrics collector.

        Args:
            storage: Monitoring storage backend
        """
        self.storage = storage or MonitoringStorage()

    def collect_rag_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Collect aggregated RAG metrics.

        Args:
            start_time: Start time for metrics collection
            end_time: End time for metrics collection

        Returns:
            Dictionary of aggregated metrics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=1)

        traces = self.storage.query_traces(
            trace_type="rag",
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )

        if not traces:
            return {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "p50_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0,
                "avg_results_count": 0.0,
                "ppr_usage_rate": 0.0
            }

        response_times = []
        results_counts = []
        ppr_used_count = 0

        for trace in traces:
            perf = trace.get("performance", {})
            total_time = perf.get("total_time", 0.0)
            if total_time > 0:
                response_times.append(total_time)

            final_results = trace.get("final_results", [])
            results_counts.append(len(final_results))

            if trace.get("ppr_results"):
                ppr_used_count += 1

        # Calculate percentiles
        response_times_sorted = sorted(response_times) if response_times else [0.0]
        n = len(response_times_sorted)

        metrics = {
            "total_queries": len(traces),
            "avg_response_time": statistics.mean(response_times) if response_times else 0.0,
            "p50_response_time": response_times_sorted[n // 2] if n > 0 else 0.0,
            "p95_response_time": response_times_sorted[int(n * 0.95)] if n > 0 else 0.0,
            "p99_response_time": response_times_sorted[int(n * 0.99)] if n > 0 else 0.0,
            "avg_results_count": statistics.mean(results_counts) if results_counts else 0.0,
            "ppr_usage_rate": ppr_used_count / len(traces) if traces else 0.0,
            "min_response_time": min(response_times) if response_times else 0.0,
            "max_response_time": max(response_times) if response_times else 0.0
        }

        return metrics

    def collect_kg_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Collect aggregated KG metrics.

        Args:
            start_time: Start time for metrics collection
            end_time: End time for metrics collection

        Returns:
            Dictionary of aggregated metrics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=1)

        traces = self.storage.query_traces(
            trace_type="kg",
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )

        if not traces:
            return {
                "total_operations": 0,
                "avg_operation_time": 0.0,
                "total_nodes_added": 0,
                "total_edges_added": 0,
                "ppr_calculations": 0
            }

        operation_times = []
        total_nodes = 0
        total_edges = 0
        ppr_count = 0
        operations_by_type = defaultdict(int)

        for trace in traces:
            perf = trace.get("performance", {})
            op_time = perf.get("operation_time", 0.0)
            if op_time > 0:
                operation_times.append(op_time)

            nodes_added = trace.get("nodes_added", [])
            edges_added = trace.get("edges_added", [])
            total_nodes += len(nodes_added)
            total_edges += len(edges_added)

            if trace.get("ppr_calculation"):
                ppr_count += 1

            operation = trace.get("operation", "unknown")
            operations_by_type[operation] += 1

        metrics = {
            "total_operations": len(traces),
            "avg_operation_time": statistics.mean(operation_times) if operation_times else 0.0,
            "total_nodes_added": total_nodes,
            "total_edges_added": total_edges,
            "ppr_calculations": ppr_count,
            "operations_by_type": dict(operations_by_type),
            "min_operation_time": min(operation_times) if operation_times else 0.0,
            "max_operation_time": max(operation_times) if operation_times else 0.0
        }

        return metrics

    def save_metrics_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """
        Generate and save a comprehensive metrics report.

        Args:
            start_time: Start time for report
            end_time: End time for report

        Returns:
            Path to saved report
        """
        rag_metrics = self.collect_rag_metrics(start_time, end_time)
        kg_metrics = self.collect_kg_metrics(start_time, end_time)

        report = {
            "timestamp": datetime.now().isoformat(),
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            "rag_metrics": rag_metrics,
            "kg_metrics": kg_metrics
        }

        return self.storage.save_metrics(report)

