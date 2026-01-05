"""
Storage backend for monitoring data.

Supports JSON file storage and future database backends.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings


class MonitoringStorage:
    """Storage backend for monitoring data."""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize monitoring storage.

        Args:
            storage_path: Path to store monitoring data
        """
        self.storage_path = Path(storage_path or settings.MONITORING_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.rag_dir = self.storage_path / "rag"
        self.kg_dir = self.storage_path / "kg"
        self.metrics_dir = self.storage_path / "metrics"

        for dir_path in [self.rag_dir, self.kg_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)

    def save_rag_trace(self, trace_data: Dict[str, Any]) -> str:
        """
        Save RAG trace data.

        Args:
            trace_data: Dictionary containing trace information

        Returns:
            Path to saved trace file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_id = trace_data.get("trace_id", f"rag_{timestamp}")
        filename = f"{trace_id}.json"
        filepath = self.rag_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)

        return str(filepath)

    def save_kg_trace(self, trace_data: Dict[str, Any]) -> str:
        """
        Save KG trace data.

        Args:
            trace_data: Dictionary containing trace information

        Returns:
            Path to saved trace file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_id = trace_data.get("trace_id", f"kg_{timestamp}")
        filename = f"{trace_id}.json"
        filepath = self.kg_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)

        return str(filepath)

    def save_metrics(self, metrics_data: Dict[str, Any]) -> str:
        """
        Save metrics data.

        Args:
            metrics_data: Dictionary containing metrics

        Returns:
            Path to saved metrics file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{timestamp}.json"
        filepath = self.metrics_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

        return str(filepath)

    def load_trace(self, trace_id: str, trace_type: str = "rag") -> Optional[Dict[str, Any]]:
        """
        Load trace data by ID.

        Args:
            trace_id: Trace identifier
            trace_type: Type of trace ("rag" or "kg")

        Returns:
            Trace data dictionary or None
        """
        if trace_type == "rag":
            search_dir = self.rag_dir
        elif trace_type == "kg":
            search_dir = self.kg_dir
        else:
            return None

        # Search for trace file
        for filepath in search_dir.glob(f"{trace_id}*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        return None

    def query_traces(
        self,
        trace_type: str = "rag",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        query: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query traces by criteria.

        Args:
            trace_type: Type of trace ("rag" or "kg")
            start_time: Start time filter
            end_time: End time filter
            query: Query string filter
            limit: Maximum number of results

        Returns:
            List of trace data dictionaries
        """
        if trace_type == "rag":
            search_dir = self.rag_dir
        elif trace_type == "kg":
            search_dir = self.kg_dir
        else:
            return []

        traces = []
        for filepath in sorted(search_dir.glob("*.json"), reverse=True)[:limit * 2]:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    trace_data = json.load(f)

                # Apply filters
                timestamp_str = trace_data.get("timestamp")
                if timestamp_str:
                    trace_time = datetime.fromisoformat(timestamp_str)

                    if start_time and trace_time < start_time:
                        continue
                    if end_time and trace_time > end_time:
                        continue

                if query:
                    query_lower = query.lower()
                    trace_str = json.dumps(trace_data).lower()
                    if query_lower not in trace_str:
                        continue

                traces.append(trace_data)

                if len(traces) >= limit:
                    break

            except Exception:
                continue

        return traces

