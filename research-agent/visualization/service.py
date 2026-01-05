"""
Visualization Service.

Unified API for generating visualizations.
"""

import os
from typing import Any, Dict, List, Optional

from config import settings
from visualization.graph_viz import GraphVisualizer
from visualization.retrieval_trace import RetrievalTraceVisualizer


class VisualizationService:
    """Unified service for generating visualizations."""

    def __init__(self):
        """Initialize visualization service."""
        self.output_dir = settings.VISUALIZATION_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_graph(
        self,
        graph_store,
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate Neo4j Browser instructions for graph visualization.

        Args:
            graph_store: GraphStore instance
            output_path: Output file path for instructions
            **kwargs: Additional options (ignored, kept for compatibility)

        Returns:
            Path to saved instructions file
        """
        visualizer = GraphVisualizer(graph_store)

        if output_path is None:
            output_path = os.path.join(self.output_dir, "neo4j_browser_instructions.txt")

        return visualizer.export_instructions(output_path)

    def visualize_retrieval_trace(
        self,
        trace_data: Dict[str, Any],
        output_path: Optional[str] = None,
        include_timeline: bool = True
    ) -> List[str]:
        """
        Generate retrieval trace visualization.

        Args:
            trace_data: Trace data from RAGMonitor
            output_path: Base output path (without extension)
            include_timeline: Whether to also generate timeline visualization

        Returns:
            List of paths to generated visualizations
        """
        visualizer = RetrievalTraceVisualizer()

        output_paths = []

        # Main trace visualization
        if output_path is None:
            trace_id = trace_data.get("trace_id", "unknown")
            output_path = os.path.join(self.output_dir, f"retrieval_trace_{trace_id}")

        trace_path = visualizer.visualize_trace(trace_data, f"{output_path}.html")
        output_paths.append(trace_path)

        # Timeline visualization
        if include_timeline:
            timeline_path = visualizer.visualize_timeline(
                trace_data,
                f"{output_path}_timeline.html"
            )
            output_paths.append(timeline_path)

        return output_paths

    def visualize_kg_operation(
        self,
        trace_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate KG operation visualization.

        Args:
            trace_data: Trace data from KGMonitor
            output_path: Output file path

        Returns:
            Path to generated visualization
        """
        # Simple text-based visualization for KG operations
        if output_path is None:
            trace_id = trace_data.get("trace_id", "unknown")
            output_path = os.path.join(self.output_dir, f"kg_operation_{trace_id}.txt")

        lines = []
        lines.append("=" * 80)
        lines.append("KG OPERATION TRACE")
        lines.append("=" * 80)
        lines.append(f"Operation: {trace_data.get('operation', 'N/A')}")
        lines.append(f"Timestamp: {trace_data.get('timestamp', 'N/A')}")
        lines.append("")

        # Nodes added
        nodes_added = trace_data.get("nodes_added", [])
        if nodes_added:
            lines.append(f"Nodes Added: {len(nodes_added)}")
            for node in nodes_added[:20]:  # Limit display
                lines.append(f"  - {node.get('id', 'unknown')} ({node.get('type', 'Unknown')})")
            lines.append("")

        # Edges added
        edges_added = trace_data.get("edges_added", [])
        if edges_added:
            lines.append(f"Edges Added: {len(edges_added)}")
            for edge in edges_added[:20]:  # Limit display
                lines.append(
                    f"  - {edge.get('source', 'unknown')} --[{edge.get('relation', 'related_to')}]--> "
                    f"{edge.get('target', 'unknown')}"
                )
            lines.append("")

        # PPR calculation
        ppr_calc = trace_data.get("ppr_calculation", {})
        if ppr_calc:
            lines.append("PPR Calculation:")
            lines.append(f"  Query: {ppr_calc.get('query', 'N/A')}")
            lines.append(f"  Seed Nodes: {', '.join(ppr_calc.get('seed_nodes', []))}")
            lines.append(f"  Calculation Time: {ppr_calc.get('calculation_time', 0.0):.3f}s")
            top_nodes = ppr_calc.get("top_nodes", [])[:10]
            if top_nodes:
                lines.append("  Top Nodes:")
                for node in top_nodes:
                    lines.append(f"    - {node['node_id']} (PPR: {node['ppr_score']:.4f})")
            lines.append("")

        # Performance
        performance = trace_data.get("performance", {})
        if performance:
            lines.append("Performance:")
            lines.append(f"  Operation Time: {performance.get('operation_time', 0.0):.3f}s")
            lines.append(f"  Nodes Processed: {performance.get('nodes_processed', 0)}")
            lines.append(f"  Edges Processed: {performance.get('edges_processed', 0)}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path


# Global service instance
_service: Optional[VisualizationService] = None


def get_visualization_service() -> VisualizationService:
    """Get or create the global visualization service."""
    global _service
    if _service is None:
        _service = VisualizationService()
    return _service

