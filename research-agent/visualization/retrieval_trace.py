"""
Retrieval Trace Visualization.

Visualizes the complete retrieval path: query → entities → PPR activation → documents.
"""

import os
from typing import Any, Dict, Optional

from config import settings


class RetrievalTraceVisualizer:
    """Visualizer for RAG retrieval traces."""

    def visualize_trace(
        self,
        trace_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Visualize a single retrieval trace.

        Args:
            trace_data: Trace data dictionary from RAGMonitor
            output_path: Path to save visualization

        Returns:
            Path to generated visualization file
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            # Fallback to text-based visualization
            return self._visualize_text(trace_data, output_path)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Query & Entities", "Vector Search Results", "PPR Activation", "Final Results"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # 1. Query and entities
        query = trace_data.get("query", "")
        ppr_results = trace_data.get("ppr_results", {})
        query_entities = ppr_results.get("query_entities", [])

        if query_entities:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(query_entities))),
                    y=[1] * len(query_entities),
                    mode="markers+text",
                    text=query_entities,
                    textposition="middle right",
                    name="Query Entities",
                    marker=dict(size=15, color="red")
                ),
                row=1, col=1
            )

        # 2. Vector search results with scores
        vector_results = trace_data.get("vector_results", [])
        if vector_results:
            scores = [r.get("score", 0.0) for r in vector_results]
            sources = [r.get("source", "unknown")[:20] for r in vector_results]

            fig.add_trace(
                go.Bar(
                    x=sources,
                    y=scores,
                    name="Vector Scores",
                    marker_color="lightblue"
                ),
                row=1, col=2
            )

        # 3. PPR activation
        top_nodes = ppr_results.get("top_nodes", [])
        if top_nodes:
            node_ids = [n["node_id"][:20] for n in top_nodes]
            ppr_scores = [n["ppr_score"] for n in top_nodes]

            fig.add_trace(
                go.Scatter(
                    x=node_ids,
                    y=ppr_scores,
                    mode="markers",
                    name="PPR Scores",
                    marker=dict(size=10, color="green")
                ),
                row=2, col=1
            )

        # 4. Final results
        final_results = trace_data.get("final_results", [])
        if final_results:
            final_sources = [r.get("source", "unknown")[:20] for r in final_results]
            ranks = list(range(1, len(final_results) + 1))

            fig.add_trace(
                go.Bar(
                    x=final_sources,
                    y=ranks,
                    name="Final Ranking",
                    marker_color="orange",
                    orientation="h"
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Retrieval Trace: {query[:50]}",
            showlegend=True
        )

        # Save
        if output_path is None:
            output_dir = settings.VISUALIZATION_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            trace_id = trace_data.get("trace_id", "unknown")
            output_path = os.path.join(output_dir, f"retrieval_trace_{trace_id}.html")

        fig.write_html(output_path)
        return output_path

    def visualize_timeline(
        self,
        trace_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Visualize retrieval timeline.

        Args:
            trace_data: Trace data dictionary
            output_path: Path to save visualization

        Returns:
            Path to generated visualization file
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return self._visualize_text(trace_data, output_path)

        performance = trace_data.get("performance", {})

        stages = []
        times = []
        colors = []

        vector_time = performance.get("vector_search_time", 0.0)
        if vector_time > 0:
            stages.append("Vector Search")
            times.append(vector_time)
            colors.append("lightblue")

        ppr_time = performance.get("ppr_time", 0.0)
        if ppr_time > 0:
            stages.append("PPR Calculation")
            times.append(ppr_time)
            colors.append("lightgreen")

        rerank_time = performance.get("rerank_time", 0.0)
        if rerank_time > 0:
            stages.append("Reranking")
            times.append(rerank_time)
            colors.append("lightcoral")

        total_time = performance.get("total_time", sum(times))
        if total_time > sum(times):
            stages.append("Other")
            times.append(total_time - sum(times))
            colors.append("lightgray")

        fig = go.Figure(data=[
            go.Bar(
                x=stages,
                y=times,
                marker_color=colors,
                text=[f"{t:.3f}s" for t in times],
                textposition="auto"
            )
        ])

        fig.update_layout(
            title="Retrieval Timeline",
            xaxis_title="Stage",
            yaxis_title="Time (seconds)",
            height=400
        )

        if output_path is None:
            output_dir = settings.VISUALIZATION_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            trace_id = trace_data.get("trace_id", "unknown")
            output_path = os.path.join(output_dir, f"timeline_{trace_id}.html")

        fig.write_html(output_path)
        return output_path

    def _visualize_text(
        self,
        trace_data: Dict[str, Any],
        output_path: Optional[str]
    ) -> str:
        """Fallback text-based visualization."""
        if output_path is None:
            output_dir = settings.VISUALIZATION_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            trace_id = trace_data.get("trace_id", "unknown")
            output_path = os.path.join(output_dir, f"retrieval_trace_{trace_id}.txt")

        lines = []
        lines.append("=" * 80)
        lines.append("RETRIEVAL TRACE")
        lines.append("=" * 80)
        lines.append(f"Query: {trace_data.get('query', 'N/A')}")
        lines.append(f"Timestamp: {trace_data.get('timestamp', 'N/A')}")
        lines.append("")

        # Vector results
        vector_results = trace_data.get("vector_results", [])
        if vector_results:
            lines.append("Vector Search Results:")
            for i, result in enumerate(vector_results[:10], 1):
                lines.append(f"  {i}. {result.get('source', 'unknown')} (score: {result.get('score', 0.0):.4f})")
            lines.append("")

        # PPR results
        ppr_results = trace_data.get("ppr_results", {})
        if ppr_results:
            lines.append("PPR Retrieval:")
            lines.append(f"  Query Entities: {', '.join(ppr_results.get('query_entities', []))}")
            top_nodes = ppr_results.get("top_nodes", [])[:10]
            if top_nodes:
                lines.append("  Top Nodes:")
                for node in top_nodes:
                    lines.append(f"    - {node['node_id']} (PPR: {node['ppr_score']:.4f})")
            lines.append("")

        # Performance
        performance = trace_data.get("performance", {})
        if performance:
            lines.append("Performance:")
            lines.append(f"  Total Time: {performance.get('total_time', 0.0):.3f}s")
            lines.append(f"  Vector Search: {performance.get('vector_search_time', 0.0):.3f}s")
            if performance.get("ppr_time"):
                lines.append(f"  PPR: {performance.get('ppr_time', 0.0):.3f}s")
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

