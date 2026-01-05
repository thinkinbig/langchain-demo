"""
Knowledge Graph Visualization using Neo4j native tools.

This module provides integration with Neo4j Browser and Bloom for visualization.
For custom graph structure visualization, use Neo4j Browser directly.
"""

import os
from typing import Optional

from config import settings


class GraphVisualizer:
    """Visualizer for knowledge graphs using Neo4j native tools."""

    def __init__(self, graph_store):
        """
        Initialize graph visualizer.

        Args:
            graph_store: GraphStore instance to visualize
        """
        self.graph_store = graph_store
        self.is_neo4j = hasattr(graph_store, 'driver')  # Check if Neo4j backend

    def get_neo4j_browser_url(self) -> str:
        """
        Get Neo4j Browser URL and connection instructions.

        Neo4j Browser is accessible at http://localhost:7474 (default).

        Returns:
            Instructions string with URL and connection info
        """
        if not self.is_neo4j:
            return (
                "Currently using NetworkX backend, not Neo4j.\n"
                "To use Neo4j Browser, set in config.py:\n"
                "  GRAPH_STORE_BACKEND = 'neo4j'\n"
                "  And configure Neo4j connection information."
            )

        from config import settings
        uri = settings.NEO4J_URI

        # Extract host and port from URI
        if "://" in uri:
            _, host_port = uri.split("://", 1)
            if ":" in host_port:
                host, port = host_port.split(":", 1)
            else:
                host = host_port
                port = "7474"
        else:
            host = "localhost"
            port = "7474"

        browser_url = f"http://{host}:{port}"

        instructions = f"""
Neo4j Browser Visualization:
1. Open a browser and navigate to: {browser_url}
2. Log in with username and password
   - Username: {settings.NEO4J_USERNAME}
   - Password: {settings.NEO4J_PASSWORD}
3. Run Cypher queries in the query box to visualize the knowledge graph

Tips:
- Use Neo4j Browser for interactive graph exploration
- Supports node filtering, relationship expansion, and style customization
- For advanced visualization, use Neo4j Bloom
        """

        return instructions.strip()

    def export_instructions(
        self,
        output_path: Optional[str] = None,
        include_sample_queries: bool = True
    ) -> str:
        """
        Export Neo4j Browser usage instructions and sample queries.

        Args:
            output_path: Path to save instructions file
            include_sample_queries: Whether to include sample Cypher queries

        Returns:
            Path to saved instructions file
        """
        if output_path is None:
            output_dir = settings.VISUALIZATION_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "neo4j_browser_instructions.txt")

        instructions = []
        instructions.append("=" * 80)
        instructions.append("Neo4j Browser Visualization Guide")
        instructions.append("=" * 80)
        instructions.append("")
        instructions.append(self.get_neo4j_browser_url())
        instructions.append("")

        if include_sample_queries:
            instructions.append("Sample Cypher Queries:")
            instructions.append("-" * 80)
            instructions.append("")

            # Basic query
            instructions.append("1. View all nodes and relationships:")
            instructions.append("   MATCH (n:Node)-[r:RELATED]->(m:Node)")
            instructions.append("   RETURN n, r, m")
            instructions.append("   LIMIT 100")
            instructions.append("")

            # Query by type
            instructions.append("2. View nodes of a specific type (e.g., Paper):")
            instructions.append("   MATCH (n:Node)")
            instructions.append("   WHERE n.type = 'Paper'")
            instructions.append("   RETURN n")
            instructions.append("")

            # View node details
            instructions.append("3. View node details:")
            instructions.append("   MATCH (n:Node {id: 'YourNodeID'})")
            instructions.append("   RETURN n")
            instructions.append("")

            # View neighbors
            instructions.append("4. View node neighbors:")
            instructions.append("   MATCH (n:Node {id: 'YourNodeID'})-[r:RELATED]-(neighbor:Node)")
            instructions.append("   RETURN n, r, neighbor")
            instructions.append("")

            # Community detection
            instructions.append("5. Community detection (requires GDS plugin):")
            instructions.append("   CALL gds.graph.project('kg-graph', 'Node', 'RELATED')")
            instructions.append("   YIELD graphName")
            instructions.append("   CALL gds.louvain.stream('kg-graph')")
            instructions.append("   YIELD nodeId, communityId")
            instructions.append("   RETURN gds.util.asNode(nodeId).id AS node, communityId")
            instructions.append("   ORDER BY communityId")
            instructions.append("")

            # PPR related
            instructions.append("6. View document-node mappings:")
            instructions.append("   MATCH (n:Node)-[:APPEARS_IN]->(d:Document)")
            instructions.append("   RETURN n, d")
            instructions.append("   LIMIT 50")
            instructions.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(instructions))

        return output_path

