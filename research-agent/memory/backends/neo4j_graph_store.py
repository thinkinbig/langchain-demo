"""
Neo4j Graph Store Implementation.

Uses Neo4j as the graph database backend for knowledge graph operations.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from memory.graph_store import GraphStore


class Neo4jGraphStore(GraphStore):
    """Neo4j implementation of GraphStore."""

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j graph store.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4jGraphStore. "
                "Install it with: pip install neo4j"
            )

        from config import settings

        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # Create constraints and indexes
        self._create_constraints()

    def _create_constraints(self):
        """Create constraints and indexes for graph nodes."""
        with self.driver.session(database=self.database) as session:
            # Create unique constraint on node ID
            session.run("""
                CREATE CONSTRAINT node_id_unique IF NOT EXISTS
                FOR (n:Node) REQUIRE n.id IS UNIQUE
            """)

            # Create index on node type for faster queries
            session.run("""
                CREATE INDEX node_type_index IF NOT EXISTS
                FOR (n:Node) ON (n.type)
            """)

    def add_node(self, node_id: str, node_type: str, description: str = "") -> None:
        """Add a node to the graph."""
        if not node_id:
            return

        node_id = node_id.strip()

        with self.driver.session(database=self.database) as session:
            session.run("""
                MERGE (n:Node {id: $node_id})
                SET n.type = $node_type,
                    n.description = $description
            """,
                node_id=node_id,
                node_type=node_type,
                description=description
            )

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an edge between two nodes."""
        if not source or not target:
            return

        source = source.strip()
        target = target.strip()

        # Ensure nodes exist
        self.add_node(source, "Unknown", "")
        self.add_node(target, "Unknown", "")

        # Normalize relation type for Neo4j (must be valid identifier)
        # Replace invalid characters and ensure it's uppercase
        relation_type = relation.upper().replace("-", "_").replace(" ", "_")
        # Remove any characters that aren't valid for Neo4j relationship types
        relation_type = re.sub(r"[^A-Z0-9_]", "", relation_type)
        if not relation_type:
            relation_type = "RELATED_TO"

        # Prepare edge properties (store original relation name and any additional properties)
        edge_props = {}
        if properties:
            edge_props.update(properties)
        # Store original relation name in properties for reference
        if relation != relation_type.lower().replace("_", " "):
            edge_props["original_relation"] = relation

        with self.driver.session(database=self.database) as session:
            # Convert properties to JSON string for storage
            props_json = json.dumps(edge_props) if edge_props else "{}"

            # Use dynamic relationship type
            # Neo4j allows dynamic relationship types in Cypher
            query = f"""
                MATCH (source:Node {{id: $source}})
                MATCH (target:Node {{id: $target}})
                MERGE (source)-[r:`{relation_type}`]->(target)
                SET r.properties = $props_json
            """
            session.run(
                query,
                source=source,
                target=target,
                props_json=props_json
            )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information by ID."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Node {id: $node_id})
                RETURN n.id AS id, n.type AS type, n.description AS description
            """, node_id=node_id)

            record = result.single()
            if record:
                return {
                    "id": record["id"],
                    "type": record["type"],
                    "description": record.get("description", "")
                }
        return None

    def get_neighborhood(self, node_id: str, k: int = 1) -> List[Tuple[str, str, str]]:
        """Get neighbors of a node up to k hops away."""
        if not self.get_node(node_id):
            return []

        neighbors = []
        visited = set()

        with self.driver.session(database=self.database) as session:
            # Use BFS-like approach with path queries
            for depth in range(1, k + 1):
                result = session.run("""
                    MATCH path = (start:Node {id: $node_id})-[*1..$depth]-(neighbor:Node)
                    WHERE NOT neighbor.id IN $visited
                    RETURN DISTINCT 
                        start.id AS source,
                        type(last(relationships(path))) AS relation,
                        neighbor.id AS target
                    LIMIT 100
                """,
                    node_id=node_id,
                    depth=depth,
                    visited=list(visited)
                )

                for record in result:
                    source = record["source"]
                    # Get relation type name and convert back to lowercase
                    relation_type = record.get("relation", "RELATED_TO")
                    relation = relation_type.lower().replace("_", " ")
                    target = record["target"]

                    if target not in visited:
                        neighbors.append((source, relation, target))
                        visited.add(target)

        return neighbors

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for nodes by fuzzy matching on ID or description."""
        if not query:
            return []

        query_lower = query.lower()
        matches = []

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Node)
                WHERE toLower(n.id) CONTAINS $query 
                   OR toLower(n.description) CONTAINS $query
                   OR toLower(n.type) CONTAINS $query
                RETURN n.id AS id, n.type AS type, n.description AS description
                LIMIT $limit
            """,
                query=query_lower,
                limit=limit
            )

            for record in result:
                score = 0
                node_id_lower = record["id"].lower()
                desc_lower = (record.get("description") or "").lower()
                type_lower = (record.get("type") or "").lower()

                # Calculate simple relevance score
                if query_lower in node_id_lower:
                    score += 2
                if query_lower in desc_lower:
                    score += 1
                if query_lower in type_lower:
                    score += 0.5

                matches.append((score, {
                    "id": record["id"],
                    "type": record.get("type", "Unknown"),
                    "description": record.get("description", "")
                }))

        # Sort by score
        matches.sort(key=lambda x: x[0], reverse=True)
        return [match[1] for match in matches[:limit]]

    def save(self) -> None:
        """Persist graph to disk (Neo4j handles persistence automatically)."""
        # Neo4j persists automatically, but we can ensure a checkpoint
        pass

    def load(self) -> None:
        """Load graph from disk (Neo4j loads automatically on connection)."""
        # Neo4j loads automatically on connection
        pass

    def get_communities(self) -> List[List[str]]:
        """Detect communities in the graph using Neo4j's community detection."""
        with self.driver.session(database=self.database) as session:
            # Use Neo4j's GDS library for community detection
            # First check if GDS is available
            try:
                # Try to use GDS Louvain algorithm
                # Match all relationship types, not just RELATED
                result = session.run("""
                    CALL gds.graph.project(
                        'kg-graph',
                        'Node',
                        {
                            '*': {
                                orientation: 'UNDIRECTED'
                            }
                        }
                    )
                    YIELD graphName
                    RETURN graphName
                """)

                # Run Louvain community detection
                communities_result = session.run("""
                    CALL gds.louvain.stream('kg-graph')
                    YIELD nodeId, communityId
                    RETURN communityId, collect(gds.util.asNode(nodeId).id) AS nodes
                """)

                communities = []
                for record in communities_result:
                    communities.append(record["nodes"])

                # Drop the projected graph
                session.run("CALL gds.graph.drop('kg-graph')")

                return communities
            except Exception:
                # Fallback: simple connected components
                # Match all relationship types
                result = session.run("""
                    CALL gds.wcc.stream({
                        nodeQuery: 'MATCH (n:Node) RETURN id(n) AS id',
                        relationshipQuery: 'MATCH (n:Node)-[r]-(m:Node) RETURN id(n) AS source, id(m) AS target'
                    })
                    YIELD nodeId, componentId
                    RETURN componentId, collect(gds.util.asNode(nodeId).id) AS nodes
                """)

                communities = []
                for record in result:
                    communities.append(record["nodes"])

                return communities

    # Additional methods for node-document mappings
    def add_node_document_mapping(
        self,
        node_id: str,
        document_source: str
    ) -> None:
        """Add mapping between a node and a document."""
        if not node_id or not document_source:
            return

        with self.driver.session(database=self.database) as session:
            # Create Document node if it doesn't exist (for backward compatibility)
            session.run("""
                MERGE (d:Document {source: $document_source})
            """, document_source=document_source)

            # Create relationship
            session.run("""
                MATCH (n:Node {id: $node_id})
                MATCH (d:Document {source: $document_source})
                MERGE (n)-[:APPEARS_IN]->(d)
            """,
                node_id=node_id,
                document_source=document_source
            )

    def link_node_to_document_chunk(
        self,
        node_id: str,
        document_id: str
    ) -> None:
        """
        Link a knowledge graph node to a specific document chunk (vector store).

        Args:
            node_id: ID of the knowledge graph node
            document_id: ID of the document chunk in vector store
        """
        if not node_id or not document_id:
            return

        with self.driver.session(database=self.database) as session:
            # Link Node to Document chunk (vector store)
            session.run("""
                MATCH (n:Node {id: $node_id})
                MATCH (d:Document {id: $document_id})
                MERGE (n)-[:MENTIONED_IN]->(d)
            """,
                node_id=node_id,
                document_id=document_id
            )

    def get_documents_for_nodes(
        self,
        node_ids: List[str]
    ) -> Set[str]:
        """Get all documents that contain any of the given nodes."""
        if not node_ids:
            return set()

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Node)-[:APPEARS_IN]->(d:Document)
                WHERE n.id IN $node_ids
                RETURN DISTINCT d.source AS source
            """, node_ids=node_ids)

            documents = set()
            for record in result:
                documents.add(record["source"])

            return documents

    def personalized_pagerank(
        self,
        seed_nodes: List[str],
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """Compute Personalized PageRank using Neo4j GDS."""
        if not seed_nodes:
            return {}

        # Filter to valid nodes
        valid_seeds = []
        for node_id in seed_nodes:
            if self.get_node(node_id):
                valid_seeds.append(node_id)

        if not valid_seeds:
            return {}

        with self.driver.session(database=self.database) as session:
            try:
                # Project graph - match all relationship types
                session.run("""
                    CALL gds.graph.project(
                        'ppr-graph',
                        'Node',
                        {
                            '*': {
                                orientation: 'DIRECTED'
                            }
                        }
                    )
                """)

                # Create personalization map
                personalization = {seed: 1.0 / len(valid_seeds) for seed in valid_seeds}

                # Run PPR
                result = session.run("""
                    CALL gds.pageRank.stream(
                        'ppr-graph',
                        {
                            maxIterations: $max_iter,
                            dampingFactor: $alpha,
                            sourceNodes: $seed_node_ids
                        }
                    )
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS node_id, score
                """,
                    max_iter=max_iter,
                    alpha=alpha,
                    seed_node_ids=[self._get_node_internal_id(seed) for seed in valid_seeds]
                )

                ppr_scores = {}
                for record in result:
                    ppr_scores[record["node_id"]] = record["score"]

                # Drop projected graph
                session.run("CALL gds.graph.drop('ppr-graph')")

                return ppr_scores
            except Exception as e:
                # Fallback implementation
                print(f"  ⚠️  GDS PPR failed: {e}, using fallback")
                return self._fallback_ppr(valid_seeds, alpha, max_iter)

    def _get_node_internal_id(self, node_id: str) -> int:
        """Get Neo4j internal node ID from node ID string."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Node {id: $node_id})
                RETURN id(n) AS internal_id
            """, node_id=node_id)
            record = result.single()
            return record["internal_id"] if record else None

    def _fallback_ppr(
        self,
        seed_nodes: List[str],
        alpha: float,
        max_iter: int
    ) -> Dict[str, float]:
        """Fallback PPR implementation using iterative computation."""
        # Simplified iterative PPR
        # This is a basic implementation - for production, use GDS
        scores = {node: 1.0 / len(seed_nodes) for node in seed_nodes}

        for _ in range(max_iter):
            new_scores = {node: 0.0 for node in scores}

            for node_id in scores:
                # Get neighbors
                neighbors = self.get_neighborhood(node_id, k=1)

                # Distribute score to neighbors
                if neighbors:
                    neighbor_count = len(set(tgt for _, _, tgt in neighbors))
                    for _, _, target in neighbors:
                        if target in new_scores:
                            new_scores[target] += scores[node_id] * (1 - alpha) / neighbor_count

                # Add restart probability
                if node_id in seed_nodes:
                    new_scores[node_id] += alpha / len(seed_nodes)

            scores = new_scores

        return scores

    def get_top_nodes_by_ppr(
        self,
        seed_nodes: List[str],
        top_k: int = 20,
        alpha: float = 0.85
    ) -> List[Tuple[str, float]]:
        """Get top-K nodes by Personalized PageRank score."""
        ppr_scores = self.personalized_pagerank(seed_nodes, alpha=alpha)

        if not ppr_scores:
            return []

        # Sort by score (descending) and return top-K
        sorted_nodes = sorted(
            ppr_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_nodes[:top_k]

    def get_node_count(self) -> int:
        """Get total number of nodes."""
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n:Node) RETURN count(n) AS count")
            record = result.single()
            return record["count"] if record else 0

    def get_edge_count(self) -> int:
        """Get total number of edges."""
        with self.driver.session(database=self.database) as session:
            # Count all relationship types, not just RELATED
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            record = result.single()
            return record["count"] if record else 0

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

