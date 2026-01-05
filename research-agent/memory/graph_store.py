"""
Graph Store Implementation for Knowledge Graph.

Provides abstract interface and NetworkX-based implementation for storing
and querying knowledge graphs with persistence and community detection.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("  ⚠️  NetworkX not available. GraphRAG will be disabled.")


class GraphStore(ABC):
    """Abstract base class for graph storage backends."""

    @abstractmethod
    def add_node(self, node_id: str, node_type: str, description: str = "") -> None:
        """Add a node to the graph."""
        pass

    @abstractmethod
    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            relation: Relationship type
            properties: Optional dict with edge properties (version, confidence, context)
        """
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information by ID."""
        pass

    @abstractmethod
    def get_neighborhood(self, node_id: str, k: int = 1) -> List[Tuple[str, str, str]]:
        """
        Get neighbors of a node up to k hops away.

        Returns:
            List of (source, relation, target) tuples
        """
        pass

    @abstractmethod
    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nodes by fuzzy matching on ID or description.

        Returns:
            List of node dicts with 'id', 'type', 'description'
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist graph to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load graph from disk."""
        pass

    @abstractmethod
    def get_communities(self) -> List[List[str]]:
        """
        Detect communities in the graph.

        Returns:
            List of communities, where each community is a list of node IDs
        """
        pass


class NetworkXGraphStore(GraphStore):
    """
    NetworkX-based graph store with persistence.

    Uses NetworkX for in-memory graph operations and JSON for persistence.
    """

    def __init__(self, persist_path: str = "graph_store.json"):
        """
        Initialize NetworkX graph store.

        Args:
            persist_path: Path to JSON file for persistence
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for NetworkXGraphStore")

        self.graph = nx.DiGraph()  # Directed graph
        # node_id -> {type, description}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        # node_id -> set of document sources
        self.node_to_documents: Dict[str, Set[str]] = {}
        # document_source -> set of node_ids
        self.document_to_nodes: Dict[str, Set[str]] = {}
        self.persist_path = persist_path

        # Load existing graph if it exists
        if os.path.exists(persist_path):
            try:
                self.load()
            except Exception as e:
                print(f"  ⚠️  Failed to load graph from {persist_path}: {e}")
                print("  ℹ️  Starting with empty graph.")

    def add_node(self, node_id: str, node_type: str, description: str = "") -> None:
        """Add a node to the graph."""
        if not node_id:
            return

        # Normalize node_id (lowercase for consistency)
        node_id = node_id.strip()

        # Add to graph
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id)

        # Store metadata
        self.node_metadata[node_id] = {
            "type": node_type,
            "description": description
        }

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source: Source node ID
            target: Target node ID
            relation: Relationship type
            properties: Optional dict with edge properties (version, confidence, context)
        """
        if not source or not target:
            return

        source = source.strip()
        target = target.strip()

        # Ensure nodes exist
        if not self.graph.has_node(source):
            self.add_node(source, "Unknown", "")
        if not self.graph.has_node(target):
            self.add_node(target, "Unknown", "")

        # Add edge with relation and properties as attributes
        edge_attrs = {"relation": relation}
        if properties:
            # Store properties as edge attributes
            edge_attrs.update(properties)
        self.graph.add_edge(source, target, **edge_attrs)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information by ID."""
        if not self.graph.has_node(node_id):
            return None

        metadata = self.node_metadata.get(node_id, {})
        return {
            "id": node_id,
            "type": metadata.get("type", "Unknown"),
            "description": metadata.get("description", "")
        }

    def get_neighborhood(self, node_id: str, k: int = 1) -> List[Tuple[str, str, str]]:
        """
        Get neighbors of a node up to k hops away.

        Returns:
            List of (source, relation, target) tuples
        """
        if not self.graph.has_node(node_id):
            return []

        neighbors = []
        visited = set()

        # BFS to get neighbors up to k hops
        queue = [(node_id, 0)]  # (node, depth)
        visited.add(node_id)

        while queue:
            current, depth = queue.pop(0)

            if depth >= k:
                continue

            # Get outgoing edges
            for successor in self.graph.successors(current):
                if successor not in visited:
                    visited.add(successor)
                    edge_data = self.graph.edges[current, successor]
                    relation = edge_data.get("relation", "related_to")
                    neighbors.append((current, relation, successor))
                    if depth + 1 < k:
                        queue.append((successor, depth + 1))

            # Get incoming edges
            for predecessor in self.graph.predecessors(current):
                if predecessor not in visited:
                    visited.add(predecessor)
                    edge_data = self.graph.edges[predecessor, current]
                    relation = edge_data.get("relation", "related_to")
                    neighbors.append((predecessor, relation, current))
                    if depth + 1 < k:
                        queue.append((predecessor, depth + 1))

        return neighbors

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nodes by fuzzy matching on ID or description.

        Uses simple substring matching. For production, consider using
        fuzzy string matching libraries like fuzzywuzzy or rapidfuzz.
        """
        if not query:
            return []

        query_lower = query.lower()
        matches = []

        for node_id, metadata in self.node_metadata.items():
            score = 0

            # Check if query matches node ID
            if query_lower in node_id.lower():
                score += 2

            # Check if query matches description
            description = metadata.get("description", "").lower()
            if query_lower in description:
                score += 1

            # Check if query matches type
            node_type = metadata.get("type", "").lower()
            if query_lower in node_type:
                score += 0.5

            if score > 0:
                matches.append((score, {
                    "id": node_id,
                    "type": metadata.get("type", "Unknown"),
                    "description": metadata.get("description", "")
                }))

        # Sort by score (descending) and return top matches
        matches.sort(key=lambda x: x[0], reverse=True)
        return [match[1] for match in matches[:limit]]

    def get_communities(self) -> List[List[str]]:
        """
        Detect communities in the graph using greedy modularity.

        Returns:
            List of communities, where each community is a list of node IDs
        """
        if self.graph.number_of_nodes() == 0:
            return []

        try:
            # Convert to undirected graph for community detection
            undirected = self.graph.to_undirected()

            # Use greedy modularity communities
            communities = community.greedy_modularity_communities(undirected)

            # Convert to list of lists
            return [list(comm) for comm in communities]
        except Exception as e:
            print(f"  ⚠️  Community detection failed: {e}")
            # Fallback: return each node as its own community
            return [[node] for node in self.graph.nodes()]

    def save(self) -> None:
        """Persist graph to JSON file."""
        try:
            data = {
                "nodes": [],
                "edges": [],
                "node_to_documents": {},
                "document_to_nodes": {}
            }

            # Save nodes with metadata
            for node_id in self.graph.nodes():
                metadata = self.node_metadata.get(node_id, {})
                data["nodes"].append({
                    "id": node_id,
                    "type": metadata.get("type", "Unknown"),
                    "description": metadata.get("description", "")
                })

            # Save edges with properties
            for source, target, attrs in self.graph.edges(data=True):
                edge_data = {
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation", "related_to")
                }
                # Extract properties (version, confidence, context) if present
                properties = {}
                if "version" in attrs:
                    properties["version"] = attrs["version"]
                if "confidence" in attrs:
                    properties["confidence"] = attrs["confidence"]
                if "context" in attrs:
                    properties["context"] = attrs["context"]
                # Only include properties if non-empty
                if properties:
                    edge_data["properties"] = properties
                data["edges"].append(edge_data)

            # Save node-document mappings (convert sets to lists for JSON)
            for node_id, doc_set in self.node_to_documents.items():
                data["node_to_documents"][node_id] = list(doc_set)

            for doc_source, node_set in self.document_to_nodes.items():
                data["document_to_nodes"][doc_source] = list(node_set)

            # Write to file
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ⚠️  Failed to save graph to {self.persist_path}: {e}")

    def load(self) -> None:
        """Load graph from JSON file."""
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Clear existing graph
            self.graph.clear()
            self.node_metadata.clear()
            self.node_to_documents.clear()
            self.document_to_nodes.clear()

            # Load nodes
            for node_data in data.get("nodes", []):
                node_id = node_data.get("id")
                if node_id:
                    self.add_node(
                        node_id,
                        node_data.get("type", "Unknown"),
                        node_data.get("description", "")
                    )

            # Load edges with properties
            for edge_data in data.get("edges", []):
                source = edge_data.get("source")
                target = edge_data.get("target")
                relation = edge_data.get("relation", "related_to")
                properties = edge_data.get("properties")  # Load properties if present
                if source and target:
                    self.add_edge(source, target, relation, properties=properties)

            # Load node-document mappings (convert lists back to sets)
            for node_id, doc_list in data.get("node_to_documents", {}).items():
                self.node_to_documents[node_id] = set(doc_list)

            for doc_source, node_list in data.get("document_to_nodes", {}).items():
                self.document_to_nodes[doc_source] = set(node_list)

        except Exception as e:
            print(f"  ⚠️  Failed to load graph from {self.persist_path}: {e}")
            raise

    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()

    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()

    def add_node_document_mapping(
        self,
        node_id: str,
        document_source: str
    ) -> None:
        """
        Add mapping between a node and a document.

        Args:
            node_id: ID of the graph node
            document_source: Source identifier of the document
        """
        if not node_id or not document_source:
            return

        node_id = node_id.strip()
        document_source = document_source.strip()

        # Add to node -> documents mapping
        if node_id not in self.node_to_documents:
            self.node_to_documents[node_id] = set()
        self.node_to_documents[node_id].add(document_source)

        # Add to document -> nodes mapping
        if document_source not in self.document_to_nodes:
            self.document_to_nodes[document_source] = set()
        self.document_to_nodes[document_source].add(node_id)

    def get_documents_for_nodes(
        self,
        node_ids: List[str]
    ) -> Set[str]:
        """
        Get all documents that contain any of the given nodes.

        Args:
            node_ids: List of node IDs to look up

        Returns:
            Set of document source identifiers
        """
        documents = set()
        for node_id in node_ids:
            if node_id in self.node_to_documents:
                documents.update(self.node_to_documents[node_id])
        return documents

    def personalized_pagerank(
        self,
        seed_nodes: List[str],
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute Personalized PageRank (PPR) starting from seed nodes.

        This implements the spreading activation mechanism from HippoRAG,
        where activation propagates from seed nodes through the graph.

        Args:
            seed_nodes: List of node IDs to use as seed nodes (restart points)
            alpha: Damping factor (probability of following links vs restarting)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance

        Returns:
            Dictionary mapping node_id to PPR score
        """
        if not seed_nodes or self.graph.number_of_nodes() == 0:
            return {}

        # Filter seed nodes to only those that exist in the graph
        valid_seeds = [node for node in seed_nodes if self.graph.has_node(node)]
        if not valid_seeds:
            return {}

        # Create personalized restart vector
        # Equal probability for all seed nodes, 0 for others
        personalization = {}
        seed_prob = 1.0 / len(valid_seeds)
        for node in self.graph.nodes():
            if node in valid_seeds:
                personalization[node] = seed_prob
            else:
                personalization[node] = 0.0

        # Compute Personalized PageRank
        try:
            ppr_scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=tol
            )
            return ppr_scores
        except Exception as e:
            print(f"  ⚠️  PPR computation failed: {e}")
            return {}

    def get_top_nodes_by_ppr(
        self,
        seed_nodes: List[str],
        top_k: int = 20,
        alpha: float = 0.85
    ) -> List[Tuple[str, float]]:
        """
        Get top-K nodes by Personalized PageRank score.

        Args:
            seed_nodes: List of node IDs to use as seed nodes
            top_k: Number of top nodes to return
            alpha: Damping factor for PPR

        Returns:
            List of (node_id, ppr_score) tuples, sorted by score descending
        """
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

