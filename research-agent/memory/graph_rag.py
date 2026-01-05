"""
GraphRAG Service Implementation.

Handles:
1. LLM-based Entity & Relation Extraction (Indexing)
2. Graph Navigation & Context Retrieval (Retrieval)
3. Community Detection for Global Context Understanding
"""

import json
from typing import Any, Dict, Optional

from config import settings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from memory.graph_store import GraphStore, NetworkXGraphStore
from prompts import (
    GRAPH_ENTITY_EXTRACTION_MAIN,
    GRAPH_ENTITY_EXTRACTION_SYSTEM,
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
)


class GraphExtractor:
    """Extracts graph elements (nodes/edges) from text using LLM."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract knowledge graph elements from text.
        Returns dict with 'nodes' and 'edges'.
        """
        if not text or len(text) < 50:
            return {"nodes": [], "edges": []}

        try:
            messages = [
                SystemMessage(content=GRAPH_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Text to analyze:\n{text[:10000]}"
                )  # Truncate to avoid context overflow
            ]

            # Use JSON mode if supported, or just rely on prompt instructions
            # For robustness, we request JSON object
            response = self.llm.invoke(
                messages,
                response_format={"type": "json_object"}
            ).content

            # Parse JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Fallback: try to find first { and last }
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(response[start : end + 1])
                else:
                    return {"nodes": [], "edges": []}

            return data

        except Exception as e:
            print(f"  ❌ Graph Extraction Failed: {e}")
            return {"nodes": [], "edges": []}


class GraphRAGManager:
    """
    Manages Knowledge Graph operations: Indexing and Retrieval.
    Functions as the 'Navigation Layer' for the research agent.
    """

    def __init__(self, graph_store: Optional[GraphStore] = None):
        if graph_store:
            self.graph_store = graph_store
        else:
            # Default to local NetworkX store
            self.graph_store = NetworkXGraphStore(
                persist_path=settings.GRAPH_PERSIST_PATH
            )

        # We need an LLM for extraction
        try:
            from llm.factory import get_llm_by_model_choice
            self.llm = get_llm_by_model_choice("plus")
            self.extractor = GraphExtractor(self.llm)
        except ImportError:
            print("  ⚠️  LLM factory not available, graphing will be disabled.")
            self.extractor = None

    def index_document(self, text: str, source_metadata: Optional[Dict] = None) -> None:
        """
        Process text, extract entities, and update the graph.
        Also stores node-document mappings for HippoRAG retrieval.
        """
        if not self.extractor or not settings.GRAPH_ENABLED:
            return

        print("  🕸️  Indexing document into Knowledge Graph...")
        data = self.extractor.extract(text)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Extract document source from metadata
        document_source = "unknown"
        if source_metadata:
            document_source = source_metadata.get("source", "unknown")

        count_nodes = 0
        count_edges = 0

        # Add Nodes and store node-document mappings
        for node in nodes:
            nid = node.get("id")
            ntype = node.get("type", "Unknown")
            desc = node.get("description", "")
            if nid:
                self.graph_store.add_node(nid, ntype, desc)
                # Store mapping: this node appears in this document
                self.graph_store.add_node_document_mapping(nid, document_source)
                count_nodes += 1

        # Add Edges
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            rel = edge.get("relation", "related_to")
            properties = edge.get("properties")  # Extract properties if present
            if src and tgt:
                self.graph_store.add_edge(src, tgt, rel, properties=properties)
                count_edges += 1

        # Save after batch update
        self.graph_store.save()
        print(f"  ✅ Added {count_nodes} nodes and {count_edges} edges to graph.")

    def extract_entities_from_query(self, query: str) -> list[str]:
        """
        Extract entity IDs from query using LLM.

        Returns:
            List of entity IDs that match query terms
        """
        if not self.extractor or not settings.GRAPH_ENABLED:
            # Fallback to fuzzy search
            nodes = self.graph_store.search_nodes(query, limit=5)
            return [node["id"] for node in nodes]

        try:
            # Use LLM to extract entities from query
            prompt = GRAPH_ENTITY_EXTRACTION_MAIN.format(query=query)

            messages = [
                SystemMessage(content=GRAPH_ENTITY_EXTRACTION_SYSTEM),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(
                messages,
                response_format={"type": "json_object"}
            ).content

            # Parse JSON
            try:
                data = json.loads(response)
                entities = data.get("entities", [])
            except json.JSONDecodeError:
                # Fallback: try to find JSON in response
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(response[start : end + 1])
                    entities = data.get("entities", [])
                else:
                    entities = []

            # Match extracted entities to graph nodes
            matched_ids = []
            for entity in entities:
                if isinstance(entity, str):
                    # Search for matching nodes
                    matches = self.graph_store.search_nodes(entity, limit=3)
                    matched_ids.extend([m["id"] for m in matches])

            return list(set(matched_ids))  # Remove duplicates

        except Exception as e:
            print(f"  ⚠️  Entity extraction from query failed: {e}")
            # Fallback to fuzzy search
            nodes = self.graph_store.search_nodes(query, limit=5)
            return [node["id"] for node in nodes]

    def get_community_summary(self, query: str) -> str:
        """
        Find relevant communities in graph and generate summaries.

        Uses community detection to find clusters of related entities,
        then generates context about entities in those communities.
        """
        if not settings.GRAPH_ENABLED:
            return ""

        try:
            # Extract entities from query
            query_entities = self.extract_entities_from_query(query)

            if not query_entities:
                return ""

            # Get communities
            communities = self.graph_store.get_communities()

            # Find communities containing query entities
            relevant_communities = []
            for community_nodes in communities:
                if any(
                    entity in community_nodes for entity in query_entities
                ):
                    relevant_communities.append(community_nodes)

            if not relevant_communities:
                return ""

            # Generate summary of relevant communities
            summary_lines = ["Knowledge Graph Community Context:"]

            # Limit to top 2 communities
            for i, comm in enumerate(relevant_communities[:2]):
                summary_lines.append(f"\nCommunity {i+1} ({len(comm)} entities):")

                # Get details for entities in community
                for node_id in comm[:10]:  # Limit to 10 entities
                    node = self.graph_store.get_node(node_id)
                    if node:
                        desc = node.get('description', '')[:100]
                        summary_lines.append(
                            f"  - {node_id} ({node.get('type')}): {desc}"
                        )

            return "\n".join(summary_lines)

        except Exception as e:
            print(f"  ⚠️  Community detection failed: {e}")
            return ""

    def retrieve_with_ppr(
        self,
        query: str,
        top_k_nodes: int = 20,
        top_k_docs: int = 10,
        alpha: float = 0.85
    ) -> str:
        """
        Retrieve context using Personalized PageRank (HippoRAG methodology).

        This implements the spreading activation mechanism:
        1. Extract entities from query
        2. Run PPR with seed nodes = extracted entities
        3. Get top-K nodes by PPR score
        4. Retrieve documents containing those nodes
        5. Return formatted context

        Args:
            query: Search query
            top_k_nodes: Number of top nodes to retrieve from PPR
            top_k_docs: Number of documents to retrieve from top nodes
            alpha: Damping factor for PPR (restart probability)

        Returns:
            Formatted context string with document information
        """
        if not settings.GRAPH_ENABLED:
            return ""

        try:
            # 1. Extract entities from query
            seed_nodes = self.extract_entities_from_query(query)
            if not seed_nodes:
                return ""

            # 2. Run PPR to get top-K nodes
            top_nodes = self.graph_store.get_top_nodes_by_ppr(
                seed_nodes,
                top_k=top_k_nodes,
                alpha=alpha
            )

            if not top_nodes:
                return ""

            # 3. Get documents for top nodes
            node_ids = [node_id for node_id, _ in top_nodes]
            document_sources = self.graph_store.get_documents_for_nodes(node_ids)

            if not document_sources:
                # Fallback: return node information if no documents mapped
                context_lines = [
                    "Knowledge Graph Context (PPR-based retrieval):"
                ]
                for node_id, score in top_nodes[:top_k_nodes]:
                    node = self.graph_store.get_node(node_id)
                    if node:
                        desc = node.get("description", "")[:100]
                        context_lines.append(
                            f"- {node_id} (PPR: {score:.4f}): {desc}"
                        )
                return "\n".join(context_lines) if len(context_lines) > 1 else ""

            # 4. Format context with document information
            context_lines = [
                "Knowledge Graph Context (PPR-based retrieval):",
                f"Found {len(document_sources)} documents via "
                f"{len(node_ids)} activated nodes."
            ]

            # Limit to top_k_docs
            document_list = list(document_sources)[:top_k_docs]
            for doc_source in document_list:
                context_lines.append(f"\nDocument: {doc_source}")
                # Get nodes that appear in this document
                doc_nodes = [
                    (node_id, score) for node_id, score in top_nodes
                    if node_id in node_ids and doc_source in
                    self.graph_store.node_to_documents.get(node_id, set())
                ]
                if doc_nodes:
                    context_lines.append("  Relevant entities:")
                    for node_id, score in doc_nodes[:5]:  # Limit to 5 per doc
                        node = self.graph_store.get_node(node_id)
                        if node:
                            desc = node.get("description", "")[:80]
                            context_lines.append(
                                f"    - {node_id} (PPR: {score:.4f}): {desc}"
                            )

            return "\n".join(context_lines)

        except Exception as e:
            print(f"  ⚠️  PPR retrieval failed: {e}")
            return ""

    def get_graph_connectivity_score(
        self,
        document_text: str,
        query_entities: Optional[list[str]] = None
    ) -> float:
        """
        Calculate graph connectivity score for a document.

        Checks if entities mentioned in the document are connected in the graph.
        Higher score = more connections found.

        Returns:
            Score between 0.0 and 1.0
        """
        if not settings.GRAPH_ENABLED or not document_text:
            return 0.0

        try:
            # Extract entities from document (simple keyword matching for now)
            # In production, could use LLM extraction here too
            doc_lower = document_text.lower()

            # Get all nodes from graph
            all_nodes = list(self.graph_store.node_metadata.keys())

            # Count how many graph entities appear in document
            matches = 0
            for node_id in all_nodes:
                if node_id.lower() in doc_lower:
                    matches += 1

            # Calculate connectivity: if document mentions multiple connected entities
            if matches == 0:
                return 0.0

            # Check if matched entities are connected
            matched_entities = [
                node_id for node_id in all_nodes
                if node_id.lower() in doc_lower
            ]
            connections = 0

            for i, entity1 in enumerate(matched_entities):
                for entity2 in matched_entities[i+1:]:
                    # Check if there's a path between them (up to 2 hops)
                    neighbors1 = self.graph_store.get_neighborhood(
                        entity1, k=2
                    )
                    if any(tgt == entity2 for _, _, tgt in neighbors1):
                        connections += 1

            # Normalize score
            if len(matched_entities) < 2:
                # Base score for single entity
                return min(0.5, matches / 10.0)

            # Higher score for more connections
            max_possible = (
                len(matched_entities) * (len(matched_entities) - 1) / 2
            )
            if max_possible > 0:
                connectivity_ratio = connections / max_possible
            else:
                connectivity_ratio = 0.0

            # Combine: base score (0.3) + connectivity bonus (0.7)
            base_score = 0.3 * min(1.0, matches / 5.0)
            conn_bonus = 0.7 * min(1.0, connectivity_ratio)
            score = base_score + conn_bonus
            return min(1.0, score)

        except Exception as e:
            print(f"  ⚠️  Graph connectivity scoring failed: {e}")
            return 0.0
