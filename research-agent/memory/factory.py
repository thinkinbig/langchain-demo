"""
Factory for creating vector and graph stores.

Supports multiple backends based on configuration.
"""

from typing import Optional

from config import settings
from memory.backends.chroma_store import ChromaVectorStore
from memory.graph_store import GraphStore, NetworkXGraphStore
from memory.vector_store import VectorStore


def create_vector_store(
    collection_name: str = "enterprise_knowledge",
    embedding_function=None,
    persist_directory: Optional[str] = None,
    backend: Optional[str] = None
) -> VectorStore:
    """
    Create a vector store instance based on configuration.

    Args:
        collection_name: Name of the collection
        embedding_function: Embedding function to use
        persist_directory: Directory to persist the database
        backend: Backend type ("chroma", "neo4j", etc.). If None, uses config.

    Returns:
        VectorStore instance
    """
    backend = backend or getattr(settings, "VECTOR_STORE_BACKEND", "chroma")

    if backend == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    elif backend == "neo4j":
        try:
            from memory.backends.neo4j_vector_store import Neo4jVectorStore
            return Neo4jVectorStore(
                collection_name=collection_name,
                embedding_function=embedding_function
            )
        except ImportError as e:
            raise ImportError(
                "Neo4j backend requires neo4j package. Install with: pip install neo4j"
            ) from e
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")


def create_graph_store(
    persist_path: Optional[str] = None,
    backend: Optional[str] = None
) -> GraphStore:
    """
    Create a graph store instance based on configuration.

    Args:
        persist_path: Path to persist the graph
        backend: Backend type ("networkx", "neo4j"). If None, uses config.

    Returns:
        GraphStore instance
    """
    backend = backend or getattr(settings, "GRAPH_STORE_BACKEND", "networkx")

    if backend == "networkx":
        persist_path = persist_path or getattr(settings, "GRAPH_PERSIST_PATH", "graph_store.json")
        return NetworkXGraphStore(persist_path=persist_path)
    elif backend == "neo4j":
        try:
            from memory.backends.neo4j_graph_store import Neo4jGraphStore
            return Neo4jGraphStore()
        except ImportError as e:
            raise ImportError(
                "Neo4j backend requires neo4j package. Install with: pip install neo4j"
            ) from e
    else:
        raise ValueError(f"Unknown graph store backend: {backend}")

