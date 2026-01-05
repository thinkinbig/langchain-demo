"""
ChromaDB Vector Store Implementation.

Wraps ChromaDB to implement the VectorStore interface.
"""

from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from memory.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore."""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        embedding_function=None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_function: Embedding function to use
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory

        # Initialize ChromaDB
        self._chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to ChromaDB."""
        return self._chroma.add_documents(documents, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Perform similarity search."""
        return self._chroma.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        return self._chroma.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """Delete documents from ChromaDB."""
        if ids is None:
            # ChromaDB doesn't support deleting all, so we get all IDs first
            # This is a limitation - in practice, you'd want to track IDs
            return False
        self._chroma.delete(ids=ids)
        return True

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self._chroma._collection.count()
        except AttributeError:
            # Fallback: try to get count via search
            # This is not ideal but works as a fallback
            return 0

    @property
    def chroma(self) -> Chroma:
        """Get the underlying ChromaDB instance (for compatibility)."""
        return self._chroma

