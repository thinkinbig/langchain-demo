"""
Vector Store Abstract Interface.

Provides abstract base class for vector storage backends,
supporting multiple implementations (ChromaDB, Neo4j, Qdrant, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain_core.documents import Document


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            ids: Optional list of document IDs. If None, IDs will be generated.

        Returns:
            List of document IDs that were added
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of Document objects
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores.

        Args:
            query: Search query string
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        pass

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete. If None, deletes all.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        pass

