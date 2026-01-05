"""
Neo4j Vector Store Implementation.

Uses Neo4j with vector index for similarity search.
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from memory.vector_store import VectorStore


class Neo4jVectorStore(VectorStore):
    """Neo4j implementation of VectorStore using vector index."""

    def __init__(
        self,
        collection_name: str = "enterprise_knowledge",
        embedding_function=None,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """
        Initialize Neo4j vector store.

        Args:
            collection_name: Name of the collection/index
            embedding_function: Embedding function to use
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package is required for Neo4jVectorStore. "
                "Install it with: pip install neo4j"
            )

        from config import settings

        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # Create vector index if it doesn't exist
        self._create_vector_index()

    def _create_vector_index(self):
        """Create vector index in Neo4j if it doesn't exist."""
        # Neo4j vector index creation
        # Note: This requires Neo4j 5.x+ with vector index support
        # or a custom implementation using stored procedures
        index_name = f"{self.collection_name}_vector_index"

        with self.driver.session(database=self.database) as session:
            # Check if index exists
            result = session.run(
                "SHOW INDEXES YIELD name WHERE name = $name",
                name=index_name
            )

            if not result.single():
                # Create vector index
                # This is a simplified version - actual implementation
                # depends on Neo4j version and vector plugin
                try:
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (d:Document) ON d.embedding
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: 384,
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """)
                except Exception as e:
                    # If vector index creation fails, we'll use a workaround
                    # with stored embeddings and cosine similarity
                    print(f"  ⚠️  Vector index creation failed: {e}")
                    print("  ℹ️  Using fallback similarity search")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.embedding_function:
            return self.embedding_function.embed_query(text)
        else:
            raise ValueError("Embedding function is required")

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to Neo4j."""
        if not documents:
            return []

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]

        added_ids = []

        with self.driver.session(database=self.database) as session:
            for doc, doc_id in zip(documents, ids):
                # Generate embedding
                embedding = self._get_embedding(doc.page_content)

                # Create document node
                session.run("""
                    MERGE (d:Document {id: $id})
                    SET d.content = $content,
                        d.embedding = $embedding,
                        d.source = $source,
                        d.collection = $collection
                """,
                    id=doc_id,
                    content=doc.page_content,
                    embedding=embedding,
                    source=doc.metadata.get("source", "unknown"),
                    collection=self.collection_name
                )
                added_ids.append(doc_id)

        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Perform similarity search."""
        results_with_score = self.similarity_search_with_score(query, k=k, filter=filter)
        return [doc for doc, _ in results_with_score]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search with scores."""
        query_embedding = self._get_embedding(query)

        # Build filter query
        filter_clause = ""
        filter_params = {}
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(f"d.{key} = ${key}")
                filter_params[key] = value
            if filter_conditions:
                filter_clause = " AND " + " AND ".join(filter_conditions)

        # Use vector index if available, otherwise use cosine similarity
        with self.driver.session(database=self.database) as session:
            # Try vector index search first
            try:
                result = session.run(f"""
                    CALL db.index.vector.queryNodes(
                        '{self.collection_name}_vector_index',
                        $k,
                        $queryEmbedding
                    )
                    YIELD node, score
                    WHERE node.collection = $collection {filter_clause}
                    RETURN node, score
                    LIMIT $k
                """,
                    k=k,
                    queryEmbedding=query_embedding,
                    collection=self.collection_name,
                    **filter_params
                )
            except Exception:
                # Fallback to cosine similarity calculation
                result = session.run(f"""
                    MATCH (d:Document)
                    WHERE d.collection = $collection {filter_clause}
                    WITH d, 
                         gds.similarity.cosine(d.embedding, $queryEmbedding) AS score
                    RETURN d, score
                    ORDER BY score DESC
                    LIMIT $k
                """,
                    collection=self.collection_name,
                    queryEmbedding=query_embedding,
                    k=k,
                    **filter_params
                )

            results = []
            for record in result:
                node = record["node"]
                score = record.get("score", 0.0)

                # Convert distance to similarity if needed
                # (Neo4j vector index may return distance, not similarity)
                if score > 1.0:
                    # Likely a distance metric, convert to similarity
                    similarity = 1.0 / (1.0 + score)
                else:
                    similarity = score

                doc = Document(
                    page_content=node.get("content", ""),
                    metadata={
                        "source": node.get("source", "unknown"),
                        "id": node.get("id")
                    }
                )
                # Return distance (lower is better) for consistency with ChromaDB
                distance = 1.0 - similarity
                results.append((doc, distance))

        return results

    def delete(self, ids: Optional[List[str]] = None) -> bool:
        """Delete documents from Neo4j."""
        with self.driver.session(database=self.database) as session:
            if ids is None:
                # Delete all documents in collection
                session.run("""
                    MATCH (d:Document)
                    WHERE d.collection = $collection
                    DETACH DELETE d
                """, collection=self.collection_name)
            else:
                # Delete specific documents
                session.run("""
                    MATCH (d:Document)
                    WHERE d.id IN $ids
                    DETACH DELETE d
                """, ids=ids)
        return True

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (d:Document)
                WHERE d.collection = $collection
                RETURN count(d) AS count
            """, collection=self.collection_name)
            record = result.single()
            return record["count"] if record else 0

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

