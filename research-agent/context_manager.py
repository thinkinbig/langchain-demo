"""
Context Manager for Long-Term Memory (Enterprise Knowledge)

This module handles the loading and formatting of the static "Knowledge Base"
for the agent. It implements the "Long Context" + "Prompt Caching" pattern
by constructing a cacheable system prompt block.

Enhanced with multi-stage retrieval and reranking for better context selection.
"""

import glob
import os
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Global cache for the loaded knowledge retrieval
# RAG Implementation

# Singleton instance
_VECTOR_STORE = None


class EnhancedRetriever:
    """
    Enhanced retriever with multi-stage retrieval and reranking.

    Implements:
    - Coarse retrieval (initial broad search)
    - Fine retrieval (reranking for precision)
    - Query expansion
    - Deduplication
    """

    def __init__(self, vector_store: Chroma):
        """
        Initialize enhanced retriever.

        Args:
            vector_store: Chroma vector store instance
        """
        self.vector_store = vector_store

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with related terms (simple implementation).

        In production, this could use LLM-based query expansion.

        Args:
            query: Original query

        Returns:
            List of expanded query variants
        """
        # Simple keyword extraction and expansion
        # In production, use LLM or NLP tools for better expansion
        expanded = [query]

        # Add query with common variations
        query_lower = query.lower()

        # Add question form if not present
        if not query_lower.endswith('?') and '?' not in query:
            expanded.append(query + "?")

        # Add "what is" prefix if not present
        if not query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            expanded.append(f"what is {query}")

        return expanded

    def coarse_retrieve(
        self,
        query: str,
        k: int = 20,
        threshold: float = 2.0
    ) -> List[Tuple[Document, float]]:
        """
        Coarse retrieval: broad search to get candidate documents.

        Args:
            query: Search query
            k: Number of candidates to retrieve
            threshold: L2 distance threshold (higher = more permissive)

        Returns:
            List of (Document, score) tuples
        """
        from config import settings

        # Retrieve more candidates than needed
        results = self.vector_store.similarity_search_with_score(query, k=k)

        # Filter by threshold
        candidates = [
            (doc, score) for doc, score in results
            if score < (threshold or settings.RETRIEVAL_L2_THRESHOLD)
        ]

        return candidates

    def rerank(
        self,
        candidates: List[Tuple[Document, float]],
        query: str,
        k: int = 5,
        graph_rag_manager=None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank candidates using multiple signals including graph connectivity.

        Uses a combination of:
        - Semantic similarity (from vector search)
        - Keyword overlap
        - Document length (prefer medium-length docs)
        - Source diversity
        - Graph connectivity (if GraphRAG is enabled)

        Args:
            candidates: List of (Document, score) tuples
            query: Original query
            k: Number of top results to return
            graph_rag_manager: Optional GraphRAGManager for graph-based reranking

        Returns:
            Reranked list of (Document, score) tuples
        """
        if not candidates:
            return []

        from config import settings

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Extract query entities for graph scoring (if GraphRAG enabled)
        query_entities = None
        if graph_rag_manager and settings.GRAPH_ENABLED:
            try:
                query_entities = graph_rag_manager.extract_entities_from_query(query)
            except Exception as e:
                print(f"  ⚠️  Failed to extract query entities for reranking: {e}")
                query_entities = None

        scored = []
        seen_sources = set()

        for doc, original_score in candidates:
            # Convert distance to similarity (lower distance = higher similarity)
            similarity = 1.0 / (1.0 + original_score) if original_score > 0 else 1.0

            # Calculate keyword overlap score
            doc_lower = doc.page_content.lower()
            doc_words = set(doc_lower.split())
            overlap = len(query_words & doc_words)
            keyword_score = overlap / max(len(query_words), 1) if query_words else 0

            # Length score (prefer medium-length documents)
            doc_length = len(doc.page_content)
            if 200 <= doc_length <= 2000:
                length_score = 1.0
            elif doc_length < 200:
                length_score = 0.5  # Too short
            else:
                length_score = max(0.7, 1.0 - (doc_length - 2000) / 5000)  # Too long

            # Source diversity bonus
            source = doc.metadata.get("source", "unknown")
            diversity_bonus = 0.0 if source in seen_sources else 0.1
            seen_sources.add(source)

            # Graph connectivity score (if GraphRAG enabled)
            graph_score = 0.0
            if graph_rag_manager and settings.GRAPH_ENABLED:
                try:
                    graph_score = graph_rag_manager.get_graph_connectivity_score(
                        doc.page_content,
                        query_entities
                    )
                except Exception:
                    # Graceful degradation: if graph scoring fails, use 0.0
                    pass

            # Combined score with graph weight from config
            graph_weight = settings.GRAPH_RERANK_WEIGHT
            base_weight = 1.0 - graph_weight

            # Normalize base weights to sum to base_weight
            similarity_weight = 0.4 * base_weight
            keyword_weight = 0.3 * base_weight
            length_weight = 0.15 * base_weight
            diversity_weight = 0.15 * base_weight  # Adjusted to fit

            combined_score = (
                similarity * similarity_weight +
                keyword_score * keyword_weight +
                length_score * length_weight +
                diversity_bonus * diversity_weight +
                graph_score * graph_weight  # Graph connectivity boost
            )

            scored.append((doc, combined_score))

        # Sort by combined score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:k]

    def retrieve_with_reranking(
        self,
        query: str,
        k: int = 5,
        coarse_k: int = 20,
        use_query_expansion: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Multi-stage retrieval with reranking.

        Args:
            query: Search query
            k: Final number of results
            coarse_k: Number of candidates for reranking
            use_query_expansion: Whether to use query expansion

        Returns:
            List of (Document, score) tuples
        """
        # Stage 1: Query expansion (optional)
        queries = [query]
        if use_query_expansion:
            queries.extend(self.expand_query(query))

        # Stage 2: Coarse retrieval from all query variants
        all_candidates = []
        for q in queries:
            candidates = self.coarse_retrieve(q, k=coarse_k)
            all_candidates.extend(candidates)

        # Deduplicate by document content
        seen_content = set()
        unique_candidates = []
        for doc, score in all_candidates:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_candidates.append((doc, score))

        # Stage 3: Rerank (with optional graph reranking)
        # Get GraphRAG manager if available
        graph_rag_manager = None
        try:
            from config import settings
            if settings.GRAPH_ENABLED:
                from memory.graph_rag import GraphRAGManager
                graph_rag_manager = GraphRAGManager()
        except Exception:
            # Graceful degradation: continue without graph reranking
            pass

        reranked = self.rerank(
            unique_candidates, query, k=k, graph_rag_manager=graph_rag_manager
        )

        return reranked


class VectorStoreManager:
    def __init__(self, persist_dir: str = "chroma_db", data_dir: str = "data"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.persist_dir = os.path.join(base_dir, persist_dir)
        self.data_dir = os.path.join(base_dir, data_dir)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="enterprise_knowledge",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        # Initialize enhanced retriever
        self.enhanced_retriever = EnhancedRetriever(self.vector_store)

    def ingest_documents(self):
        """Check for documents and ingest if DB is likely empty or upon request."""
        # Simple check: if DB has data, skip (for MVP).
        # In prod, we'd check file hashes.
        existing_count = self.vector_store._collection.count()
        if existing_count > 0:
            msg = (
                f"  📚 Knowledge Base loaded from persistence "
                f"({existing_count} chunks)."
            )
            print(msg)
            return

        print("  📚 Ingesting documents into Vector Store (First Run)...")
        documents = []

        # 1. Load Texts
        text_files = glob.glob(os.path.join(self.data_dir, "*.txt")) + \
                     glob.glob(os.path.join(self.data_dir, "*.md")) + \
                     glob.glob(os.path.join(self.data_dir, "*.csv"))

        for file_path in text_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": os.path.basename(file_path)}
                    ))
            except Exception as e:
                print(f"  ❌ Failed to load {file_path}: {e}")

        # 2. Load PDFs
        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        for file_path in pdf_files:
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                documents.append(Document(
                    page_content=text,
                    metadata={"source": os.path.basename(file_path)}
                ))
            except Exception as e:
                print(f"  ❌ Failed to load PDF {file_path}: {e}")

        if not documents:
            print("  ⚠️  No documents found to ingest.")
            return

        # 3. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        print(f"  🧩 Split {len(documents)} docs into {len(splits)} chunks.")

        # 4. Index into vector store
        self.vector_store.add_documents(splits)
        print("  ✅ Vector store ingestion complete.")

        # 5. Index into Knowledge Graph (if enabled)
        from config import settings
        if settings.GRAPH_ENABLED:
            try:
                from memory.graph_rag import GraphRAGManager
                graph_rag_manager = GraphRAGManager()

                total = len(splits)
                print(f"  🕸️  Indexing {total} chunks into Knowledge Graph...")
                for i, chunk in enumerate(splits):
                    if (i + 1) % 10 == 0 or i == total - 1:
                        print(
                            f"  🕸️  Indexing chunk {i+1}/{total} "
                            f"into Knowledge Graph..."
                        )
                    graph_rag_manager.index_document(
                        chunk.page_content,
                        source_metadata=chunk.metadata
                    )
                print("  ✅ Knowledge Graph indexing complete.")
            except Exception as e:
                print(f"  ⚠️  Knowledge Graph indexing failed: {e}")
                print("  ℹ️  Continuing without graph indexing.")

        print("  ✅ Ingestion complete.")

    def retrieve(
        self,
        query: str,
        k: int = 4,
        use_reranking: bool = True
    ) -> tuple[str, list[str]]:
        """Retrieve relevant context for a query.

        Implements hybrid retrieval:
        1. Vector search (semantic similarity)
        2. PPR-based graph retrieval (HippoRAG, if enabled)
        3. Merge and deduplicate results
        4. Rerank combined results

        Args:
            query: Search query
            k: Number of results to return
            use_reranking: Whether to use enhanced retrieval with reranking

        Returns:
            (context_str, list_of_sources)
        """
        if not query:
            return "", []

        from config import settings

        # Step 1: Vector search (always performed)
        if use_reranking:
            # Use enhanced retrieval with reranking
            vector_results = self.enhanced_retriever.retrieve_with_reranking(
                query,
                k=k,
                coarse_k=k * 4,  # Retrieve 4x candidates for reranking
                use_query_expansion=False  # Can be enabled for better recall
            )

            # Convert to document list (results are already reranked)
            relevant_results = [doc for doc, score in vector_results]
        else:
            # Fallback to original retrieval method
            results_with_score = self.vector_store.similarity_search_with_score(
                query, k=k
            )

            relevant_results = []
            for doc, score in results_with_score:
                if score < settings.RETRIEVAL_L2_THRESHOLD:
                    relevant_results.append(doc)

        # Step 2: PPR-based graph retrieval (if enabled)
        ppr_document_sources = set()
        ppr_context = ""
        if settings.GRAPH_ENABLED and settings.USE_PPR_RETRIEVAL:
            try:
                from memory.graph_rag import GraphRAGManager
                graph_rag_manager = GraphRAGManager()

                # Get PPR-based retrieval results
                ppr_context = graph_rag_manager.retrieve_with_ppr(
                    query,
                    top_k_nodes=settings.PPR_TOP_K_NODES,
                    top_k_docs=settings.PPR_TOP_K_DOCS,
                    alpha=settings.PPR_ALPHA
                )

                # Extract document sources from PPR results
                # The retrieve_with_ppr method returns formatted context,
                # but we also need the raw document sources for merging
                if ppr_context:
                    # Get document sources directly from graph store
                    query_entities = (
                        graph_rag_manager.extract_entities_from_query(query)
                    )
                    if query_entities:
                        top_nodes = (
                            graph_rag_manager.graph_store.get_top_nodes_by_ppr(
                                query_entities,
                                top_k=settings.PPR_TOP_K_NODES,
                                alpha=settings.PPR_ALPHA
                            )
                        )
                        node_ids = [node_id for node_id, _ in top_nodes]
                        ppr_document_sources = (
                            graph_rag_manager.graph_store.get_documents_for_nodes(
                                node_ids
                            )
                        )

            except Exception as e:
                print(f"  ⚠️  PPR retrieval failed: {e}")
                # Graceful degradation: continue without PPR
                pass

        # Step 3: Merge results
        # Collect sources from vector search
        vector_sources = set()
        for doc in relevant_results:
            source = doc.metadata.get("source", "Unknown")
            vector_sources.add(source)

        # Step 4: If PPR found additional sources, try to retrieve them
        # (This is a best-effort approach - we may not have all chunks)
        additional_docs = []
        if ppr_document_sources:
            # Get documents that are from PPR sources but not in vector results
            ppr_only_sources = ppr_document_sources - vector_sources

            # Try to retrieve chunks from these sources
            # Note: This requires searching the vector store, which we do by
            # searching for the source name in the query
            for source in list(ppr_only_sources)[:settings.PPR_TOP_K_DOCS]:
                try:
                    # Search for documents containing the source name
                    source_results = (
                        self.vector_store.similarity_search_with_score(
                            source, k=2
                        )
                    )
                    for doc, _ in source_results:
                        if doc.metadata.get("source") == source:
                            additional_docs.append(doc)
                            break
                except Exception:
                    pass

        # Combine all results
        all_results = relevant_results + additional_docs

        # Step 5: Rerank combined results if reranking is enabled
        if use_reranking and all_results:
            try:
                # Get graph manager for reranking
                graph_rag_manager = None
                if settings.GRAPH_ENABLED:
                    from memory.graph_rag import GraphRAGManager
                    graph_rag_manager = GraphRAGManager()

                # Rerank with graph connectivity scores
                # Give equal initial scores to all results
                reranked = self.enhanced_retriever.rerank(
                    [(doc, 1.0) for doc in all_results],
                    query,
                    k=k,
                    graph_rag_manager=graph_rag_manager
                )
                all_results = [doc for doc, score in reranked]

            except Exception:
                # Fallback: use original results
                pass

        # Step 6: Format output
        if not all_results:
            return "(No relevant internal documents found)", []

        context_parts = []
        sources = set()
        for doc in all_results[:k]:  # Limit to k results
            source = doc.metadata.get("source", "Unknown")
            sources.add(source)
            context_parts.append(f"Source: {source}\nContent:\n{doc.page_content}\n---")

        # Add PPR context as additional information (if available)
        if ppr_context and settings.USE_PPR_RETRIEVAL:
            context_parts.append(
                f"\n--- Knowledge Graph Context (PPR-based) ---\n"
                f"{ppr_context}"
            )

        return "\n".join(context_parts), list(sources)


def get_vector_manager():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = VectorStoreManager()
        _VECTOR_STORE.ingest_documents()
    return _VECTOR_STORE


def get_system_context(role_instructions: str, include_knowledge: bool = True) -> str:
    """
    Get system context.
    NOTE: For RAG, we don't inject the *retrieved* context here (dynamic).
    We inject a static instruction telling the agent that context will be provided
    in the human message or that it checks the vector store.
    """
    if include_knowledge:
        # We perform lazy retrieval or setup here if needed,
        # but the prompt injection happens at runtime in the graph.
        # This function just returns the static role instructions.
        pass
    return role_instructions


def retrieve_knowledge(
    query: str,
    k: int = 4,
    use_reranking: bool = True
) -> tuple[str, list[str]]:
    """
    Public helper to get RAG context string and sources.

    Args:
        query: Search query
        k: Number of results
        use_reranking: Whether to use enhanced retrieval with reranking
    """
    mgr = get_vector_manager()
    return mgr.retrieve(query, k=k, use_reranking=use_reranking)


def clear_cache():
    global _VECTOR_STORE
    _VECTOR_STORE = None

