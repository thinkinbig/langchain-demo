"""
Context Manager for Long-Term Memory (Enterprise Knowledge)

This module handles the loading and formatting of the static "Knowledge Base"
for the agent. It implements the "Long Context" + "Prompt Caching" pattern
by constructing a cacheable system prompt block.
"""

import glob
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Global cache for the loaded knowledge retrieval
# RAG Implementation

# Singleton instance
_VECTOR_STORE = None

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

        # 4. Index
        self.vector_store.add_documents(splits)
        print("  ✅ Ingestion complete.")

    def retrieve(self, query: str, k: int = 4) -> tuple[str, list[str]]:
        """Retrieve relevant context for a query.

        Returns:
            (context_str, list_of_sources)
        """
        if not query:
            return "", []

        # Use similarity_search_with_score to filter irrelevant results
        # Chroma/HF default is L2 distance. Lower is better.
        # Threshold: 1.0 (Empirical: Relevant ~0.7, Irrelevant > 1.1)
        results_with_score = self.vector_store.similarity_search_with_score(query, k=k)

        relevant_results = []
        for doc, score in results_with_score:
            if score < 1.0:
                relevant_results.append(doc)
            else:
                # Debug log (optional, or remove in prod)
                # print(f"  [RAG] Skipped low relevance: {score:.3f}")
                pass

        if not relevant_results:
            return "(No relevant internal documents found)", []

        context_parts = []
        sources = set()
        for doc in relevant_results:
            source = doc.metadata.get("source", "Unknown")
            sources.add(source)
            context_parts.append(f"Source: {source}\nContent:\n{doc.page_content}\n---")

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

def retrieve_knowledge(query: str, k: int = 4) -> tuple[str, list[str]]:
    """Public helper to get RAG context string and sources."""
    mgr = get_vector_manager()
    return mgr.retrieve(query, k=k)

def clear_cache():
    global _VECTOR_STORE
    _VECTOR_STORE = None
