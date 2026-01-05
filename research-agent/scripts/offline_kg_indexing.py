#!/usr/bin/env python3
"""
Offline Knowledge Graph Indexing Script

Index documents into Neo4j knowledge graph in one batch,
avoiding repeated indexing on each run.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Find and load .env file (must be before importing config)
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Also try project root directory
    root_env = os.path.join(script_dir, "..", ".env")
    if os.path.exists(root_env):
        load_dotenv(root_env)
    else:
        # Finally try current directory
        load_dotenv()

# Must import after load_dotenv() since we need to set path first
from config import settings  # noqa: E402
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from memory.graph_rag import GraphRAGManager  # noqa: E402


def load_documents(data_dir: str = "data"):
    """Load all documents from the data directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        return []

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Supported document types
    supported_extensions = {".txt", ".pdf"}

    print(f"📂 Scanning directory: {data_dir}")
    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                print(f"  📄 Loading: {file_path.name}")
                if file_path.suffix.lower() == ".txt":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                elif file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                else:
                    continue

                docs = loader.load()
                # Add source information
                for doc in docs:
                    doc.metadata["source"] = str(file_path.relative_to(data_path))

                # Split documents
                splits = text_splitter.split_documents(docs)
                documents.extend(splits)
                print(f"    ✅ Loaded, split into {len(splits)} chunks")

            except Exception as e:
                print(f"    ⚠️  Failed to load: {e}")
                continue

    return documents


def index_documents_offline():
    """Offline index documents into knowledge graph"""
    print("=" * 80)
    print("Offline Knowledge Graph Indexing")
    print("=" * 80)
    print()

    # Check configuration
    if not settings.GRAPH_ENABLED:
        print("❌ Knowledge graph is not enabled (GRAPH_ENABLED=False)")
        print("   Please set GRAPH_ENABLED=True in config.py")
        return

    if settings.GRAPH_STORE_BACKEND != "neo4j":
        print(f"⚠️  Current backend: {settings.GRAPH_STORE_BACKEND}")
        print("   Recommend using neo4j backend for better performance")
        print()

    # Check Neo4j connection
    if settings.GRAPH_STORE_BACKEND == "neo4j":
        print("🔌 Checking Neo4j connection...")
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            driver.verify_connectivity()
            driver.close()
            print(f"  ✅ Neo4j connection successful: {settings.NEO4J_URI}")
        except Exception as e:
            print(f"  ❌ Neo4j connection failed: {e}")
            print(
                "   Please check if Neo4j is running and connection config is correct"
            )
            return
        print()

    # Load documents
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    documents = load_documents(data_dir)

    if not documents:
        print("❌ No documents found to index")
        return

    print()
    print(f"📊 Total: {len(documents)} document chunks")
    print()

    # Initialize GraphRAGManager
    print("🔧 Initializing knowledge graph manager...")
    try:
        graph_rag_manager = GraphRAGManager()
        if not graph_rag_manager.extractor:
            print("  ❌ Failed to initialize entity extractor")
            return
        print("  ✅ Initialization complete")
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return

    print()
    print("=" * 80)
    print("Starting indexing...")
    print("=" * 80)
    print()

    # Batch indexing
    import time

    total = len(documents)
    indexed = 0
    errors = 0
    start_time = time.time()

    for i, doc in enumerate(documents, 1):
        chunk_start = time.time()
        progress = (i / total) * 100

        # Show progress for every chunk
        source = doc.metadata.get("source", "unknown")
        print(f"📊 [{i}/{total}] ({progress:.1f}%) Processing: {source[:50]}...")

        try:
            graph_rag_manager.index_document(
                doc.page_content,
                source_metadata=doc.metadata
            )
            indexed += 1
            chunk_time = time.time() - chunk_start
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total - i)

            # Show detailed progress every 10 chunks or at the end
            if i % 10 == 0 or i == total:
                print(
                    f"   ✅ Indexed {indexed}/{i} chunks | "
                    f"Time: {chunk_time:.1f}s | "
                    f"ETA: {remaining/60:.1f}m"
                )
        except Exception as e:
            errors += 1
            chunk_time = time.time() - chunk_start
            print(f"  ⚠️  Indexing failed [{i}/{total}] ({chunk_time:.1f}s): {e}")
            continue

    print()
    print("=" * 80)
    print("Indexing complete")
    print("=" * 80)
    print(f"✅ Successfully indexed: {indexed} chunks")
    if errors > 0:
        print(f"❌ Failed: {errors} chunks")
    print()

    # Display statistics
    if settings.GRAPH_STORE_BACKEND == "neo4j":
        print("📊 Neo4j Statistics:")
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                # Node count
                result = session.run("MATCH (n:Node) RETURN count(n) AS count")
                node_count = result.single()["count"]
                print(f"  📍 Node nodes: {node_count}")

                # Edge count
                result = session.run(
                    "MATCH ()-[r:RELATED]->() RETURN count(r) AS count"
                )
                edge_count = result.single()["count"]
                print(f"  🔗 RELATED relationships: {edge_count}")

                # Document mappings
                result = session.run(
                    "MATCH ()-[r:APPEARS_IN]->() RETURN count(r) AS count"
                )
                mapping_count = result.single()["count"]
                print(f"  📄 Document mappings: {mapping_count}")

            driver.close()
        except Exception as e:
            print(f"  ⚠️  Failed to get statistics: {e}")

    print()
    print("💡 Tip: You can view the knowledge graph in Neo4j Browser")
    print("   URL: http://localhost:7474")
    query = "MATCH (n:Node)-[r:RELATED]->(m:Node) RETURN n, r, m LIMIT 100"
    print(f"   Query: {query}")


if __name__ == "__main__":
    try:
        index_documents_offline()
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

