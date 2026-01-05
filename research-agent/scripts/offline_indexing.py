#!/usr/bin/env python3
"""
Unified Offline Indexing Script

Index documents into both vector store and knowledge graph in one batch.
This script combines document chunk indexing (for RAG) and knowledge graph
indexing (for graph-based retrieval) into a single offline process.
"""

import os
import sys

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Find and load .env file (must be before importing config)
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    root_env = os.path.join(script_dir, "..", ".env")
    if os.path.exists(root_env):
        load_dotenv(root_env)
    else:
        load_dotenv()

from config import settings  # noqa: E402
from context_manager import VectorStoreManager  # noqa: E402


def main():
    """Unified offline indexing: vector store + knowledge graph."""
    print("=" * 80)
    print("Unified Offline Indexing")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Index document chunks into vector store (for semantic search)")
    print("  2. Index entities and relationships into knowledge graph")
    print("  3. Link knowledge graph nodes to document chunks")
    print()
    print("📊 Configuration:")
    print(f"   Vector Store Backend: {settings.VECTOR_STORE_BACKEND}")
    print(f"   Graph Store Backend: {settings.GRAPH_STORE_BACKEND}")
    print(f"   Graph Enabled: {settings.GRAPH_ENABLED}")
    print("   Data Directory: research-agent/data")
    print()

    # Check Neo4j connection if using Neo4j
    if settings.VECTOR_STORE_BACKEND == "neo4j" or settings.GRAPH_STORE_BACKEND == "neo4j":
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
            print("   Please check if Neo4j is running and connection config is correct")
            return
        print()

    # Initialize vector store manager
    print("🔧 Initializing vector store manager...")
    try:
        manager = VectorStoreManager()
        print("  ✅ Initialization complete")
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 80)
    print("Starting unified indexing...")
    print("=" * 80)
    print()

    # Check current counts
    current_vector_count = manager.vector_store.get_collection_count()
    print(f"📊 Current chunks in vector store: {current_vector_count}")

    if settings.GRAPH_ENABLED and settings.GRAPH_STORE_BACKEND == "neo4j":
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run("MATCH (n:Node) RETURN count(n) AS count")
                current_node_count = result.single()["count"]
                result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS count"
                )
                current_edge_count = result.single()["count"]
            driver.close()
            print(f"📊 Current nodes in knowledge graph: {current_node_count}")
            print(f"📊 Current edges in knowledge graph: {current_edge_count}")
        except Exception as e:
            print(f"  ⚠️  Failed to get graph statistics: {e}")
    print()

    # Ingest documents (this will index both vector store and knowledge graph)
    try:
        manager.ingest_documents()
    except Exception as e:
        print(f"  ❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Check final counts
    final_vector_count = manager.vector_store.get_collection_count()
    print()
    print("=" * 80)
    print("Indexing complete")
    print("=" * 80)
    print(f"✅ Total chunks in vector store: {final_vector_count}")
    print(f"   Added: {final_vector_count - current_vector_count} chunks")
    print()

    if settings.GRAPH_ENABLED and settings.GRAPH_STORE_BACKEND == "neo4j":
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                # Node count
                result = session.run("MATCH (n:Node) RETURN count(n) AS count")
                final_node_count = result.single()["count"]
                print(f"✅ Total nodes in knowledge graph: {final_node_count}")
                print(f"   Added: {final_node_count - current_node_count} nodes")

                # Edge count (all relationship types)
                result = session.run(
                    "MATCH ()-[r]->() RETURN count(r) AS count"
                )
                final_edge_count = result.single()["count"]
                print(f"✅ Total edges in knowledge graph: {final_edge_count}")
                print(f"   Added: {final_edge_count - current_edge_count} edges")

                # Node-Document chunk links
                result = session.run(
                    "MATCH (n:Node)-[:MENTIONED_IN]->(d:Document) "
                    "RETURN count(*) AS count"
                )
                link_count = result.single()["count"]
                print(f"✅ Node-Document chunk links: {link_count}")

                # Relationship types distribution
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) AS rel_type, count(*) AS count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                print()
                print("📊 Top relationship types:")
                for record in result:
                    print(f"   {record['rel_type']}: {record['count']}")

            driver.close()
        except Exception as e:
            print(f"  ⚠️  Failed to get final statistics: {e}")

    print()
    print("🔍 Verification queries for Neo4j Browser:")
    print()
    print("1. Count document chunks:")
    print("   MATCH (d:Document)")
    print("   WHERE d.collection = 'enterprise_knowledge'")
    print("   RETURN count(d) AS total_chunks")
    print()
    print("2. Count knowledge graph nodes:")
    print("   MATCH (n:Node)")
    print("   RETURN count(n) AS total_nodes")
    print()
    print("3. View node-document chunk relationships:")
    print("   MATCH (n:Node)-[:MENTIONED_IN]->(d:Document)")
    print("   RETURN n.id AS node, d.source AS document")
    print("   LIMIT 20")
    print()
    print("4. View relationship types:")
    print("   MATCH ()-[r]->()")
    print("   RETURN type(r) AS relation_type, count(*) AS count")
    print("   ORDER BY count DESC")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

