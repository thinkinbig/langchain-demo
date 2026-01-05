#!/usr/bin/env python3
"""
检查Neo4j中的知识图谱数据
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from neo4j import GraphDatabase


def check_neo4j_data():
    """检查Neo4j中存储的数据"""
    print("=" * 80)
    print("Neo4j 知识图谱数据检查")
    print("=" * 80)
    print("\n连接信息:")
    print(f"  URI: {settings.NEO4J_URI}")
    print(f"  Database: {settings.NEO4J_DATABASE}")
    print(f"  Username: {settings.NEO4J_USERNAME}")
    print()

    try:
        driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            # 1. 检查所有节点标签
            print("1. 节点标签统计:")
            print("-" * 80)
            result = session.run("CALL db.labels()")
            labels = [record["label"] for record in result]
            print(f"   找到的标签: {labels}")
            print()

            # 2. 检查Node节点数量
            if "Node" in labels:
                result = session.run("MATCH (n:Node) RETURN count(n) AS count")
                node_count = result.single()["count"]
                print(f"2. Node节点数量: {node_count}")

                if node_count > 0:
                    # 查看前10个节点
                    result = session.run("""
                        MATCH (n:Node)
                        RETURN n.id AS id, n.type AS type, n.description AS description
                        LIMIT 10
                    """)
                    print("\n   前10个节点示例:")
                    for record in result:
                        desc = record["description"][:50] if record["description"] else ""
                        print(f"     - {record['id']} (type: {record['type']}, desc: {desc}...)")
            else:
                print("2. Node节点: 未找到")
            print()

            # 3. 检查Document节点数量
            if "Document" in labels:
                result = session.run("MATCH (d:Document) RETURN count(d) AS count")
                doc_count = result.single()["count"]
                print(f"3. Document节点数量: {doc_count}")

                if doc_count > 0:
                    # 查看前5个文档
                    result = session.run("""
                        MATCH (d:Document)
                        RETURN d.source AS source, d.collection AS collection
                        LIMIT 5
                    """)
                    print("\n   前5个文档示例:")
                    for record in result:
                        print(f"     - {record['source']} (collection: {record.get('collection', 'N/A')})")
            else:
                print("3. Document节点: 未找到")
            print()

            # 4. 检查关系类型
            print("4. 关系类型统计:")
            print("-" * 80)
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record["relationshipType"] for record in result]
            print(f"   找到的关系类型: {rel_types}")
            print()

            # 5. 检查RELATED关系数量
            if "RELATED" in rel_types:
                result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS count")
                rel_count = result.single()["count"]
                print(f"5. RELATED关系数量: {rel_count}")

                if rel_count > 0:
                    # 查看前5个关系
                    result = session.run("""
                        MATCH (n:Node)-[r:RELATED]->(m:Node)
                        RETURN n.id AS source, r.relation AS relation, m.id AS target
                        LIMIT 5
                    """)
                    print("\n   前5个关系示例:")
                    for record in result:
                        print(f"     - {record['source']} --[{record.get('relation', 'related_to')}]--> {record['target']}")
            else:
                print("5. RELATED关系: 未找到")
            print()

            # 6. 检查APPEARS_IN关系
            if "APPEARS_IN" in rel_types:
                result = session.run("MATCH ()-[r:APPEARS_IN]->() RETURN count(r) AS count")
                appears_count = result.single()["count"]
                print(f"6. APPEARS_IN关系数量: {appears_count}")
            else:
                print("6. APPEARS_IN关系: 未找到")
            print()

            # 7. 检查是否有Node到Document的连接
            result = session.run("""
                MATCH (n:Node)-[:APPEARS_IN]->(d:Document)
                RETURN count(*) AS count
            """)
            node_doc_count = result.single()["count"]
            print(f"7. Node到Document的连接数: {node_doc_count}")
            print()

            # 8. 按类型统计节点
            if "Node" in labels:
                result = session.run("""
                    MATCH (n:Node)
                    RETURN n.type AS type, count(n) AS count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                print("8. 节点类型统计 (Top 10):")
                print("-" * 80)
                for record in result:
                    print(f"     {record['type']}: {record['count']} 个节点")
            print()

            # 9. 提供查询建议
            print("=" * 80)
            print("查询建议:")
            print("=" * 80)
            print("\n在Neo4j Browser中运行以下查询来查看数据:\n")

            if "Node" in labels:
                print("1. 查看所有Node节点和RELATED关系:")
                print("   MATCH (n:Node)-[r:RELATED]->(m:Node)")
                print("   RETURN n, r, m")
                print("   LIMIT 100")
                print()

                print("2. 查看所有Node节点（不包含关系）:")
                print("   MATCH (n:Node)")
                print("   RETURN n")
                print("   LIMIT 100")
                print()

            if "Document" in labels:
                print("3. 查看Document节点:")
                print("   MATCH (d:Document)")
                print("   RETURN d")
                print("   LIMIT 50")
                print()

            if "Node" in labels and "Document" in labels:
                print("4. 查看Node和Document的连接:")
                print("   MATCH (n:Node)-[:APPEARS_IN]->(d:Document)")
                print("   RETURN n, d")
                print("   LIMIT 50")
                print()

            print("5. 查看完整的知识图谱（包含所有节点和关系）:")
            print("   MATCH (n)")
            print("   OPTIONAL MATCH (n)-[r]->(m)")
            print("   RETURN n, r, m")
            print("   LIMIT 200")
            print()

        driver.close()
        print("\n✅ 检查完成!")

    except Exception as e:
        print(f"\n❌ 连接Neo4j失败: {e}")
        print("\n请检查:")
        print("  1. Neo4j是否正在运行")
        print("  2. 连接信息是否正确 (URI, username, password)")
        print("  3. 数据库名称是否正确")
        sys.exit(1)


if __name__ == "__main__":
    check_neo4j_data()

