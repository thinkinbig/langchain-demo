# Neo4j Visualization Guide

## Overview

Neo4j provides powerful native visualization tools that are better suited for exploring knowledge graphs than custom visualizations. This guide explains how to use Neo4j's visualization tools.

## Neo4j Browser

Neo4j Browser is Neo4j's built-in web interface, running by default at `http://localhost:7474`.

### Access

1. Start the Neo4j database
2. Open a browser and navigate to `http://localhost:7474`
3. Log in with your configured username and password (default: `neo4j` / `password`)

### Common Queries

#### View all nodes and relationships
```cypher
MATCH (n:Node)-[r:RELATED]->(m:Node)
RETURN n, r, m
LIMIT 100
```

#### View nodes of a specific type
```cypher
MATCH (n:Node)
WHERE n.type = 'Paper'
RETURN n
```

#### View node neighbors
```cypher
MATCH (n:Node {id: 'YourNodeID'})-[r:RELATED]-(neighbor:Node)
RETURN n, r, neighbor
```

#### View document-node mappings
```cypher
MATCH (n:Node)-[:APPEARS_IN]->(d:Document)
RETURN n, d
LIMIT 50
```

#### Community detection (requires GDS plugin)
```cypher
CALL gds.graph.project('kg-graph', 'Node', 'RELATED')
YIELD graphName

CALL gds.louvain.stream('kg-graph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).id AS node, communityId
ORDER BY communityId
```

## Neo4j Bloom

Neo4j Bloom is a more advanced visualization tool that provides:
- More beautiful interface
- Interactive exploration
- Pre-configured scenes and styles
- Better performance (suitable for large graphs)

### Usage

1. Install Neo4j Desktop
2. Start the database in Desktop
3. Open the Bloom application
4. Create scenes and style rules

## Custom Visualization vs Neo4j Native Tools

### Use Neo4j Browser/Bloom for:
- ✅ Exploring graph structure
- ✅ Viewing node and relationship details
- ✅ Interactive querying and filtering
- ✅ Large graph visualization (better performance)

### Use custom visualization for:
- ✅ Visualizing retrieval paths (query → entities → PPR → documents)
- ✅ Performance metrics and timeline visualization
- ✅ Integration with monitoring system, automatic report generation
- ✅ Displaying complete RAG retrieval flow

## Usage in Code

```python
from visualization.service import get_visualization_service
from memory.factory import create_graph_store

# Create graph store
graph_store = create_graph_store()

# Get visualization service
viz_service = get_visualization_service()

# Export Neo4j Browser usage guide
instructions_path = viz_service.visualize_graph(graph_store)
print(f"Neo4j Browser guide saved to: {instructions_path}")

# Visualize retrieval trace (custom visualization)
trace_viz = viz_service.visualize_retrieval_trace(trace_data)
```

## Recommendations

For knowledge graph structure visualization, **strongly recommend using Neo4j Browser or Bloom**, as they:
- Are more powerful
- Have better performance
- Are more interactive
- Require no additional code

Custom visualization is mainly used for:
- Retrieval path visualization (which Neo4j Browser cannot directly display)
- Performance monitoring and reporting
- Integration with monitoring systems

