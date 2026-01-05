"""Graph module - main entry point for graph access"""

# All imports are direct - no circular dependencies!
# - graph.helpers was moved to graph_utils.py (root level)
# - graph.routing was merged into graph.builder
# - route_source_necessity was moved to nodes/subagent/subgraph.py
# This breaks all circular dependency chains
from graph.builder import app

__all__ = ["app"]
