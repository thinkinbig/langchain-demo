"""
Subgraph isolation utilities for ensuring complete context isolation
between different subagents and subgraphs.
"""

from typing import Any, Dict, Optional


class SubgraphContextSandbox:
    """
    Context sandbox for subgraph isolation.

    Ensures that subgraphs can only access and modify their own state,
    preventing state leakage between parallel subagents.
    """

    def __init__(self, allowed_fields: Optional[list] = None):
        """
        Initialize context sandbox.

        Args:
            allowed_fields: List of field names that the subgraph can access/modify
        """
        self.allowed_fields = allowed_fields or []

    def sanitize_input_state(
        self,
        state: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sanitize input state to only include fields the subgraph should see.

        Args:
            state: Full state dictionary
            task_id: Optional task ID for additional filtering

        Returns:
            Sanitized state with only allowed fields
        """
        if not self.allowed_fields:
            # If no restrictions, return minimal state for subagent
            return {
                "query": state.get("query", ""),
                "subagent_tasks": state.get("subagent_tasks", []),
                "visited_sources": state.get("visited_sources", [])
            }

        # Filter to only allowed fields
        sanitized = {
            k: v for k, v in state.items()
            if k in self.allowed_fields
        }

        # Always include essential fields
        essential_fields = ["query", "subagent_tasks", "visited_sources"]
        for field in essential_fields:
            if field not in sanitized and field in state:
                sanitized[field] = state[field]

        return sanitized

    def validate_output_state(
        self,
        output: Dict[str, Any],
        node_name: str
    ) -> Dict[str, Any]:
        """
        Validate and filter output state to prevent unauthorized modifications.

        Args:
            output: Output state from subgraph
            node_name: Name of the node producing the output

        Returns:
            Validated and filtered output
        """
        # Define allowed output fields per node type
        allowed_outputs = {
            "subagent": [
                "subagent_findings",
                "visited_sources",
                "all_extracted_citations"
            ],
            "retrieve_node": [
                "internal_result",
                "web_result"
            ],
            "web_search_node": [
                "web_result",
                "visited_sources"
            ],
            "analysis_node": [
                "subagent_findings",
                "extracted_citations"
            ]
        }

        allowed = allowed_outputs.get(node_name, [])

        if not allowed:
            # Unknown node: return as-is (backward compatibility)
            return output

        # Filter to only allowed fields
        filtered = {
            k: v for k, v in output.items()
            if k in allowed
        }

        return filtered

    def create_isolated_config(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create isolated configuration for subgraph execution.

        Args:
            base_config: Base configuration
            task_id: Task ID for namespace isolation
            namespace: Optional explicit namespace

        Returns:
            Configuration with isolation settings
        """
        from memory.write_isolation import NamespaceIsolator

        # Generate namespace from task_id if provided
        if not namespace and task_id:
            namespace = NamespaceIsolator.get_subagent_namespace(task_id)

        # Create config with namespace
        config = NamespaceIsolator.create_config_with_namespace(
            base_config=base_config,
            namespace=namespace
        )

        return config


def create_subgraph_sandbox(node_name: str) -> SubgraphContextSandbox:
    """
    Create a context sandbox for a specific node type.

    Args:
        node_name: Name of the node/subgraph

    Returns:
        SubgraphContextSandbox instance
    """
    # Define allowed fields per node type
    field_policies = {
        "subagent": [
            "query",
            "subagent_tasks",
            "visited_sources",
            "internal_result",
            "web_result",
            "task_description"
        ],
        "retrieve_node": [
            "query",
            "subagent_tasks",
            "visited_sources"
        ],
        "web_search_node": [
            "query",
            "subagent_tasks",
            "visited_sources",
            "internal_result"
        ],
        "analysis_node": [
            "query",
            "subagent_tasks",
            "visited_sources",
            "internal_result",
            "web_result",
            "task_description"
        ]
    }

    allowed_fields = field_policies.get(node_name, [])
    return SubgraphContextSandbox(allowed_fields=allowed_fields)


def isolate_subgraph_state(
    state: Dict[str, Any],
    node_name: str,
    task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Isolate state for subgraph execution.

    This is a convenience function that combines sanitization and validation.

    Args:
        state: Full state dictionary
        node_name: Name of the node/subgraph
        task_id: Optional task ID for additional isolation

    Returns:
        Isolated state for subgraph
    """
    sandbox = create_subgraph_sandbox(node_name)
    return sandbox.sanitize_input_state(state, task_id=task_id)

