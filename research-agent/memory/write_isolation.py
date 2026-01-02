"""
Write isolation utilities for context engineering.

Provides mechanisms to isolate state writes between different nodes,
subgraphs, and agents to prevent state pollution.
"""

from typing import Any, Dict, Optional, Set


class WriteFilter:
    """
    Filter that controls which state fields can be written by a node.

    This enables write isolation by restricting what each node/subgraph
    can modify in the shared state.
    """

    def __init__(
        self,
        allowed_fields: Optional[Set[str]] = None,
        denied_fields: Optional[Set[str]] = None,
        allow_all: bool = False
    ):
        """
        Initialize write filter.

        Args:
            allowed_fields: Set of field names that CAN be written (whitelist)
            denied_fields: Set of field names that CANNOT be written (blacklist)
            allow_all: If True, allow all fields (default: False)

        Note:
            If both allowed_fields and denied_fields are provided,
            allowed_fields takes precedence (whitelist mode).
        """
        self.allowed_fields = allowed_fields or set()
        self.denied_fields = denied_fields or set()
        self.allow_all = allow_all

    def filter_state_update(
        self,
        state_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter state update to only include allowed fields.

        Args:
            state_update: Dictionary of state updates

        Returns:
            Filtered dictionary with only allowed fields
        """
        if self.allow_all:
            return state_update

        if self.allowed_fields:
            # Whitelist mode: only allow specified fields
            return {
                k: v for k, v in state_update.items()
                if k in self.allowed_fields
            }

        if self.denied_fields:
            # Blacklist mode: deny specified fields
            return {
                k: v for k, v in state_update.items()
                if k not in self.denied_fields
            }

        # Default: allow all if no restrictions
        return state_update


class NamespaceIsolator:
    """
    Manages namespace-based isolation for checkpointers.

    Uses LangGraph's configurable namespace feature to isolate
    state writes between different graphs/subgraphs.
    """

    @staticmethod
    def create_config_with_namespace(
        base_config: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create config dict with namespace isolation.

        Args:
            base_config: Base configuration dict
            namespace: Namespace identifier for isolation
            thread_id: Thread ID for state tracking

        Returns:
            Configuration dict with namespace isolation
        """
        config = base_config or {}

        # Ensure configurable dict exists
        if "configurable" not in config:
            config["configurable"] = {}

        # Set namespace if provided
        if namespace:
            config["configurable"]["namespace"] = namespace

        # Set thread_id if provided
        if thread_id:
            config["configurable"]["thread_id"] = thread_id

        return config

    @staticmethod
    def get_subagent_namespace(task_id: str) -> str:
        """
        Generate namespace for a subagent based on task ID.

        Args:
            task_id: Task identifier

        Returns:
            Namespace string for the subagent
        """
        return f"subagent_{task_id}"

    @staticmethod
    def get_node_namespace(node_name: str, graph_name: Optional[str] = None) -> str:
        """
        Generate namespace for a node.

        Args:
            node_name: Name of the node
            graph_name: Optional graph name for hierarchical namespaces

        Returns:
            Namespace string for the node
        """
        if graph_name:
            return f"{graph_name}.{node_name}"
        return f"node_{node_name}"


def create_write_filter_for_node(node_name: str) -> WriteFilter:
    """
    Create a write filter for a specific node based on node name.

    This provides default write isolation policies for different node types.

    Args:
        node_name: Name of the node

    Returns:
        WriteFilter instance configured for the node
    """
    # Define allowed fields per node type
    node_policies: Dict[str, Set[str]] = {
        "lead_researcher": {
            "research_plan",
            "subagent_tasks",
            "scratchpad",
            "iteration_count",
            "error",
            "retry_count"
        },
        "subagent": {
            "subagent_findings",
            "visited_sources",
            "all_extracted_citations"
        },
        "filter_findings": {
            "filtered_findings"
        },
        "synthesizer": {
            "synthesized_results",
            "error",
            "retry_count"
        },
        "decision": {
            "needs_more_research"
        },
        "verifier": {
            "final_report"
        },
        "citation_agent": {
            "citations",
            "final_report"
        }
    }

    allowed_fields = node_policies.get(node_name)

    if allowed_fields:
        return WriteFilter(allowed_fields=allowed_fields)
    else:
        # Unknown node: allow all (backward compatibility)
        return WriteFilter(allow_all=True)


def apply_write_isolation(
    state_update: Dict[str, Any],
    node_name: str,
    namespace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Apply write isolation to a state update.

    This is a convenience function that combines filtering and namespacing.

    Args:
        state_update: State update dictionary
        node_name: Name of the node making the update
        namespace: Optional namespace for additional isolation

    Returns:
        Filtered and namespaced state update
    """
    # Apply write filter
    filter_obj = create_write_filter_for_node(node_name)
    filtered_update = filter_obj.filter_state_update(state_update)

    # Note: Namespace is typically applied at the config level,
    # not in the state update itself. This function is for documentation
    # and future extensibility.

    return filtered_update

