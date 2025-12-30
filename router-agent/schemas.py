from typing import TypedDict


class RouterState(TypedDict):
    """State for router agent"""
    task: str  # User task/query
    task_type: str  # Detected task type (e.g., "code", "analysis", "writing", "calculation")
    routed_to: str  # Which expert node was routed to
    expert_output: str  # Output from the expert
    routing_reason: str  # Why this route was chosen

