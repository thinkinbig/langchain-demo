"""State transformer for checkpointer serialization

This module provides state transformation functions to ensure
non-serializable fields are excluded before checkpoint serialization.
"""

from typing import Any, Dict


def transform_state_for_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform state before checkpoint serialization.
    Removes non-serializable fields like MessageChannel.
    
    Args:
        state: State dictionary to transform
        
    Returns:
        Transformed state dictionary with non-serializable fields removed
    """
    # Create a copy to avoid modifying the original
    transformed = dict(state)

    # Remove message_channel as it's not serializable
    if "message_channel" in transformed:
        del transformed["message_channel"]

    # Remove any other non-serializable runtime objects
    # (add more as needed)

    return transformed


def transform_state_from_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform state after loading from checkpoint.
    Recreates runtime objects that were excluded during serialization.
    
    Args:
        state: State dictionary loaded from checkpoint
        
    Returns:
        Transformed state dictionary with runtime objects recreated
    """
    # State loaded from checkpoint won't have message_channel
    # It will be recreated on-demand by nodes when needed
    # No transformation needed here

    return state

