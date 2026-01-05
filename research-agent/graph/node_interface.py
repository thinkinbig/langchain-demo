"""Standardized node interface for message passing

This module defines the standard interface that all nodes should use
for sending and receiving messages, ensuring consistency across the graph.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from graph.message_channel import (
    MessageChannel,
    MessageTag,
    MessageType,
    create_message_channel,
)
from graph.state_utils import (
    extract_node_state,
)
from langchain_core.messages import BaseMessage


class NodeInterface(ABC):
    """Abstract base class for node interfaces"""

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the node logic and return state updates"""
        pass


class MessageAwareNode:
    """Mixin class that provides message channel awareness to nodes"""

    def __init__(self, node_name: str):
        self.node_name = node_name

    def get_message_channel(
        self, state: Dict[str, Any], create_if_missing: bool = True
    ) -> Optional[MessageChannel]:
        """Get or create message channel from state"""
        # Get channel from state (may be None if not yet created)
        # Use getattr for Pydantic models, get() for dicts
        if hasattr(state, "get"):
            channel = state.get("message_channel")
        else:
            channel = getattr(state, "message_channel", None)

        if channel is None and create_if_missing:
            # Create channel but don't store in state (not serializable)
            # Channel is ephemeral and recreated as needed
            channel = create_message_channel()
            # Try to store temporarily, but handle both dict and Pydantic model
            try:
                if isinstance(state, dict):
                    state["message_channel"] = channel
                elif hasattr(state, "__dict__"):
                    # Pydantic model - use setattr (won't be serialized anyway)
                    state.message_channel = channel
            except (TypeError, AttributeError):
                # If we can't store it, that's fine - channel is ephemeral
                pass
        return channel

    def send_message(
        self,
        state: Dict[str, Any],
        message_type: MessageType,
        tag: MessageTag,
        content: Any,
        target_node: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a message through the message channel"""
        channel = self.get_message_channel(state)
        if channel:
            channel.send(
                message_type=message_type,
                tag=tag,
                content=content,
                source_node=self.node_name,
                target_node=target_node,
                metadata=metadata or {},
            )

    def receive_messages(
        self,
        state: Dict[str, Any],
        message_type: Optional[MessageType] = None,
        tag: Optional[MessageTag] = None,
        limit: Optional[int] = None,
    ) -> List:
        """Receive messages for this node"""
        channel = self.get_message_channel(state, create_if_missing=False)
        if channel:
            return channel.receive(
                self.node_name, message_type=message_type, tag=tag, limit=limit
            )
        return []

    def get_langchain_messages(
        self, state: Dict[str, Any], include_system: bool = True
    ) -> List[BaseMessage]:
        """Get LangChain messages for this node"""
        channel = self.get_message_channel(state, create_if_missing=False)
        if channel:
            return channel.to_langchain_messages(
                self.node_name, include_system=include_system
            )
        return []

    def add_langchain_messages(
        self, state: Dict[str, Any], messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Add LangChain messages to the channel"""
        channel = self.get_message_channel(state)
        if channel:
            channel.add_langchain_messages(messages, self.node_name)
            # Don't return message_channel in state update - it's not serializable
            return {}
        return {}

    def get_node_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get state relevant to this node"""
        return extract_node_state(state, self.node_name)


def create_node_wrapper(node_func, node_name: str):
    """Create a wrapper function that adds message awareness to a node"""

    class WrappedNode(MessageAwareNode):
        def __init__(self, node_func, node_name):
            super().__init__(node_name)
            self.node_func = node_func

        def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
            """Process node with message awareness"""
            # Ensure message channel exists
            self.get_message_channel(state)

            # Call original node function
            result = self.node_func(state)

            # Ensure message channel is preserved in result
            if result and "message_channel" not in result:
                channel = state.get("message_channel")
                if channel:
                    result["message_channel"] = channel

            return result

    wrapped = WrappedNode(node_func, node_name)
    return wrapped.process



