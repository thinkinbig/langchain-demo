"""Unified message channel system for LangGraph nodes

This module provides a standardized message passing mechanism for nodes,
replacing the scattered message lists with a unified channel-based approach.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class MessageType(str, Enum):
    """Message type classification for routing and processing"""
    # Node communication
    NODE_INPUT = "node_input"
    NODE_OUTPUT = "node_output"
    NODE_ERROR = "node_error"

    # Research workflow
    RESEARCH_QUERY = "research_query"
    RESEARCH_TASK = "research_task"
    RESEARCH_FINDING = "research_finding"
    RESEARCH_SYNTHESIS = "research_synthesis"
    RESEARCH_DECISION = "research_decision"

    # LLM conversation
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_SYSTEM = "llm_system"

    # Feedback and control
    FEEDBACK = "feedback"
    CONTROL = "control"
    RETRY = "retry"


class MessageTag(str, Enum):
    """Message tags for categorization and filtering"""
    LEAD_RESEARCHER = "lead_researcher"
    SYNTHESIZER = "synthesizer"
    DECISION = "decision"
    VERIFIER = "verifier"
    CITATION_AGENT = "citation_agent"
    SUBAGENT = "subagent"

    # Content tags
    PLANNING = "planning"
    SYNTHESIS = "synthesis"
    VERIFICATION = "verification"
    CITATION = "citation"

    # Status tags
    INCREMENTAL = "incremental"
    RETRY = "retry"
    ERROR = "error"


@dataclass
class NodeMessage:
    """Standardized message format for node communication"""
    message_type: MessageType
    tag: MessageTag
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source_node: Optional[str] = None
    target_node: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_type": self.message_type.value,
            "tag": self.tag.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source_node": self.source_node,
            "target_node": self.target_node,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeMessage":
        """Create message from dictionary"""
        return cls(
            message_type=MessageType(data["message_type"]),
            tag=MessageTag(data["tag"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_node=data.get("source_node"),
            target_node=data.get("target_node"),
        )


class MessageChannel:
    """Unified message channel for node communication"""

    def __init__(self, channel_id: str = "default"):
        self.channel_id = channel_id
        self._messages: List[NodeMessage] = []
        self._index_by_tag: Dict[MessageTag, List[int]] = {}
        self._index_by_type: Dict[MessageType, List[int]] = {}
        self._index_by_node: Dict[str, List[int]] = {}

    def send(
        self,
        message_type: MessageType,
        tag: MessageTag,
        content: Any,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NodeMessage:
        """Send a message to the channel"""
        message = NodeMessage(
            message_type=message_type,
            tag=tag,
            content=content,
            source_node=source_node,
            target_node=target_node,
            metadata=metadata or {}
        )

        # Add to messages list
        index = len(self._messages)
        self._messages.append(message)

        # Update indices
        if tag not in self._index_by_tag:
            self._index_by_tag[tag] = []
        self._index_by_tag[tag].append(index)

        if message_type not in self._index_by_type:
            self._index_by_type[message_type] = []
        self._index_by_type[message_type].append(index)

        if source_node:
            if source_node not in self._index_by_node:
                self._index_by_node[source_node] = []
            self._index_by_node[source_node].append(index)

        if target_node:
            if target_node not in self._index_by_node:
                self._index_by_node[target_node] = []
            self._index_by_node[target_node].append(index)

        return message

    def receive(
        self,
        node_name: str,
        message_type: Optional[MessageType] = None,
        tag: Optional[MessageTag] = None,
        limit: Optional[int] = None
    ) -> List[NodeMessage]:
        """Receive messages for a specific node"""
        indices = set()

        # Filter by target node
        if node_name in self._index_by_node:
            indices.update(self._index_by_node[node_name])

        # Filter by message type
        if message_type:
            type_indices = set(self._index_by_type.get(message_type, []))
            indices = indices.intersection(type_indices) if indices else type_indices

        # Filter by tag
        if tag:
            tag_indices = set(self._index_by_tag.get(tag, []))
            indices = indices.intersection(tag_indices) if indices else tag_indices

        # Get messages and sort by timestamp
        messages = [self._messages[i] for i in sorted(indices)]

        if limit:
            messages = messages[-limit:]  # Get most recent

        return messages

    def get_all(self, limit: Optional[int] = None) -> List[NodeMessage]:
        """Get all messages in the channel"""
        messages = self._messages.copy()
        if limit:
            messages = messages[-limit:]
        return messages

    def clear(self):
        """Clear all messages from the channel"""
        self._messages.clear()
        self._index_by_tag.clear()
        self._index_by_type.clear()
        self._index_by_node.clear()

    def to_langchain_messages(
        self,
        node_name: str,
        include_system: bool = True
    ) -> List[BaseMessage]:
        """Convert channel messages to LangChain message format for a node"""
        messages = self.receive(node_name)
        langchain_messages = []

        for msg in messages:
            if msg.message_type == MessageType.LLM_SYSTEM and include_system:
                langchain_messages.append(SystemMessage(content=str(msg.content)))
            elif msg.message_type == MessageType.LLM_REQUEST:
                langchain_messages.append(HumanMessage(content=str(msg.content)))
            elif msg.message_type == MessageType.LLM_RESPONSE:
                langchain_messages.append(AIMessage(content=str(msg.content)))

        return langchain_messages

    def add_langchain_messages(
        self,
        messages: List[BaseMessage],
        node_name: str,
        message_type: Optional[MessageType] = None
    ):
        """Add LangChain messages to the channel"""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                msg_type = message_type or MessageType.LLM_SYSTEM
                tag = (
                    MessageTag.LEAD_RESEARCHER
                    if "lead" in node_name.lower()
                    else MessageTag.SYNTHESIZER
                )
                self.send(
                    message_type=msg_type,
                    tag=tag,
                    content=msg.content,
                    source_node=node_name
                )
            elif isinstance(msg, HumanMessage):
                msg_type = message_type or MessageType.LLM_REQUEST
                tag = (
                    MessageTag.LEAD_RESEARCHER
                    if "lead" in node_name.lower()
                    else MessageTag.SYNTHESIZER
                )
                self.send(
                    message_type=msg_type,
                    tag=tag,
                    content=msg.content,
                    source_node=node_name
                )
            elif isinstance(msg, AIMessage):
                msg_type = message_type or MessageType.LLM_RESPONSE
                tag = (
                    MessageTag.LEAD_RESEARCHER
                    if "lead" in node_name.lower()
                    else MessageTag.SYNTHESIZER
                )
                self.send(
                    message_type=msg_type,
                    tag=tag,
                    content=msg.content,
                    source_node=node_name
                )


def create_message_channel(channel_id: str = "default") -> MessageChannel:
    """Factory function to create a message channel"""
    return MessageChannel(channel_id=channel_id)

