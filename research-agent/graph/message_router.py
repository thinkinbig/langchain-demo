"""Message router for intelligent message routing in LangGraph

This module provides routing logic based on message types and tags,
simplifying conditional edge logic in the graph builder.
"""

from typing import Literal

from graph.message_channel import MessageType
from graph.utils import should_retry
from schemas import ResearchState


class MessageRouter:
    """Router for message-based node routing decisions"""

    @staticmethod
    def route_after_lead_researcher(state: ResearchState) -> Literal["subagent", "lead_researcher"]:
        """Route after lead researcher: to subagents or retry"""
        # Check for retry condition
        if should_retry(state):
            print("  ↺ Routing back to lead_researcher for retry...")
            return "lead_researcher"

        tasks = state.get("subagent_tasks", [])
        if not tasks:
            print("  ⚠️  No tasks to distribute, routing back to lead_researcher")
            return "lead_researcher"

        # Check message channel for routing hints
        channel = state.get("message_channel")
        if channel and hasattr(channel, "receive"):
            # Check for error messages
            error_messages = channel.receive(
                "lead_researcher",
                message_type=MessageType.NODE_ERROR,
                limit=1
            )
            if error_messages:
                return "lead_researcher"

        return "subagent"

    @staticmethod
    def route_after_synthesizer(state: ResearchState) -> Literal["synthesizer", "decision"]:
        """Route after synthesizer: retry, early decision, or normal flow"""
        # Check for retry condition
        if should_retry(state):
            print("  ↺ Routing back to synthesizer for retry...")
            return "synthesizer"

        # Early decision optimization: if partial synthesis is done, route to decision
        partial_synthesis_done = state.get("partial_synthesis_done", False)
        early_decision_enabled = state.get("early_decision_enabled", True)

        if early_decision_enabled and partial_synthesis_done:
            print("  ⚡ Early decision: routing to decision after partial synthesis (S+C)")
            return "decision"

        # Normal flow: complete synthesis, route to decision
        return "decision"

    @staticmethod
    def route_after_decision(state: ResearchState) -> Literal["lead_researcher", "citation_agent"]:
        """Route after decision: continue research or finish"""
        needs_more = state.get("needs_more_research", False)

        # Check message channel for routing hints
        channel = state.get("message_channel")
        if channel and hasattr(channel, "receive"):
            # Check for decision messages
            decision_messages = channel.receive(
                "decision",
                message_type=MessageType.RESEARCH_DECISION,
                limit=1
            )
            if decision_messages:
                # Extract decision from message if available
                msg = decision_messages[0]
                if isinstance(msg.content, dict):
                    needs_more = msg.content.get("needs_more_research", needs_more)

        if needs_more:
            return "lead_researcher"
        return "citation_agent"

    @staticmethod
    def route_by_message_type(
        state: ResearchState,
        node_name: str,
        default_route: str
    ) -> str:
        """Route based on message type in channel"""
        channel = state.get("message_channel")
        if not channel or not hasattr(channel, "receive"):
            return default_route

        # Check for error messages
        error_messages = channel.receive(
            node_name,
            message_type=MessageType.NODE_ERROR,
            limit=1
        )
        if error_messages:
            # Route to error handler or retry
            return f"{node_name}_retry"

        # Check for control messages
        control_messages = channel.receive(
            node_name,
            message_type=MessageType.CONTROL,
            limit=1
        )
        if control_messages:
            msg = control_messages[0]
            if isinstance(msg.content, dict):
                target = msg.content.get("target_node")
                if target:
                    return target

        return default_route


# Convenience functions for use in graph builder
def route_decision(state: ResearchState) -> Literal["lead_researcher", "citation_agent"]:
    """Route based on decision node result"""
    return MessageRouter.route_after_decision(state)


def route_synthesizer(state: ResearchState) -> Literal["synthesizer", "decision"]:
    """Route synthesizer retry, partial synthesis, or success"""
    return MessageRouter.route_after_synthesizer(state)

