"""State utility functions for simplified state access and updates

This module provides helper functions for working with ResearchState,
simplifying common operations and ensuring consistency.
"""

from typing import Any, Dict, List, Optional, Set

from schemas import Finding


def get_node_messages(
    state: Dict[str, Any],
    node_name: str,
    default: Optional[List] = None
) -> List:
    """Get messages for a specific node from state"""
    if default is None:
        default = []

    # Get messages from unified message channel
    channel = state.get("message_channel")
    if channel and hasattr(channel, "receive"):
        return channel.receive(node_name)

    return default


def set_node_messages(
    state: Dict[str, Any],
    node_name: str,
    messages: List
) -> Dict[str, Any]:
    """Set messages for a specific node in state"""
    # Add messages to unified message channel
    channel = state.get("message_channel")
    if channel and hasattr(channel, "add_langchain_messages"):
        channel.add_langchain_messages(messages, node_name)
        # Don't return message_channel (not serializable)
        return {}

    return {}


def get_processed_items(
    state: Dict[str, Any],
    item_type: str = "findings"
) -> Set[str]:
    """Get set of processed item IDs"""
    if item_type == "findings":
        field = "processed_findings_ids"
    elif item_type == "findings_sent":
        field = "sent_finding_hashes"
    else:
        field = f"processed_{item_type}_ids"

    items = state.get(field, [])
    return set(items) if items else set()


def mark_item_processed(
    state: Dict[str, Any],
    item_id: str,
    item_type: str = "findings"
) -> Dict[str, Any]:
    """Mark an item as processed"""
    if item_type == "findings":
        field = "processed_findings_ids"
    elif item_type == "findings_sent":
        field = "sent_finding_hashes"
    else:
        field = f"processed_{item_type}_ids"

    current = state.get(field, [])
    if item_id not in current:
        current = list(current) + [item_id]

    return {field: current}


def get_new_findings(
    state: Dict[str, Any],
    all_findings: List[Finding]
) -> List[Finding]:
    """Get findings that haven't been processed yet"""
    processed_ids = get_processed_items(state, "findings")

    new_findings = []
    for finding in all_findings:
        # Compute finding ID (same logic as in nodes)
        if isinstance(finding, dict):
            task = finding.get("task", "")
            summary = finding.get("summary", "")
        else:
            task = finding.task if hasattr(finding, "task") else ""
            summary = finding.summary if hasattr(finding, "summary") else ""

        import hashlib
        finding_id = hashlib.sha256((task + summary).encode()).hexdigest()[:16]

        if finding_id not in processed_ids:
            new_findings.append(finding)

    return new_findings


def get_new_findings_for_lead_researcher(
    state: Dict[str, Any],
    all_findings: List[Finding]
) -> List[Finding]:
    """Get findings that haven't been sent to lead researcher yet"""
    sent_ids = get_processed_items(state, "findings_sent")

    new_findings = []
    for finding in all_findings:
        # Compute finding ID
        if isinstance(finding, dict):
            task = finding.get("task", "")
            summary = finding.get("summary", "")
        else:
            task = finding.task if hasattr(finding, "task") else ""
            summary = finding.summary if hasattr(finding, "summary") else ""

        import hashlib
        finding_id = hashlib.sha256((task + summary).encode()).hexdigest()[:16]

        if finding_id not in sent_ids:
            new_findings.append(finding)

    return new_findings


def update_previous_metrics(
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Update previous iteration metrics from current state"""
    return {
        "previous_synthesis_length": len(state.get("synthesized_results", "")),
        "previous_findings_count": len(state.get("subagent_findings", [])),
        "previous_citation_count": len(state.get("all_extracted_citations", [])),
    }


def get_state_snapshot(
    state: Dict[str, Any],
    fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get a snapshot of specific state fields"""
    if fields is None:
        # Default fields for common use cases
        fields = [
            "query",
            "iteration_count",
            "subagent_findings",
            "synthesized_results",
            "needs_more_research",
            "decision_reasoning",
        ]

    snapshot = {}
    for field in fields:
        if field in state:
            snapshot[field] = state[field]

    return snapshot


def merge_state_updates(
    *updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge multiple state update dictionaries"""
    merged = {}
    for update in updates:
        if update:
            merged.update(update)
    return merged


def extract_node_state(
    state: Dict[str, Any],
    node_name: str
) -> Dict[str, Any]:
    """Extract state relevant to a specific node"""
    # Node-specific state extraction logic
    node_state = {
        "query": state.get("query", ""),
        "iteration_count": state.get("iteration_count", 0),
    }

    if node_name == "lead_researcher":
        node_state.update({
            "scratchpad": state.get("scratchpad", ""),
            "subagent_findings": state.get("subagent_findings", []),
            "decision_reasoning": state.get("decision_reasoning"),
            "decision_key_factors": state.get("decision_key_factors", []),
            "selected_approach": state.get("selected_approach"),
            "selection_reasoning": state.get("selection_reasoning"),
            "complexity_analysis": state.get("complexity_analysis"),
        })
    elif node_name == "synthesizer":
        node_state.update({
            "subagent_findings": state.get("subagent_findings", []),
            "filtered_findings": state.get("filtered_findings", []),
            "synthesized_results": state.get("synthesized_results", ""),
            "reflection_analysis": state.get("reflection_analysis"),
            "complexity_analysis": state.get("complexity_analysis"),
        })
    elif node_name == "decision":
        node_state.update({
            "subagent_findings": state.get("subagent_findings", []),
            "synthesized_results": state.get("synthesized_results", ""),
            "all_extracted_citations": state.get("all_extracted_citations", []),
            "complexity_analysis": state.get("complexity_analysis"),
            "has_partial_synthesis": state.get("has_partial_synthesis", False),
        })
    elif node_name == "verifier":
        node_state.update({
            "final_report": state.get("final_report", ""),
        })
    elif node_name == "citation_agent":
        node_state.update({
            "subagent_findings": state.get("subagent_findings", []),
            "synthesized_results": state.get("synthesized_results", ""),
            "all_extracted_citations": state.get("all_extracted_citations", []),
        })

    return node_state

