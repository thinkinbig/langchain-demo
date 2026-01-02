"""Retrieve node: Retrieve internal knowledge (RAG)"""

from retrieval import RetrievalService
from schemas import SubagentState


def retrieve_node(state: SubagentState):
    """Node: Retrieve internal knowledge (RAG)"""
    tasks = state.get("subagent_tasks", [])
    if not tasks:
        return {}

    # Get task description
    rs_task = tasks[0]
    task_description = rs_task.description

    # Get visited sources from unified format
    visited_sources = state.get("visited_sources", [])
    visited_identifiers = [
        vs.identifier
        for vs in visited_sources
        if vs.source_type == "internal"
    ]

    print(f"  🧠 [RAG] Retrieving context for task: {task_description[:50]}...")

    # Use unified retrieval service
    result = RetrievalService.retrieve_internal(
        query=task_description,
        visited_sources=visited_identifiers,
        k=4
    )

    # Store in state using new unified format
    return {
        "internal_result": result,
        "task_description": task_description,  # Cache for downstream
    }

