"""Subagent subgraph definition and wrapper"""

from typing import Literal

from langgraph.graph import END, START, StateGraph
from nodes.subagent.analysis import analysis_node
from nodes.subagent.retrieve import retrieve_node
from nodes.subagent.web_search import web_search_node
from retrieval import RetrievalResult
from schemas import SubagentState, VisitedSource


def route_source_necessity(
    state: SubagentState
) -> Literal["analysis_node", "web_search_node"]:
    """Gating Node: Check if internal knowledge is sufficient"""
    internal_result = state.get("internal_result")

    if internal_result:
        if isinstance(internal_result, RetrievalResult):
            if not internal_result.is_empty():
                print("  🚫 [Gating] Internal knowledge found. Skipping Web Search.")
                return "analysis_node"
        elif isinstance(internal_result, dict):
            # Handle dict representation (from state serialization)
            has_content = internal_result.get("has_content")
            content = internal_result.get("content", "")
            if has_content and not content.startswith("(No relevant"):
                print("  🚫 [Gating] Internal knowledge found. Skipping Web Search.")
                return "analysis_node"

    print("  🔄 [Gating] Internal knowledge insufficient. Routing to Web Search.")
    return "web_search_node"

# Define the Subgraph
subagent_workflow = StateGraph(SubagentState)
subagent_workflow.add_node("retrieve_node", retrieve_node)
subagent_workflow.add_node("web_search_node", web_search_node)
subagent_workflow.add_node("analysis_node", analysis_node)

subagent_workflow.add_edge(START, "retrieve_node")
subagent_workflow.add_conditional_edges(
    "retrieve_node",
    route_source_necessity,
    {
        "analysis_node": "analysis_node",
        "web_search_node": "web_search_node"
    }
)
subagent_workflow.add_edge("web_search_node", "analysis_node")
subagent_workflow.add_edge("analysis_node", END)

# Compile Subgraph
subagent_app = subagent_workflow.compile()


def subagent_node(state: SubagentState):
    """Wrapper to invoke the subagent subgraph and filter output"""
    # Invoke the subgraph
    # We pass the state directly. The subgraph runs and returns its final state.
    result_state = subagent_app.invoke(state)

    # Extract visited sources from retrieval results and convert to VisitedSource format
    visited_sources = []

    # Get internal result sources
    internal_result = result_state.get("internal_result")
    if internal_result:
        if isinstance(internal_result, RetrievalResult):
            for source in internal_result.sources:
                visited_sources.append(VisitedSource(
                    identifier=source.identifier,
                    source_type="internal"
                ))
        elif isinstance(internal_result, dict):
            for s in internal_result.get("sources", []):
                visited_sources.append(VisitedSource(
                    identifier=s.get("identifier", ""),
                    source_type="internal"
                ))

    # Get web result sources
    web_result = result_state.get("web_result")
    if web_result:
        if isinstance(web_result, RetrievalResult):
            for source in web_result.sources:
                visited_sources.append(VisitedSource(
                    identifier=source.identifier,
                    source_type="web"
                ))
        elif isinstance(web_result, dict):
            for s in web_result.get("sources", []):
                visited_sources.append(VisitedSource(
                    identifier=s.get("identifier", ""),
                    source_type="web"
                ))

    # Filter output: return ONLY what needs to be merged to the parent state
    # This avoids 'InvalidConcurrentGraphUpdate' on non-reducer keys
    # like 'subagent_tasks'
    return {
        "subagent_findings": result_state.get("subagent_findings", []),
        "visited_sources": visited_sources,
        "all_extracted_citations": result_state.get("extracted_citations", [])
    }

