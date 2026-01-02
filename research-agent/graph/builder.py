"""Graph builder: Main graph building logic and routing functions"""

from typing import Literal

from graph_helpers import should_retry
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from nodes.citation_agent import citation_agent_node
from nodes.decision import decision_node
from nodes.filter_findings import filter_findings_node
from nodes.lead_researcher import lead_researcher_node
from nodes.subagent.subgraph import subagent_node
from nodes.synthesizer import synthesizer_node
from nodes.verifier import verifier_node
from schemas import ResearchState

# ============================================================================
# Routing Functions
# ============================================================================


def assign_subagents(state: ResearchState):
    """Fan-out: Assign each task to a subagent OR retry"""
    # Check for retry condition
    if should_retry(state):
        print("  ↺ Routing back to lead_researcher for retry...")
        return "lead_researcher"

    tasks = state.get("subagent_tasks", [])
    print(f"\n🔀 [Fan-out] Distributing {len(tasks)} tasks to subagents...")

    # Send task objects directly
    # CONTEXT ISOLATION: We create a minimal state for the subagent
    # It receives ONLY the query and its specific task.
    # It does NOT receive the full history or other agent findings.
    return [
        Send(
            "subagent",
            {
                "subagent_tasks": [task],
                "query": state["query"],
                # Pass unified visited sources
                "visited_sources": state.get("visited_sources", [])
            },
        )
        for task in tasks
    ]


def route_decision(
    state: ResearchState,
) -> Literal["lead_researcher", "citation_agent"]:
    """Route based on decision node result"""
    if state.get("needs_more_research", False):
        return "lead_researcher"
    return "citation_agent"


def route_synthesizer(state: ResearchState) -> Literal["synthesizer", "decision"]:
    """Route synthesizer retry or success"""
    if should_retry(state):
        print("  ↺ Routing back to synthesizer for retry...")
        return "synthesizer"
    return "decision"


# ============================================================================
# Graph Building
# ============================================================================

# Build graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("lead_researcher", lead_researcher_node)
workflow.add_node("subagent", subagent_node)
workflow.add_node("filter_findings", filter_findings_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("decision", decision_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("citation_agent", citation_agent_node)

# Add edges
workflow.add_edge(START, "lead_researcher")
workflow.add_conditional_edges(
    "lead_researcher",
    assign_subagents,
    ["subagent", "lead_researcher"],
)
workflow.add_edge("subagent", "filter_findings")
workflow.add_edge("filter_findings", "synthesizer")
workflow.add_conditional_edges(
    "synthesizer",
    route_synthesizer,
    ["synthesizer", "decision"],
)
workflow.add_conditional_edges(
    "decision",
    route_decision,
    {
        "lead_researcher": "lead_researcher",
        "citation_agent": "verifier",
    },
)
workflow.add_edge("verifier", "citation_agent")
workflow.add_edge("citation_agent", END)

# Compile app
# Compile app with persistence (MemorySaver)
# This enables "Short-Term Memory" (thread-scoped)
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
