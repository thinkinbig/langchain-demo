"""Graph builder: Main graph building logic and routing functions"""

from typing import Literal

from graph.utils import should_retry
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from memory.checkpointer_factory import get_checkpointer
from nodes.citation_agent import citation_agent_node
from nodes.complexity_analyzer import complexity_analyzer_node
from nodes.decision import decision_node
from nodes.filter_findings import filter_findings_node
from nodes.lead_researcher import lead_researcher_node
from nodes.strategy_evaluator import strategy_evaluator_node
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

    # Respect recommended_workers from complexity analysis
    complexity_analysis = state.get("complexity_analysis")
    recommended_workers = None
    if complexity_analysis:
        if hasattr(complexity_analysis, "recommended_workers"):
            recommended_workers = complexity_analysis.recommended_workers
        elif isinstance(complexity_analysis, dict):
            recommended_workers = complexity_analysis.get("recommended_workers")

    # Limit tasks to recommended_workers if specified
    tasks_to_distribute = tasks
    if recommended_workers is not None and len(tasks) > recommended_workers:
        tasks_to_distribute = tasks[:recommended_workers]
        print(f"\n🔀 [Fan-out] Distributing {len(tasks_to_distribute)} of {len(tasks)} tasks to subagents...")
        print(f"   ⚠️  Limited to {recommended_workers} workers based on complexity analysis")
    else:
        print(f"\n🔀 [Fan-out] Distributing {len(tasks_to_distribute)} tasks to subagents...")

    # Send task objects directly
    # CONTEXT ISOLATION: We create a minimal state for the subagent
    # It receives ONLY the query and its specific task.
    # It does NOT receive the full history or other agent findings.
    # WRITE ISOLATION: Each subagent gets its own namespace for state writes
    return [
        Send(
            "subagent",
            {
                "subagent_tasks": [task],
                "query": state["query"],
                # Pass unified visited sources
                "visited_sources": state.get("visited_sources", []),
                # Metadata for namespace isolation (used in node wrapper)
                "_task_id": task.id if hasattr(task, 'id') else f"task_{i}",
            },
        )
        for i, task in enumerate(tasks_to_distribute)
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


def route_after_complexity(
    state: ResearchState,
) -> Literal["lead_researcher"]:
    """
    Route after complexity analysis: Always go to lead_researcher first.
    Lead researcher will generate strategies if needed, then route to evaluator.
    """
    return "lead_researcher"


def route_after_lead_researcher(
    state: ResearchState,
):
    """
    Route after lead_researcher:
    - If strategies were generated (ToT mode), route to strategy_evaluator
    - If tasks were generated, use assign_subagents for parallel execution
    - If retry needed, route back to lead_researcher
    """
    # Check for retry condition
    if should_retry(state):
        print("  ↺ Routing back to lead_researcher for retry...")
        return "lead_researcher"

    # Check if strategies were generated (need evaluation)
    strategies = state.get("strategies", [])
    selected_strategy_index = state.get("selected_strategy_index")

    if len(strategies) >= 3 and selected_strategy_index is None:
        print("  🌳 Routing to strategy_evaluator to evaluate strategies...")
        return "strategy_evaluator"

    # Check if tasks were generated (normal flow) - use assign_subagents
    tasks = state.get("subagent_tasks", [])
    if len(tasks) > 0:
        return assign_subagents(state)

    # Default: retry lead_researcher
    return "lead_researcher"


# ============================================================================
# Graph Building
# ============================================================================

# Build graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("complexity_analyzer", complexity_analyzer_node)
workflow.add_node("strategy_evaluator", strategy_evaluator_node)
workflow.add_node("lead_researcher", lead_researcher_node)
workflow.add_node("subagent", subagent_node)
workflow.add_node("filter_findings", filter_findings_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("decision", decision_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("citation_agent", citation_agent_node)

# Add edges
workflow.add_edge(START, "complexity_analyzer")
workflow.add_edge("complexity_analyzer", "lead_researcher")
workflow.add_conditional_edges(
    "lead_researcher",
    route_after_lead_researcher,
    {
        "strategy_evaluator": "strategy_evaluator",
        "subagent": "subagent",
        "lead_researcher": "lead_researcher",
    },
)
workflow.add_edge("strategy_evaluator", "lead_researcher")
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
# Compile app with persistence (configurable via CHECKPOINTER_BACKEND env var)
# Supports: 'memory', 'sqlite' (default), 'postgres'
# This enables "Short-Term Memory" (thread-scoped) with optional persistence
checkpointer = get_checkpointer()
app = workflow.compile(checkpointer=checkpointer)
