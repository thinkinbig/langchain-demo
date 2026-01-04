"""Strategy evaluator node: Evaluate multiple research strategies and select optimal one"""

from config import settings
from graph.utils import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice, get_max_llm
from prompts import STRATEGY_EVALUATOR_MAIN, STRATEGY_EVALUATOR_SYSTEM
from schemas import ResearchState, ResearchStrategy, StrategyEvaluationResult


def _format_strategies_summary(strategies: list[ResearchStrategy]) -> str:
    """Format strategies for evaluation prompt"""
    summary_parts = []
    for i, strategy in enumerate(strategies):
        if isinstance(strategy, dict):
            desc = strategy.get("description", "")[:200]
            rationale = strategy.get("rationale", "")[:150]
            tasks = strategy.get("tasks", [])
            task_count = len(tasks) if isinstance(tasks, list) else 0
            task_descs = [
                t.get("description", "")[:60] if isinstance(t, dict) else str(t)[:60]
                for t in (tasks[:3] if isinstance(tasks, list) else [])
            ]
        else:
            # Pydantic model
            desc = strategy.description[:200] if hasattr(strategy, "description") else ""
            rationale = strategy.rationale[:150] if hasattr(strategy, "rationale") else ""
            tasks = strategy.tasks if hasattr(strategy, "tasks") else []
            task_count = len(tasks) if isinstance(tasks, list) else 0
            task_descs = [
                t.description[:60] if hasattr(t, "description") else str(t)[:60]
                for t in (tasks[:3] if isinstance(tasks, list) else [])
            ]

        summary_parts.append(
            f"Strategy {i}:\n"
            f"  Description: {desc}\n"
            f"  Rationale: {rationale}\n"
            f"  Tasks ({task_count}): {', '.join(task_descs) if task_descs else 'N/A'}"
        )

    return "\n\n".join(summary_parts)


def _rule_based_fallback(state: ResearchState) -> dict:
    """
    Fallback to rule-based selection if LLM fails.
    Selects strategy with most tasks (assuming more comprehensive).
    """
    strategies = state.get("strategies", [])
    if not strategies:
        return {"selected_strategy_index": 0}

    # Select strategy with most tasks
    max_tasks = 0
    selected_idx = 0
    for i, strategy in enumerate(strategies):
        if isinstance(strategy, dict):
            task_count = len(strategy.get("tasks", []))
        else:
            task_count = len(strategy.tasks) if hasattr(strategy, "tasks") else 0

        if task_count > max_tasks:
            max_tasks = task_count
            selected_idx = i

    return {
        "selected_strategy_index": selected_idx,
        "evaluation_reasoning": f"Rule-based: Selected strategy {selected_idx} with {max_tasks} tasks"
    }


def strategy_evaluator_node(state: ResearchState):
    """StrategyEvaluator: Evaluate multiple research strategies and select optimal one"""
    query = state.get("query", "")
    strategies = state.get("strategies", [])
    last_error = state.get("error")

    print("\n🎯 [StrategyEvaluator] Evaluating research strategies...")
    print(f"   Query: {query[:80]}...")
    print(f"   Strategies to evaluate: {len(strategies)}")

    if len(strategies) < 3:
        print(f"  ⚠️  Only {len(strategies)} strategies available, using rule-based fallback")
        return _rule_based_fallback(state)

    # Format strategies for prompt
    strategies_summary = _format_strategies_summary(strategies)

    # Build messages
    messages = [
        SystemMessage(content=STRATEGY_EVALUATOR_SYSTEM),
        HumanMessage(
            content=STRATEGY_EVALUATOR_MAIN.format(
                query=query,
                strategies_summary=strategies_summary
            )
        )
    ]

    # Add retry feedback if needed
    if last_error:
        retry_message = (
            f"\n\n<error>\nThe previous attempt failed validation:\n"
            f"{last_error}\n</error>\n"
            "<instructions>Please correct the JSON structure and try again."
            "</instructions>"
        )
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(content=messages[-1].content + retry_message)
        else:
            messages.append(HumanMessage(content=retry_message))

    # Select model based on complexity and environment
    # Use MAX model for complex tasks in production, otherwise use turbo
    complexity_analysis = state.get("complexity_analysis")
    complexity_level = "medium"
    use_max_model = False

    if complexity_analysis:
        if hasattr(complexity_analysis, "complexity_level"):
            complexity_level = complexity_analysis.complexity_level
        elif isinstance(complexity_analysis, dict):
            complexity_level = complexity_analysis.get("complexity_level", "medium")

        # Use MAX model only if:
        # 1. Task is complex
        # 2. MAX model is enabled (production environment)
        if complexity_level == "complex" and settings.ENABLE_MAX_MODEL:
            use_max_model = True
            print(
                "  🚀 Using MAX model for complex task evaluation "
                "(production mode)"
            )

    # Get appropriate LLM
    if use_max_model:
        llm = get_max_llm()
    else:
        # Use turbo for cost-effective evaluation
        llm = get_llm_by_model_choice("turbo")
        if complexity_level == "complex":
            print(
                "  💡 Complex task detected, but MAX model disabled "
                "(use AGENT_ENABLE_MAX_MODEL=true in production)"
            )

    structured_llm = llm.with_structured_output(
        StrategyEvaluationResult, include_raw=True
    )

    response = structured_llm.invoke(messages)

    # Process response with retry logic
    def fallback(s):
        print("  ⚠️  Using rule-based fallback due to LLM failure")
        return _rule_based_fallback(s)

    retry_state = process_structured_response(response, state, fallback)
    if retry_state:
        # Fallback was triggered
        return retry_state

    # Success: Parse evaluation result
    evaluation_result = response["parsed"]
    selected_idx = evaluation_result.selected_strategy_index

    print(f"  ✅ Selected Strategy {selected_idx}")
    print(f"     Coverage: {evaluation_result.coverage_score:.2f}")
    print(f"     Feasibility: {evaluation_result.feasibility_score:.2f}")
    print(f"     Efficiency: {evaluation_result.efficiency_score:.2f}")
    print(f"     Reasoning: {evaluation_result.evaluation_reasoning[:150]}...")

    return {
        "selected_strategy_index": selected_idx,
        "error": None,
        "retry_count": 0
    }

