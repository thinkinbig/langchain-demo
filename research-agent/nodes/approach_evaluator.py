"""Approach evaluator node: Evaluate research approaches for a query"""

import context_manager
from graph.utils import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from memory.temporal_memory import TemporalMemory
from prompts import (
    LEAD_RESEARCHER_APPROACH_MAIN,
    LEAD_RESEARCHER_APPROACH_SYSTEM,
)
from schemas import ApproachEvaluation, ResearchState


def _retrieve_planning_history(query: str, k: int = 5) -> str:
    """
    Retrieve relevant planning history from long-term memory.

    Args:
        query: Current research query
        k: Number of memories to retrieve

    Returns:
        Formatted string with planning history, or empty string if none found
    """
    try:
        memory = TemporalMemory()
        memories = memory.retrieve_valid_memories(
            query=query,
            k=k,
            min_relevance=0.6,
            filter_by_tags=["lead_researcher", "planning"],
            include_invalid=False
        )

        if not memories:
            return ""

        # Format memories for prompt
        history_parts = []
        for doc, relevance in memories[:k]:
            content = doc.page_content[:300]  # Limit each memory
            metadata = doc.metadata
            stored_at = metadata.get("stored_at", "unknown")
            history_parts.append(
                f"[{stored_at[:10]}] (relevance: {relevance:.2f})\n{content}"
            )

        return "\n\n".join(history_parts)
    except Exception as e:
        # Graceful fallback if memory retrieval fails
        print(f"  ⚠️  Memory retrieval failed: {e}")
        return ""


def approach_evaluator_node(state: ResearchState):
    """
    Approach Evaluator: Evaluate multiple research approaches for the query.
    
    This node generates 3 different research approaches and evaluates them,
    but does NOT select one. The selection is done by human_approach_selector
    after human input.
    
    Args:
        state: ResearchState containing query, complexity_analysis, etc.
        
    Returns:
        State update with approach_evaluation (but no selected_approach_index)
    """
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    last_error = state.get("error")
    rag_cache = state.get("rag_cache", {})

    # Only evaluate approaches on first iteration
    if iteration_count > 0:
        # Skip evaluation if already done
        existing_eval = state.get("approach_evaluation")
        if existing_eval:
            msg = (
                "  ℹ️  [ApproachEvaluator] Skipping - "
                "already evaluated in first iteration"
            )
            print(msg)
            return {}

        # If no evaluation exists and it's not first iteration, skip
        msg = (
            "  ℹ️  [ApproachEvaluator] Skipping - "
            "only evaluates on first iteration"
        )
        print(msg)
        return {}

    print(
        "  📋 [ApproachEvaluator] Evaluating research approaches..."
    )

    # RAG Caching: Check cache first
    if query in rag_cache:
        print("  🧠 [RAG] Using cached context for query...")
        retrieved_context = rag_cache[query]
    else:
        print(f"  🧠 [RAG] Retrieving context for: {query[:50]}...")
        retrieved_context, _ = context_manager.retrieve_knowledge(query, k=2)
        rag_cache[query] = retrieved_context

    # Retrieve planning history from long-term memory
    memory_context = ""
    print("  🧠 [Memory] Retrieving planning history...")
    memory_context = _retrieve_planning_history(query, k=5)
    if memory_context:
        line_count = len(memory_context.split("\n"))
        print(f"  ✅ Retrieved {line_count} lines of planning history")
    else:
        print("  ℹ️  No relevant planning history found")

    # Get complexity analysis for approach evaluation guidance
    complexity_analysis = state.get("complexity_analysis")
    complexity_info = ""
    if complexity_analysis:
        if hasattr(complexity_analysis, "complexity_level"):
            # Pydantic model
            recommended = complexity_analysis.recommended_workers
            complexity_info = (
                f"Complexity Level: {complexity_analysis.complexity_level}\n"
                f"Recommended Workers: {recommended}\n"
                f"Max Iterations: {complexity_analysis.max_iterations}\n"
                f"Rationale: {complexity_analysis.rationale}\n\n"
                f"Use this complexity assessment to guide approach evaluation. "
                f"Consider approaches that can leverage approximately {recommended} "
                f"parallel research tasks."
            )
        elif isinstance(complexity_analysis, dict):
            # Dict format
            level = complexity_analysis.get('complexity_level', 'unknown')
            workers = complexity_analysis.get('recommended_workers', 2)
            iterations = complexity_analysis.get('max_iterations', 2)
            rationale = complexity_analysis.get('rationale', '')
            complexity_info = (
                f"Complexity Level: {level}\n"
                f"Recommended Workers: {workers}\n"
                f"Max Iterations: {iterations}\n"
                f"Rationale: {rationale}\n\n"
                f"Use this complexity assessment to guide approach evaluation. "
                f"Consider approaches that can leverage approximately {workers} "
                f"parallel research tasks."
            )
        else:
            complexity_info = (
                "No complexity analysis available. "
                "Use your judgment for approach evaluation."
            )

    # Build messages for approach evaluation
    system_prompt = context_manager.get_system_context(
        LEAD_RESEARCHER_APPROACH_SYSTEM, include_knowledge=False
    )
    approach_messages = [SystemMessage(content=system_prompt)]

    memory_ctx = (
        memory_context if memory_context
        else "No previous planning history."
    )
    scratchpad = state.get("scratchpad", "")
    max_scratchpad_length = 200
    if len(scratchpad) > max_scratchpad_length:
        scratchpad = scratchpad[:max_scratchpad_length] + "..."

    approach_prompt = LEAD_RESEARCHER_APPROACH_MAIN.format(
        query=query,
        complexity_info=complexity_info,
        memory_context=memory_ctx,
        internal_knowledge=retrieved_context
    )
    approach_messages.append(HumanMessage(content=approach_prompt))

    # Add feedback if retrying
    if last_error:
        from prompts import LEAD_RESEARCHER_RETRY
        retry_prompt = LEAD_RESEARCHER_RETRY.format(
            previous_prompt=approach_messages[-1].content if approach_messages else "",
            error=last_error
        )
        if approach_messages and isinstance(approach_messages[-1], HumanMessage):
            approach_messages[-1] = HumanMessage(content=retry_prompt)
        else:
            approach_messages.append(HumanMessage(content=retry_prompt))

    # Get recommended model from complexity analysis
    from config import settings
    recommended_model = "plus"  # Default
    if complexity_analysis:
        if hasattr(complexity_analysis, "recommended_model"):
            model = complexity_analysis.recommended_model
            if model in ["turbo", "plus", "max"]:
                if model == "max" and not settings.ENABLE_MAX_MODEL:
                    recommended_model = "plus"
                else:
                    recommended_model = model
        elif isinstance(complexity_analysis, dict):
            model = complexity_analysis.get("recommended_model", "plus")
            if model in ["turbo", "plus", "max"]:
                if model == "max" and not settings.ENABLE_MAX_MODEL:
                    recommended_model = "plus"
                else:
                    recommended_model = model

    # Invoke LLM for approach evaluation
    llm = get_llm_by_model_choice(recommended_model)
    approach_llm = llm.with_structured_output(
        ApproachEvaluation, include_raw=True
    )

    approach_response = approach_llm.invoke(approach_messages)

    # Process approach evaluation response
    approach_retry_state = process_structured_response(
        approach_response, state
    )
    if approach_retry_state:
        # Retry needed, return state update
        return approach_retry_state

    approach_eval = approach_response["parsed"]

    # Display approach evaluation results (but note that selection will be done by human)
    print("\n  📊 [Approach Evaluation Results]")
    for i, approach in enumerate(approach_eval.approaches):
        print(f"\n     Approach {i + 1}:")
        print(f"        Description: {approach.description[:100]}...")
        if approach.advantages:
            adv_str = ', '.join(approach.advantages[:3])
            print(f"        Advantages: {adv_str}")
        if approach.disadvantages:
            dis_str = ', '.join(approach.disadvantages[:3])
            print(f"        Disadvantages: {dis_str}")
        if approach.suitability:
            print(f"        Suitability: {approach.suitability[:80]}...")

    print("\n  ⏸️  Waiting for human selection...")

    # Convert Pydantic model to dict for LangGraph serialization
    approach_eval_dict = (
        approach_eval.model_dump()
        if hasattr(approach_eval, "model_dump")
        else approach_eval
    )

    # Return state update with approach_evaluation
    # Note: We include the LLM's selected_approach_index in the
    # evaluation, but it will be overridden by human_approach_selector
    result = {
        "approach_evaluation": approach_eval_dict,
        "rag_cache": rag_cache,
        "error": None,
        "retry_count": 0,
    }
    return result

