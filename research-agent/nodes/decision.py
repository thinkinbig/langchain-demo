"""Decision node: Determine if more research is needed using LLM"""

from config import settings
from graph.utils import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from prompts import DECISION_MAIN, DECISION_SYSTEM
from schemas import DecisionResult, DecisionState


def _extract_findings_summary(findings, max_items: int = 5) -> str:
    """Extract summary of findings (metadata only, no full content)"""
    if not findings:
        return "No findings available."

    summary_parts = []
    for i, finding in enumerate(findings[:max_items]):
        if isinstance(finding, dict):
            task = finding.get("task", "Unknown")[:60]
            summary = finding.get("summary", "")[:100]
            sources_count = len(finding.get("sources", []))
            summary_parts.append(
                f"{i+1}. Task: {task}\n"
                f"   Summary: {summary}...\n"
                f"   Sources: {sources_count}"
            )
        else:
            # Pydantic model
            task = finding.task[:60] if hasattr(finding, "task") else "Unknown"
            summary = finding.summary[:100] if hasattr(finding, "summary") else ""
            sources_count = (
                len(finding.sources) if hasattr(finding, "sources") else 0
            )
            summary_parts.append(
                f"{i+1}. Task: {task}\n"
                f"   Summary: {summary}...\n"
                f"   Sources: {sources_count}"
            )

    if len(findings) > max_items:
        summary_parts.append(f"... and {len(findings) - max_items} more findings")

    return "\n".join(summary_parts)


def decision_node(state: DecisionState):
    """Decision: Determine if more research is needed using LLM analysis"""
    query = state.get("query", "")
    iteration_count = state.get("iteration_count", 0)
    findings = state.get("subagent_findings", [])
    findings_count = len(findings)
    synthesized_results = state.get("synthesized_results", "")
    synthesized_length = len(synthesized_results)
    extracted_citations = state.get("all_extracted_citations", [])
    citations_count = len(extracted_citations)

    # Get complexity analysis
    complexity_analysis = state.get("complexity_analysis")
    max_iterations = settings.MAX_ITERATIONS
    complexity_info = "No complexity analysis available."

    if complexity_analysis:
        if hasattr(complexity_analysis, "max_iterations"):
            max_iterations = complexity_analysis.max_iterations
            complexity_info = (
                f"Complexity: {complexity_analysis.complexity_level}\n"
                f"Max Iterations: {max_iterations}\n"
                f"Rationale: {complexity_analysis.rationale[:200]}"
            )
        elif isinstance(complexity_analysis, dict):
            max_iterations = complexity_analysis.get(
                "max_iterations", settings.MAX_ITERATIONS
            )
            level = complexity_analysis.get("complexity_level", "medium")
            rationale = complexity_analysis.get("rationale", "")[:200]
            complexity_info = (
                f"Complexity: {level}\n"
                f"Max Iterations: {max_iterations}\n"
                f"Rationale: {rationale}"
            )
        max_iterations = min(max_iterations, settings.MAX_ITERATIONS)

    # Check if this is a partial synthesis (early decision optimization)
    has_partial_synthesis = state.get("has_partial_synthesis", False)
    is_partial = has_partial_synthesis and (
        "# Resolution" not in synthesized_results or
        not synthesized_results.strip().endswith("# Resolution")
    )

    print("\n🤔 [Decision] Evaluating if more research is needed...")
    if is_partial:
        print("  ⚡ Early decision mode: evaluating based on partial synthesis (S+C only)")
    print(
        f"   Iteration: {iteration_count}/{max_iterations}, "
        f"Findings: {findings_count}, "
        f"Synthesis: {synthesized_length} chars, "
        f"Citations: {citations_count}"
    )

    # Extract optimized state information (metadata only)
    findings_summary = _extract_findings_summary(findings, max_items=5)
    synthesis_preview = synthesized_results[:500] if synthesized_results else "N/A"
    citations_info = (
        f"Found {citations_count} citations"
        if citations_count > 0
        else "No citations found"
    )

    # Add note about partial synthesis if applicable
    partial_note = ""
    if is_partial:
        partial_note = (
            "\n\n⚠️ IMPORTANT: This is a PARTIAL synthesis containing only "
            "Situation and Complication sections. The Resolution section has not "
            "been generated yet (early decision optimization). Please make your "
            "decision based on the available Situation and Complication content. "
            "If you decide to finish research, the Resolution section will be "
            "generated in the next step."
        )

    # Build prompt
    decision_prompt = DECISION_MAIN.format(
        query=query,
        complexity_info=complexity_info,
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        findings_count=findings_count,
        synthesis_length=synthesized_length,
        citations_count=citations_count,
        findings_summary=findings_summary,
        synthesis_preview=synthesis_preview,
        citations_info=citations_info,
    )

    # Append partial synthesis note if applicable
    if partial_note:
        decision_prompt += partial_note

    messages = [
        SystemMessage(content=DECISION_SYSTEM),
        HumanMessage(content=decision_prompt),
    ]

    # Get recommended model from complexity analysis
    complexity_analysis = state.get("complexity_analysis")
    recommended_model = "plus"  # Default
    if complexity_analysis:
        if hasattr(complexity_analysis, "recommended_model"):
            model = complexity_analysis.recommended_model
            if model in ["turbo", "plus", "max"]:
                # Downgrade max to plus if not enabled
                if model == "max" and not settings.ENABLE_MAX_MODEL:
                    recommended_model = "plus"
                else:
                    recommended_model = model
        elif isinstance(complexity_analysis, dict):
            model = complexity_analysis.get("recommended_model", "plus")
            if model in ["turbo", "plus", "max"]:
                # Downgrade max to plus if not enabled
                if model == "max" and not settings.ENABLE_MAX_MODEL:
                    recommended_model = "plus"
                else:
                    recommended_model = model

    # Invoke LLM with structured output
    llm = get_llm_by_model_choice(recommended_model)
    structured_llm = llm.with_structured_output(
        DecisionResult, include_raw=True
    )
    response = structured_llm.invoke(messages)

    # Process response with retry logic
    retry_state = process_structured_response(response, state)
    if retry_state:
        # Retry needed, return state update
        return retry_state

    # Success: Parse LLM decision
    decision_result = response["parsed"]
    needs_more = decision_result.needs_more_research

    print(f"  ✅ LLM Decision: {'Continue' if needs_more else 'Finish'} research")
    print(f"     Confidence: {decision_result.confidence:.2f}")
    print(f"     Reasoning: {decision_result.reasoning[:150]}...")
    if decision_result.key_factors:
        factors = ", ".join(decision_result.key_factors[:3])
        print(f"     Key Factors: {factors}")

    # Return full decision information for use in next iteration
    return_state = {
        "needs_more_research": needs_more,
        "decision_reasoning": decision_result.reasoning if needs_more else None,
        "decision_key_factors": decision_result.key_factors if needs_more else [],
    }

    # If we have partial synthesis and decision is "finish", we need to complete it
    # The synthesizer will detect this and complete the Resolution section
    if is_partial and not needs_more:
        # Decision is to finish, but we have partial synthesis
        # Keep the partial synthesis flag so synthesizer can complete it
        return_state["has_partial_synthesis"] = True
        print("  ⚡ Decision: Finish research, but partial synthesis detected - will complete Resolution")
    elif is_partial and needs_more:
        # Decision is to continue, keep partial synthesis for next iteration
        return_state["has_partial_synthesis"] = True
        print("  ⚡ Decision: Continue research, preserving partial synthesis")

    return return_state

