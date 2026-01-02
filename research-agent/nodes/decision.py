"""Decision node: Determine if more research is needed using LLM"""

from config import settings
from graph.utils import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_lead_llm
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


def _rule_based_fallback(state: DecisionState) -> dict:
    """
    Fallback to rule-based decision logic if LLM fails.
    This preserves the original decision logic as a safety net.
    """
    iteration_count = state.get("iteration_count", 0)
    findings_count = len(state.get("subagent_findings", []))
    synthesized_length = len(state.get("synthesized_results", ""))
    extracted_citations = state.get("all_extracted_citations", [])
    has_citations = len(extracted_citations) > 0

    # Get complexity analysis
    complexity_analysis = state.get("complexity_analysis")
    max_iterations = settings.MAX_ITERATIONS
    complexity_level = "medium"

    if complexity_analysis:
        if hasattr(complexity_analysis, "max_iterations"):
            max_iterations = complexity_analysis.max_iterations
            complexity_level = complexity_analysis.complexity_level
        elif isinstance(complexity_analysis, dict):
            max_iterations = complexity_analysis.get(
                "max_iterations", settings.MAX_ITERATIONS
            )
            complexity_level = complexity_analysis.get("complexity_level", "medium")
        max_iterations = min(max_iterations, settings.MAX_ITERATIONS)

    # Adaptive thresholds
    if complexity_level == "simple":
        min_findings = 2
        min_synthesis_length = 300
    elif complexity_level == "complex":
        min_findings = 5
        min_synthesis_length = 800
    else:
        min_findings = 3
        min_synthesis_length = 500

    needs_more = (
        iteration_count < max_iterations and
        (
            findings_count < min_findings or
            synthesized_length < min_synthesis_length or
            (has_citations and iteration_count < max_iterations - 1)
        )
    )

    return {"needs_more_research": needs_more}


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

    print("\n🤔 [Decision] Evaluating if more research is needed...")
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

    # Build prompt
    messages = [
        SystemMessage(content=DECISION_SYSTEM),
        HumanMessage(
            content=DECISION_MAIN.format(
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
        ),
    ]

    # Invoke LLM with structured output
    try:
        structured_llm = get_lead_llm().with_structured_output(
            DecisionResult, include_raw=True
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

        # Success: Parse LLM decision
        decision_result = response["parsed"]
        needs_more = decision_result.needs_more_research

        print(f"  ✅ LLM Decision: {'Continue' if needs_more else 'Finish'} research")
        print(f"     Confidence: {decision_result.confidence:.2f}")
        print(f"     Reasoning: {decision_result.reasoning[:150]}...")
        if decision_result.key_factors:
            factors = ", ".join(decision_result.key_factors[:3])
            print(f"     Key Factors: {factors}")

        return {"needs_more_research": needs_more}

    except Exception as e:
        # If LLM call fails completely, use fallback
        print(f"  ⚠️  LLM call failed: {e}")
        print("  ⚠️  Using rule-based fallback")
        return _rule_based_fallback(state)

