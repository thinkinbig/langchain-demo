"""Decision node: Determine if more research is needed"""

from schemas import DecisionState


def decision_node(state: DecisionState):
    """Decision: Determine if more research is needed"""
    # Only use metadata, not full content - reduces token usage
    iteration_count = state.get("iteration_count", 0)
    findings_count = len(state.get("subagent_findings", []))
    synthesized_length = len(state.get("synthesized_results", ""))

    # NEW: Check for extracted citations
    extracted_citations = state.get("all_extracted_citations", [])
    has_citations = len(extracted_citations) > 0

    print("\n🤔 [Decision] Evaluating if more research is needed...")
    print(
        f"   Iteration: {iteration_count}, "
        f"Findings: {findings_count}, "
        f"Synthesis length: {synthesized_length}, "
        f"Citations found: {len(extracted_citations)}"
    )

    # Citation-aware decision logic
    # Continue if: iteration < 3 AND (few findings OR short synthesis
    # OR has unexplored citations)
    needs_more = (
        iteration_count < 3 and  # Allow one extra iteration for citations
        (
            findings_count < 3 or
            synthesized_length < 500 or
            (has_citations and iteration_count == 1)  # Continue once if citations
        )
    )

    if needs_more:
        if has_citations and iteration_count == 1:
            print(
                f"  ✅ Decision: Continue research "
                f"(exploring {len(extracted_citations)} citations)"
            )
        else:
            print("  ✅ Decision: Continue research")
        return {"needs_more_research": True}
    else:
        print("  ✅ Decision: Finish research")
        return {"needs_more_research": False}

