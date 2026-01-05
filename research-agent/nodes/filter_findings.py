"""Filter findings node: Sanitize and filter findings before synthesis"""

from schemas import ResearchState


def filter_findings_node(state: ResearchState):
    """
    Guardrail Node: Sanitize and filter findings before synthesis.
    - Removes empty/failed findings.
    - Removes findings with no relevant content.
    - Deduplicates sources.
    """
    findings = state.get("subagent_findings", [])
    print(f"\n🛡️  [Guardrail] Filtering {len(findings)} findings...")

    valid_findings = []
    seen_content_hashes = set()

    for f in findings:
        # 1. Check for failure patterns in summary
        summary_lower = f.summary.lower()
        fail_patterns = [
            "no information found",
            "no search results",
            "unable to find",
            "failed to read",
            "analysis failed"
        ]
        if any(p in summary_lower for p in fail_patterns):
            print(f"     Dropped noise: {f.task[:40]}...")
            continue

        # 2. Check content length (heuristic relevance)
        if len(f.content) < 50 and len(f.summary) < 20:
            print(f"     Dropped empty/short: {f.task[:40]}...")
            continue

        # 3. Deduplication (simple hash of summary + task)
        # Prevents identical findings from separate subagent runs
        h = hash(f.summary + f.task)
        if h in seen_content_hashes:
            print(f"     Dropped duplicate: {f.task[:40]}...")
            continue

        seen_content_hashes.add(h)
        valid_findings.append(f)

    print(f"  ✅ Kept {len(valid_findings)}/{len(findings)} relevant findings.")

    # Aggregate all extracted citations
    from citation_parser import deduplicate_citations

    all_citations = []
    for f in valid_findings:
        # Handle both Pydantic model and dict
        if hasattr(f, "extracted_citations"):
            all_citations.extend(f.extracted_citations)
        elif isinstance(f, dict):
            all_citations.extend(f.get("extracted_citations", []))

    unique_citations = deduplicate_citations(all_citations)

    if unique_citations:
        print(
            f"  📚 [Citations] Aggregated {len(unique_citations)} "
            f"unique citations from findings"
        )

    # Convert citations with URLs to visited_sources
    # This prevents duplicate searches for papers that are already cited
    from nodes.subagent.utils import citations_to_visited_sources

    citation_sources = citations_to_visited_sources(unique_citations)
    if citation_sources:
        print(
            f"  🔗 [Sources] Converted {len(citation_sources)} citations "
            f"with URLs to visited_sources (type: citation)"
        )

    # Return REPLACEMENT list (Note: requires reducer in schema to handle
    # strict replacement if needed, but since this is a sequential node,
    # it overwrites if we change schema or just clean up here.
    # Actually, standard LangGraph behavior with Annotated[list, add] is APPEND.
    # To FILTER, we usually need to overwrite.
    # Workaround: We pass 'valid_findings' to the next step via a distinct key OR
    # we modify the schema to allow overwrite (standard set).
    # Current schema uses `operator.add`. This is tricky.
    #
    # SOLUTION: We will return a NEW key used by Synthesizer: `filtered_findings`.
    # AND update Synthesizer to look for `filtered_findings` first.
    return {
        "filtered_findings": valid_findings,
        "all_extracted_citations": unique_citations,
        "visited_sources": citation_sources,  # Add citation sources to visited_sources
    }

