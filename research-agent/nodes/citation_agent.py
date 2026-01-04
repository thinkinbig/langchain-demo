"""Citation agent node: Extract citations and create final report"""

from schemas import Citation, CitationAgentState


def citation_agent_node(state: CitationAgentState):
    """CitationAgent: Extract citations and create final report"""
    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")
    query = state["query"]

    # Get extracted citations from state (NEW!)
    extracted_citations = state.get("all_extracted_citations", [])

    print("\n📝 [CitationAgent] Extracting citations and creating final report...")
    print(f"  📊 Processing {len(findings)} findings for citations...")
    if extracted_citations:
        print(f"  📚 Including {len(extracted_citations)} extracted paper citations...")
        # Debug: show first few citations
        for i, cite in enumerate(extracted_citations[:3], 1):
            print(f"      {i}. {cite.get('title', 'Unknown')}")

    # Pre-extract all sources and build lookup map (performance optimization)
    all_source_dicts = []
    for finding in findings:
        sources = finding.sources
        all_source_dicts.extend(sources)

    # Build source metadata map (identifier → title) before loops
    source_map = {}
    for source in all_source_dicts:
        url = source.get("url", "")
        title = source.get("title", "Unknown")
        if url:
            source_map[url] = title

    # Add extracted citations to source map (NEW!)
    # These are papers mentioned in the content that we extracted
    for citation in extracted_citations:
        title = citation.get("title", "")
        context = citation.get("context", "")

        if title:
            # Create a pseudo-URL for the citation (for deduplication)
            citation_id = f"citation/{title}"
            # Format: "Title - Context"
            display_title = f"{title}"
            if context:
                display_title += f" - {context[:100]}"

            source_map[citation_id] = display_title

    # Collect unique citations using pre-built map
    all_sources = []
    seen_urls = set()
    for url, title in source_map.items():
        if url not in seen_urls:
            seen_urls.add(url)
            all_sources.append(Citation(title=title, url=url))
            print(f"      ✅ Added citation: {title[:50]} - {url[:50]}")

    print(f"  📚 Total unique citations collected: {len(all_sources)}")

    # Format citations
    citations = all_sources

    # Create final report - optimized format
    citations_text = "\n".join([
        f"{i+1}. {c.get('title', 'Unknown')} - {c.get('url', '')}"
        for i, c in enumerate(citations)
    ])

    final_report = f"""


{query}


{synthesized}


--------------------------------
{citations_text}
--------------------------------
"""

    print(f"  ✅ Final report created with {len(citations)} citations")

    return {
        "citations": citations,
        "final_report": final_report,
    }

