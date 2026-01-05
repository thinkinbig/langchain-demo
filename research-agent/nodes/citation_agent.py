"""Citation agent node: Extract citations and create final report"""

from schemas import Citation, CitationAgentState
from text_utils import clean_report_output


def citation_agent_node(state: CitationAgentState):
    """CitationAgent: Extract citations and create final report"""
    import json
    log_path = "/home/zeyuli/Code/langchain/.cursor/debug.log"

    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "citation_agent.py:8", "message": "Function entry", "data": {"findings_count": len(state.get("subagent_findings", [])), "extracted_citations_count": len(state.get("all_extracted_citations", []))}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")
    query = state["query"]

    # Get extracted citations from state (NEW!)
    # Convert to dicts if they are Citation objects (for consistency)
    extracted_citations_raw = state.get("all_extracted_citations", [])
    extracted_citations = []
    for cite in extracted_citations_raw:
        if isinstance(cite, dict):
            extracted_citations.append(cite)
        elif hasattr(cite, "model_dump"):
            extracted_citations.append(cite.model_dump())
        elif hasattr(cite, "dict"):
            extracted_citations.append(cite.dict())
        else:
            # Fallback: try to convert to dict
            extracted_citations.append(dict(cite))

    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,D", "location": "citation_agent.py:18", "message": "Extracted citations structure", "data": {"count": len(extracted_citations), "first_citation": extracted_citations[0] if extracted_citations else None, "type": str(type(extracted_citations[0]) if extracted_citations else None)}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

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

    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "citation_agent.py:32", "message": "Source dicts before mapping", "data": {"count": len(all_source_dicts), "samples": all_source_dicts[:3]}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

    # Build source metadata map (identifier → title) before loops
    source_map = {}
    for source in all_source_dicts:
        url = source.get("url", "")
        title = source.get("title", "Unknown")
        if url:
            source_map[url] = title

    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "citation_agent.py:40", "message": "Source map after regular sources", "data": {"count": len(source_map), "items": list(source_map.items())[:5]}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

    # Add extracted citations to source map (NEW!)
    # These are papers mentioned in the content that we extracted
    # Store them separately to handle formatting differently
    extracted_citation_objects = []
    for citation in extracted_citations:
        title = citation.get("title", "")
        url = citation.get("url", "")  # Use actual URL if available
        context = citation.get("context", "")

        # #region agent log
        with open(log_path, "a") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B", "location": "citation_agent.py:50", "message": "Processing extracted citation", "data": {"title": title, "url": url, "context": context[:50] if context else "", "citation_type": str(type(citation))}, "timestamp": __import__("time").time() * 1000}) + "\n")
        # #endregion

        if title:
            # For extracted citations, use title only (no context in title)
            # Use empty URL if no actual URL is provided (don't create fake URLs)
            citation_obj = Citation(
                title=title,
                url=url,  # Empty string if not provided
                context=context,
                relevance=citation.get("relevance", ""),
                year=citation.get("year"),
                authors=citation.get("authors", [])
            )
            extracted_citation_objects.append(citation_obj)

            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B", "location": "citation_agent.py:60", "message": "Created extracted citation object", "data": {"title": title, "url": url}, "timestamp": __import__("time").time() * 1000}) + "\n")
            # #endregion

    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "citation_agent.py:65", "message": "Source map after extracted citations", "data": {"count": len(source_map), "items": list(source_map.items())}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

    def get_dedup_key(citation: Citation) -> tuple:
        """
        Generate a deduplication key for a citation.

        Strategy:
        1. If citation has a URL (internal documents, web sources): use URL as key
        2. If citation has no URL but has authors: use (title, sorted_authors) as key
        3. Otherwise: use title as fallback
        """
        if citation.url:
            # For sources with URLs (internal docs, web sources), URL is unique identifier
            return ("url", citation.url)
        elif citation.authors:
            # For academic papers without URL, use title + authors
            sorted_authors = tuple(sorted(citation.authors))
            return ("title_authors", citation.title.lower().strip(), sorted_authors)
        else:
            # Fallback: use title only (for citations without URL or authors)
            return ("title", citation.title.lower().strip())

    # Collect unique citations from regular sources (findings)
    all_sources = []
    seen_keys = set()
    # Also track titles to avoid duplicates when extracted citations have no URL
    seen_titles = set()

    for url, title in source_map.items():
        citation_obj = Citation(title=title, url=url)
        dedup_key = get_dedup_key(citation_obj)
        title_normalized = title.lower().strip()

        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            seen_titles.add(title_normalized)
            all_sources.append(citation_obj)

            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B,C", "location": "citation_agent.py:75", "message": "Created Citation object from source", "data": {"url": url, "title": title, "dedup_key": str(dedup_key)}, "timestamp": __import__("time").time() * 1000}) + "\n")
            # #endregion

            print(f"      ✅ Added citation: {title[:50]} - {url[:50]}")

    # Add extracted citations (papers) - deduplicate using unified strategy
    for citation_obj in extracted_citation_objects:
        dedup_key = get_dedup_key(citation_obj)
        title_normalized = citation_obj.title.lower().strip()

        # Check both dedup_key and title to avoid duplicates
        # This handles cases where extracted citation has no URL but matches
        # a regular source by title
        if dedup_key not in seen_keys:
            # Also check if title matches an existing citation
            # (to avoid duplicates when extracted citation has no URL)
            if not citation_obj.url and title_normalized in seen_titles:
                # Skip: this extracted citation matches a regular source by title
                print(f"      ⏭️  Skipped duplicate: {citation_obj.title[:50]} (matches existing source)")
                continue

            seen_keys.add(dedup_key)
            seen_titles.add(title_normalized)
            all_sources.append(citation_obj)

            # #region agent log
            with open(log_path, "a") as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A,B", "location": "citation_agent.py:120", "message": "Added extracted citation", "data": {"title": citation_obj.title, "authors": citation_obj.authors, "url": citation_obj.url, "dedup_key": str(dedup_key)}, "timestamp": __import__("time").time() * 1000}) + "\n")
            # #endregion

            authors_str = ", ".join(citation_obj.authors[:2]) if citation_obj.authors else ""
            print(f"      ✅ Added extracted citation: {citation_obj.title[:50]}" + (f" ({authors_str})" if authors_str else ""))

    print(f"  📚 Total unique citations collected: {len(all_sources)}")

    # Format citations
    citations = all_sources

    # #region agent log
    with open(log_path, "a") as f:
        formatted_samples = []
        for i, c in enumerate(citations[:3]):
            try:
                title_val = c.get('title', 'Unknown') if hasattr(c, 'get') else getattr(c, 'title', 'Unknown')
                url_val = c.get('url', '') if hasattr(c, 'get') else getattr(c, 'url', '')
                formatted_samples.append({"index": i, "title": title_val, "url": url_val, "formatted": f"{i+1}. {title_val} - {url_val}"})
            except Exception as e:
                formatted_samples.append({"index": i, "error": str(e)})
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "C", "location": "citation_agent.py:90", "message": "Before formatting citations", "data": {"count": len(citations), "samples": formatted_samples}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

    # Create final report - format citations with proper formatting
    def format_citation_for_output(citation: Citation, index: int) -> str:
        """Format a single citation for the final report"""
        title = citation.get('title', 'Unknown') if hasattr(citation, 'get') else citation.title
        url = citation.get('url', '') if hasattr(citation, 'get') else citation.url
        authors = citation.get('authors', []) if hasattr(citation, 'get') else citation.authors
        year = citation.get('year') if hasattr(citation, 'get') else citation.year

        # Build citation string
        parts = [f"{index+1}. {title}"]

        # Add authors if available (for academic papers)
        if authors:
            authors_str = ", ".join(authors[:3])  # Show first 3 authors
            if len(authors) > 3:
                authors_str += " et al."
            parts.append(f"({authors_str})")

        # Add year if available
        if year:
            parts.append(f"[{year}]")

        # Add URL if available (for internal docs and web sources)
        if url:
            parts.append(f"- {url}")

        return " ".join(parts)

    citations_text = "\n".join([
        format_citation_for_output(c, i) for i, c in enumerate(citations)
    ])

    # #region agent log
    with open(log_path, "a") as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "ALL", "location": "citation_agent.py:95", "message": "Final citations text", "data": {"citations_text": citations_text[:500]}, "timestamp": __import__("time").time() * 1000}) + "\n")
    # #endregion

    # Clean synthesized output to remove any XML tags that might have been included
    cleaned_synthesized = clean_report_output(synthesized)

    final_report = f"""# Research Report


{query}


{cleaned_synthesized}


--------------------------------
{citations_text}
--------------------------------
"""

    print(f"  ✅ Final report created with {len(citations)} citations")

    # Convert Citation objects to dicts for JSON serialization (checkpointer)
    citations_dicts = [
        c.model_dump() if hasattr(c, "model_dump") else dict(c)
        for c in citations
    ]

    return {
        "citations": citations_dicts,
        "final_report": final_report,
    }

