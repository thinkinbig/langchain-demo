"""Citation agent node: Extract citations and create final report"""

from typing import Any, Dict, List

import context_manager
from llm.factory import get_llm_by_model_choice
from schemas import Citation, CitationAgentState
from text_utils import clean_report_output


def detect_bibliography_section(content: str) -> str:
    """
    Detect and extract bibliography/reference section from content.

    Args:
        content: Full content text to search

    Returns:
        Extracted bibliography section text, or empty string if not found
    """
    if not content:
        return ""

    content_lower = content.lower()

    # Look for bibliography/reference section markers
    markers = [
        "references",
        "bibliography",
        "works cited",
        "citations",
        "reference list"
    ]

    # Find the start of bibliography section
    start_idx = -1
    for marker in markers:
        idx = content_lower.find(marker)
        if idx != -1:
            # Check if it's a section header (usually followed by newline or colon)
            if idx == 0 or content[idx-1] in ['\n', '#', ' ']:
                start_idx = idx
                break

    if start_idx == -1:
        return ""

    # Extract from start marker to end of content (or reasonable limit)
    # Look for next major section (usually starts with # or newline followed by capital)
    bibliography_text = content[start_idx:]

    # Limit to reasonable size (bibliography sections are usually not huge)
    # Take first 5000 chars to avoid token bloat
    bibliography_text = bibliography_text[:5000]

    return bibliography_text


def match_citations_to_bibliography(
    citations: List[Dict[str, Any]],
    bibliography: List[Any]
) -> Dict[str, Any]:
    """
    Match extracted citations to bibliography entries.

    Args:
        citations: List of extracted citation dictionaries
        bibliography: List of CitationGraphElement objects from bibliography parser

    Returns:
        Dictionary with matching results:
        - matched: List of citations that match bibliography entries
        - unmatched: List of citations not found in bibliography
        - bibliography_only: List of bibliography entries not referenced in citations
    """
    matched = []
    unmatched = []
    bibliography_only = list(bibliography)  # Start with all, remove as matched

    for citation in citations:
        citation_title = citation.get("title", "").lower().strip()
        citation_authors = citation.get("authors", [])
        citation_year = citation.get("year")

        # Try to find match in bibliography
        found_match = False
        for bib_entry in bibliography:
            bib_title = getattr(bib_entry, "title", "").lower().strip()
            bib_authors = getattr(bib_entry, "authors", [])
            bib_year = getattr(bib_entry, "year", None)

            # Match by title (fuzzy - check if titles are similar)
            title_match = (
                citation_title and bib_title and
                (citation_title in bib_title or bib_title in citation_title or
                 citation_title[:30] == bib_title[:30])  # First 30 chars match
            )

            # Match by year if both present
            year_match = (
                citation_year and bib_year and
                int(citation_year) == int(bib_year)
            )

            # Match by authors if both present
            author_match = False
            if citation_authors and bib_authors:
                # Check if any author names overlap
                citation_author_names = [a.lower() for a in citation_authors]
                bib_author_names = [a.lower() for a in bib_authors]
                author_match = any(
                    ca in ba or ba in ca
                    for ca in citation_author_names
                    for ba in bib_author_names
                )

            # Consider it a match if title matches,
            # or (year matches and author matches)
            if title_match or (year_match and author_match):
                matched.append(citation)
                # Remove from bibliography_only if present
                if bib_entry in bibliography_only:
                    bibliography_only.remove(bib_entry)
                found_match = True
                break

        if not found_match:
            unmatched.append(citation)

    return {
        "matched": matched,
        "unmatched": unmatched,
        "bibliography_only": bibliography_only
    }


def citation_agent_node(state: CitationAgentState):
    """CitationAgent: Extract citations and create final report"""
    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")
    query = state["query"]

    # ========================================================================
    # Bibliography Detection and Parsing (from Knowledge Graph first, then RAG)
    # ========================================================================
    print("  📚 [CitationAgent] Retrieving bibliography from Knowledge Graph...")
    bibliography_text = ""
    bibliography_entries = []
    citation_match_results = None
    bibliography_raw = ""

    # Strategy 1: Try Knowledge Graph first (preferred method)
    # The knowledge graph should have Paper nodes with "cites" relationships
    try:
        from config import settings
        if settings.GRAPH_ENABLED:
            from memory.bibliography_parser import CitationGraphElement
            from memory.graph_rag import GraphRAGManager

            graph_rag_manager = GraphRAGManager()

            # Method 1: Search for Paper nodes related to the query
            print("  🕸️  [CitationAgent] Searching Knowledge Graph for Paper nodes...")

            # Extract source document name from query or findings
            # Try to find the source paper node first
            source_doc_name = None
            for f in findings:
                sources = getattr(f, 'sources', [])
                for source in sources:
                    if isinstance(source, dict):
                        source_id = source.get('url', '') or source.get('title', '')
                        if source_id and ('.pdf' in source_id or 'internal' in str(source_id).lower()):
                            # Extract filename
                            if '/' in source_id:
                                source_doc_name = source_id.split('/')[-1]
                            else:
                                source_doc_name = source_id
                            break
                if source_doc_name:
                    break

            # If we found a source document, look for its Paper node
            if source_doc_name:
                source_paper_id = f"Paper: {source_doc_name}"
                paper_node = graph_rag_manager.graph_store.get_node(source_paper_id)

                if paper_node:
                    print(f"  ✅ Found source paper node: {source_paper_id}")
                    # Get all papers cited by this paper (via "cites" edges)
                    cited_papers = graph_rag_manager.graph_store.get_neighborhood(
                        source_paper_id, k=1
                    )

                    # Filter for "cites" relations
                    bibliography_nodes = []
                    for _src, rel, tgt in cited_papers:
                        if rel == "cites":
                            target_node = graph_rag_manager.graph_store.get_node(tgt)
                            if target_node and target_node.get("type") == "Paper":
                                bibliography_nodes.append(target_node)

                    if bibliography_nodes:
                        print(f"  ✅ Found {len(bibliography_nodes)} cited papers in Knowledge Graph")
                        # Convert graph nodes to CitationGraphElement format
                        bibliography_entries = []
                        for node in bibliography_nodes:
                            node_id = node.get("id", "")
                            desc = node.get("description", "")

                            # Parse description to extract authors, year, etc.
                            # Format: "Paper by Author1, Author2 (Year)"
                            try:
                                # Try to extract info from description
                                # This is a simple parser - could be improved
                                title = node_id.replace("Paper: ", "") if node_id.startswith("Paper:") else node_id
                                authors = []
                                year = None

                                if "by" in desc:
                                    parts = desc.split("by")
                                    if len(parts) > 1:
                                        author_part = parts[1].split("(")[0].strip()
                                        authors = [a.strip() for a in author_part.split(",")]

                                if "(" in desc and ")" in desc:
                                    year_str = desc.split("(")[1].split(")")[0].strip()
                                    try:
                                        year = int(year_str)
                                    except ValueError:
                                        pass

                                entry = CitationGraphElement(
                                    title=title,
                                    authors=authors,
                                    year=year,
                                    venue=""
                                )
                                bibliography_entries.append(entry)
                            except Exception:
                                print(f"  ⚠️  Failed to parse graph node {node_id}")
                                continue

                # If we didn't find via source paper, try PPR retrieval
                if not bibliography_entries:
                    print("  🕸️  [CitationAgent] Trying PPR-based retrieval from Knowledge Graph...")
                    graph_context = graph_rag_manager.retrieve_with_ppr(
                        query,
                        top_k_nodes=30,
                        top_k_docs=10,
                        alpha=0.85
                    )

                    # Search for Paper nodes in the graph context
                    if graph_context and "Paper:" in graph_context:
                        # Extract Paper node IDs from context
                        import re
                        paper_ids = re.findall(r'Paper: [^\n\(\)]+', graph_context)
                        for paper_id in paper_ids[:20]:  # Limit to 20
                            node = graph_rag_manager.graph_store.get_node(paper_id)
                            if node and node.get("type") == "Paper":
                                # Skip the source paper itself
                                if paper_id != source_paper_id:
                                    desc = node.get("description", "")
                                    title = paper_id.replace("Paper: ", "")
                                    try:
                                        authors = []
                                        year = None
                                        if "by" in desc:
                                            parts = desc.split("by")
                                            if len(parts) > 1:
                                                author_part = parts[1].split("(")[0].strip()
                                                authors = [a.strip() for a in author_part.split(",")]
                                        if "(" in desc and ")" in desc:
                                            year_str = desc.split("(")[1].split(")")[0].strip()
                                            try:
                                                year = int(year_str)
                                            except ValueError:
                                                pass

                                        entry = CitationGraphElement(
                                            title=title,
                                            authors=authors,
                                            year=year,
                                            venue=""
                                        )
                                        bibliography_entries.append(entry)
                                    except Exception:
                                        continue

                        if bibliography_entries:
                            print(f"  ✅ Found {len(bibliography_entries)} papers via PPR retrieval")

            # Format bibliography entries if found
            if bibliography_entries:
                bib_lines = []
                for i, entry in enumerate(bibliography_entries, 1):
                    title = getattr(entry, "title", "Unknown")
                    authors = getattr(entry, "authors", [])
                    year = getattr(entry, "year", "")
                    venue = getattr(entry, "venue", "")

                    bib_line = f"{i}. {title}"
                    if authors:
                        bib_line += f" by {', '.join(authors[:3])}"
                        if len(authors) > 3:
                            bib_line += " et al."
                    if year:
                        bib_line += f" ({year})"
                    if venue:
                        bib_line += f" - {venue}"
                    bib_lines.append(bib_line)

                bibliography_text = "\n".join(bib_lines)
                print(f"  ✅ Retrieved {len(bibliography_entries)} bibliography entries from Knowledge Graph")

    except Exception as e:
        print(f"  ⚠️  Knowledge Graph retrieval failed: {e}")
        print("  🔄 Falling back to RAG retrieval...")

    # Strategy 2: Fallback to RAG retrieval if Knowledge Graph didn't find anything
    if not bibliography_entries:
        print("  📚 [CitationAgent] Retrieving bibliography from RAG (fallback)...")

        # Try strategy A: query + "references bibliography"
        bibliography_query_a = f"{query} references bibliography"
        rag_content_a, _ = context_manager.retrieve_knowledge(bibliography_query_a, k=6)
        if rag_content_a:
            bibliography_raw = detect_bibliography_section(rag_content_a)

        # Try strategy B: if not found, use original query with larger k
        if not bibliography_raw:
            print("  🔍 [CitationAgent] Bibliography not found with specialized query, trying with larger k...")
            rag_content_b, _ = context_manager.retrieve_knowledge(query, k=10)
            if rag_content_b:
                bibliography_raw = detect_bibliography_section(rag_content_b)

        # Also check findings content as supplement (but not primary source)
        if not bibliography_raw:
            print("  🔍 [CitationAgent] Checking findings content for bibliography...")
            all_findings_content = []
            for f in findings:
                if hasattr(f, 'content') and f.content:
                    all_findings_content.append(f.content)
            if all_findings_content:
                combined_findings_content = "\n\n".join(all_findings_content)
                bibliography_raw = detect_bibliography_section(combined_findings_content)

        # Parse RAG-retrieved bibliography if found
        if bibliography_raw:
            print("  ✅ Bibliography section detected from RAG, parsing...")
            try:
                from memory.bibliography_parser import parse_bibliography

                # Use turbo model for bibliography parsing (lighter than plus)
                bib_llm = get_llm_by_model_choice("turbo")
                rag_bibliography_entries = parse_bibliography(bibliography_raw, bib_llm)

                if rag_bibliography_entries:
                    print(f"  ✅ Parsed {len(rag_bibliography_entries)} bibliography entries from RAG")
                    # Merge with graph entries (avoid duplicates)
                    existing_titles = {e.title.lower() for e in bibliography_entries}
                    for entry in rag_bibliography_entries:
                        if entry.title.lower() not in existing_titles:
                            bibliography_entries.append(entry)
                            existing_titles.add(entry.title.lower())

                    # Reformat bibliography text
                    bib_lines = []
                    for i, entry in enumerate(bibliography_entries, 1):
                        title = getattr(entry, "title", "Unknown")
                        authors = getattr(entry, "authors", [])
                        year = getattr(entry, "year", "")
                        venue = getattr(entry, "venue", "")

                        bib_line = f"{i}. {title}"
                        if authors:
                            bib_line += f" by {', '.join(authors[:3])}"
                            if len(authors) > 3:
                                bib_line += " et al."
                        if year:
                            bib_line += f" ({year})"
                        if venue:
                            bib_line += f" - {venue}"
                        bib_lines.append(bib_line)

                    bibliography_text = "\n".join(bib_lines)
                else:
                    # Use raw text if parsing failed
                    if not bibliography_text:
                        bibliography_text = bibliography_raw[:1000]
            except Exception as e:
                print(f"  ⚠️  Bibliography parsing failed: {e}, using raw text")
                if not bibliography_text:
                    bibliography_text = bibliography_raw[:1000]
        else:
            if not bibliography_entries:
                print("  ⚠️  No bibliography/reference section found in Knowledge Graph or RAG")

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
    # Store them separately to handle formatting differently
    extracted_citation_objects = []
    for citation in extracted_citations:
        title = citation.get("title", "")
        url = citation.get("url", "")  # Use actual URL if available
        context = citation.get("context", "")

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

            authors_str = ", ".join(citation_obj.authors[:2]) if citation_obj.authors else ""
            print(f"      ✅ Added extracted citation: {citation_obj.title[:50]}" + (f" ({authors_str})" if authors_str else ""))

    print(f"  📚 Total unique citations collected: {len(all_sources)}")

    # Match extracted citations to bibliography (if bibliography was found)
    if bibliography_entries and extracted_citations:
        print("  🔍 [CitationAgent] Matching citations to bibliography...")
        citation_match_results = match_citations_to_bibliography(
            extracted_citations, bibliography_entries
        )

        matched_count = len(citation_match_results["matched"])
        unmatched_count = len(citation_match_results["unmatched"])
        print(f"  📊 Matched: {matched_count}, Unmatched: {unmatched_count}")

        if unmatched_count > 0:
            print(f"  ⚠️  {unmatched_count} citations not found in bibliography")

    # Format citations
    citations = all_sources

    # Format citations for final report (hardcoded format)
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

    # Clean synthesized output to remove any XML tags that might have been included
    cleaned_synthesized = clean_report_output(synthesized)

    # Format final report (hardcoded format)
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

    # Store bibliography information for Verifier to use
    result = {
        "citations": citations_dicts,
        "final_report": final_report,
    }

    # Store bibliography processing results for Verifier
    if bibliography_entries:
        # Convert bibliography entries to dicts for serialization
        bibliography_entries_dicts = []
        for entry in bibliography_entries:
            if hasattr(entry, "model_dump"):
                bibliography_entries_dicts.append(entry.model_dump())
            elif hasattr(entry, "dict"):
                bibliography_entries_dicts.append(entry.dict())
            else:
                bibliography_entries_dicts.append({
                    "title": getattr(entry, "title", ""),
                    "authors": getattr(entry, "authors", []),
                    "year": getattr(entry, "year", None),
                    "venue": getattr(entry, "venue", ""),
                })

        result["bibliography_entries"] = bibliography_entries_dicts
        result["bibliography_text"] = bibliography_text
        if citation_match_results:
            result["citation_match_results"] = citation_match_results

    return result

