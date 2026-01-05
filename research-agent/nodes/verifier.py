"""Verifier node: Cross-check synthesized report against source evidence"""

from typing import Any, Dict, List, Tuple

import context_manager
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from prompts import VERIFIER_MAIN, VERIFIER_SYSTEM
from schemas import (
    ResearchState,
    VerificationResult,
    extract_evidence_summaries,
)
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


def validate_citation_format(citation: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single citation format.
    
    Args:
        citation: Citation dictionary with keys like title, authors, year, etc.
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check for required fields
    title = citation.get("title", "").strip()
    if not title or len(title) < 3:
        issues.append("Missing or too short title")

    # Check if citation has at least some identifying information
    has_authors = bool(citation.get("authors")) or bool(citation.get("author"))
    has_year = bool(citation.get("year"))
    has_url = bool(citation.get("url", "").strip())

    # A valid citation should have at least title + (authors or year or url)
    if not (has_authors or has_year or has_url):
        issues.append(
            "Citation lacks sufficient identifying information "
            "(authors, year, or URL)"
        )

    # Validate year if present
    year = citation.get("year")
    if year is not None:
        try:
            year_int = int(year) if isinstance(year, str) else year
            if year_int < 1900 or year_int > 2100:
                issues.append(f"Year {year_int} seems invalid")
        except (ValueError, TypeError):
            issues.append(f"Year '{year}' is not a valid number")

    is_valid = len(issues) == 0
    return is_valid, issues


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


def verifier_node(state: ResearchState):
    """Verifier: Cross-check synthesized report against source evidence"""
    report = state.get("synthesized_results", "")
    findings = state.get("subagent_findings", [])

    if not report or not findings:
        return {"final_report": report}

    print("\n🕵️  [Verifier] Cross-checking report against source evidence...")

    # Extract evidence summaries before aggregation (token optimization)
    # Use summaries and metadata instead of full content
    evidence_summaries = extract_evidence_summaries(findings, max_length=500)

    # Aggregate evidence from summaries (much smaller than full content)
    evidence_pieces = []
    # Collect full content for bibliography detection
    all_content = []

    # 1. From Findings (Subagent work) - use summaries instead of full content
    for i, f in enumerate(findings):
        if evidence_summaries and i < len(evidence_summaries):
            evidence_pieces.append(evidence_summaries[i])
        elif f.summary:
            # Fallback to summary if extraction failed
            evidence_pieces.append(
                f"Task: {f.task[:50]}\nEvidence: {f.summary[:500]}"
            )

        # Collect full content for bibliography detection
        if hasattr(f, 'content') and f.content:
            all_content.append(f.content)

    # 2. From RAG (Direct verification)
    print("  🧠 [Verifier] Retrieving verification context...")
    # Use original query for RAG retrieval (more accurate than report snippet)
    # The query represents the core topic, while report may start with
    # intro/background
    original_query = state.get("query", "")
    if original_query:
        verification_query = original_query
    else:
        # Fallback: extract key sentences from report (first paragraph or
        # first 300 chars). This is better than just first 200 chars which
        # might be just title
        lines = report.split('\n')
        first_paragraph = lines[0] if lines else report[:300]
        verification_query = first_paragraph.strip()[:300]

    rag_evidence, _ = context_manager.retrieve_knowledge(verification_query)
    if rag_evidence:
        evidence_pieces.append(f"Internal Knowledge Base:\n{rag_evidence}")
        all_content.append(rag_evidence)

    if not evidence_pieces:
        print("  ⚠️  No full source text available for verification. Skipping.")
        return {}  # No change

    evidence_text = "\n\n".join(evidence_pieces)

    # 3. Detect and parse bibliography section (NEW)
    print("  📚 [Verifier] Detecting bibliography/reference section...")
    bibliography_text = ""
    bibliography_entries = []

    # Search for bibliography in all content
    combined_content = "\n\n".join(all_content)
    bibliography_raw = detect_bibliography_section(combined_content)

    if bibliography_raw:
        print("  ✅ Bibliography section detected, parsing...")
        try:
            from memory.bibliography_parser import parse_bibliography

            # Use turbo model for bibliography parsing
            # (lighter than plus)
            bib_llm = get_llm_by_model_choice("turbo")
            bibliography_entries = parse_bibliography(
                bibliography_raw, bib_llm
            )

            if bibliography_entries:
                print(f"  ✅ Parsed {len(bibliography_entries)} bibliography entries")
                # Format bibliography for prompt
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
                bibliography_text = bibliography_raw[:1000]
        except Exception as e:
            print(f"  ⚠️  Bibliography parsing failed: {e}, using raw text")
            bibliography_text = bibliography_raw[:1000]
    else:
        print("  ⚠️  No bibliography/reference section found in sources")

    # 4. Collect extracted citations (NEW)
    print("  📝 [Verifier] Collecting extracted citations...")
    extracted_citations_raw = state.get("all_extracted_citations", [])
    extracted_citations = []
    citation_format_issues = []

    for cite in extracted_citations_raw:
        # Convert to dict if needed
        if isinstance(cite, dict):
            cite_dict = cite
        elif hasattr(cite, "model_dump"):
            cite_dict = cite.model_dump()
        elif hasattr(cite, "dict"):
            cite_dict = cite.dict()
        else:
            cite_dict = dict(cite)

        # Validate citation format
        is_valid, issues = validate_citation_format(cite_dict)
        if not is_valid:
            citation_format_issues.extend(issues)

        extracted_citations.append(cite_dict)

    # Format extracted citations for prompt
    extracted_citations_text = ""
    if extracted_citations:
        cite_lines = []
        for i, cite in enumerate(extracted_citations, 1):
            title = cite.get("title", "Unknown")
            authors = cite.get("authors", [])
            year = cite.get("year", "")
            context = cite.get("context", "")

            cite_line = f"{i}. {title}"
            if authors:
                cite_line += f" by {', '.join(authors[:2])}"
            if year:
                cite_line += f" ({year})"
            if context:
                cite_line += f" - {context[:100]}"
            cite_lines.append(cite_line)

        extracted_citations_text = "\n".join(cite_lines)
    else:
        extracted_citations_text = "(No extracted citations found)"

    # 5. Match citations to bibliography (NEW)
    citation_match_results = None
    if bibliography_entries and extracted_citations:
        print("  🔍 [Verifier] Matching citations to bibliography...")
        citation_match_results = match_citations_to_bibliography(
            extracted_citations, bibliography_entries
        )

        matched_count = len(citation_match_results["matched"])
        unmatched_count = len(citation_match_results["unmatched"])
        print(f"  📊 Matched: {matched_count}, Unmatched: {unmatched_count}")

        if unmatched_count > 0:
            print(f"  ⚠️  {unmatched_count} citations not found in bibliography")

    # Invoke Verifier LLM
    bibliography_for_prompt = (
        bibliography_text
        if bibliography_text
        else "(No bibliography section found)"
    )
    prompt_content = VERIFIER_MAIN.format(
        report=report,
        evidence=evidence_text,
        bibliography=bibliography_for_prompt,
        extracted_citations=extracted_citations_text,
    )

    # Use lead LLM (stronger model) for verification
    llm = get_llm_by_model_choice("plus")
    structured_llm = llm.with_structured_output(VerificationResult)

    try:
        response = structured_llm.invoke([
            SystemMessage(content=VERIFIER_SYSTEM),
            HumanMessage(content=prompt_content)
        ])

        # Log citation verification results
        bibliography_found = bool(bibliography_text)
        citation_format_valid = len(citation_format_issues) == 0

        # Add citation issues from matching results
        citation_issues = (
            list(response.citation_issues)
            if hasattr(response, "citation_issues")
            else []
        )

        if citation_format_issues:
            citation_issues.extend(
                [f"Format issue: {issue}" for issue in citation_format_issues]
            )

        if citation_match_results:
            unmatched = citation_match_results["unmatched"]
            if unmatched:
                unmatched_titles = [
                    c.get("title", "Unknown")[:50]
                    for c in unmatched[:5]
                ]
                citation_issues.append(
                    f"{len(unmatched)} citations not found in "
                    f"bibliography: {', '.join(unmatched_titles)}"
                )

        if not bibliography_found and extracted_citations:
            citation_issues.append(
                "Bibliography/reference section not found, "
                "but citations were extracted"
            )

        if (
            response.is_valid
            and citation_format_valid
            and len(citation_issues) == 0
        ):
            print(
                "  ✅ Report verified: Structure, facts, "
                "and citations appear accurate."
            )
            if bibliography_found:
                print(
                    f"  ✅ Bibliography found with "
                    f"{len(bibliography_entries)} entries"
                )
            return {}  # No change
        else:
            corrections_count = len(response.corrections)
            issues_summary = []
            if corrections_count > 0:
                issues_summary.append(f"{corrections_count} factual corrections")
            if len(citation_issues) > 0:
                issues_summary.append(f"{len(citation_issues)} citation issues")
            if not citation_format_valid:
                issues_summary.append("citation format issues")

            print(f"  ⚠️  Issues found: {', '.join(issues_summary)}")

            # Log citation issues
            if citation_issues:
                print("  📋 Citation issues:")
                for issue in citation_issues[:5]:  # Show first 5 issues
                    print(f"     - {issue}")
                if len(citation_issues) > 5:
                    print(f"     ... and {len(citation_issues) - 5} more")

            # Clean corrected report to remove any XML tags
            cleaned_report = clean_report_output(response.corrected_report)
            return {
                "synthesized_results": cleaned_report
            }

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return {}

