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

    # 1. From Findings (Subagent work) - use summaries instead of full content
    for i, f in enumerate(findings):
        if evidence_summaries and i < len(evidence_summaries):
            evidence_pieces.append(evidence_summaries[i])
        elif f.summary:
            # Fallback to summary if extraction failed
            evidence_pieces.append(
                f"Task: {f.task[:50]}\nEvidence: {f.summary[:500]}"
            )

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

    if not evidence_pieces:
        print("  ⚠️  No full source text available for verification. Skipping.")
        return {}  # No change

    evidence_text = "\n\n".join(evidence_pieces)

    # 2. Get bibliography information from Citation Agent (already processed)
    print("  📚 [Verifier] Using bibliography information from Citation Agent...")
    bibliography_text = state.get("bibliography_text", "")
    bibliography_entries = state.get("bibliography_entries", [])
    citation_match_results = state.get("citation_match_results")

    if bibliography_text:
        print(f"  ✅ Using bibliography with {len(bibliography_entries)} entries from Citation Agent")
    else:
        print("  ℹ️  No bibliography found by Citation Agent (this is normal)")

    # 3. Collect extracted citations
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

    # 4. Use citation match results from Citation Agent (if available)
    if citation_match_results:
        matched_count = len(citation_match_results.get("matched", []))
        unmatched_count = len(citation_match_results.get("unmatched", []))
        print(f"  📊 Citation matching results from Citation Agent: Matched: {matched_count}, Unmatched: {unmatched_count}")

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

        # Add citation issues from matching results (from Citation Agent)
        citation_issues = (
            list(response.citation_issues)
            if hasattr(response, "citation_issues")
            else []
        )

        if citation_format_issues:
            citation_issues.extend(
                [f"Format issue: {issue}" for issue in citation_format_issues]
            )

        # Use citation match results from Citation Agent
        if citation_match_results:
            unmatched = citation_match_results.get("unmatched", [])
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
                "Bibliography/reference section not found by Citation Agent, "
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

