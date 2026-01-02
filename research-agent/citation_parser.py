"""
Simplified citation parser using unified extraction utility.

This module now serves as a thin wrapper around the generic extraction_utils,
with citation-specific helper functions.
"""

from typing import Dict, List


def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Extract academic citations from text using LLM.

    This is the main entry point for citation extraction.
    Uses the unified extraction pattern from extraction_utils.

    Args:
        text: Text to extract citations from

    Returns:
        List of citation dictionaries with keys:
        - reference: The citation as it appears
        - context: What the paper discusses
        - relevance: Why it's relevant
        - citation_id: Unique ID for deduplication
    """
    from extraction_utils import extract_citations as unified_extract

    return unified_extract(text)


def deduplicate_citations(citations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate citations based on citation_id.

    Args:
        citations: List of citation dictionaries

    Returns:
        Deduplicated list of citations
    """
    seen_ids = set()
    unique = []

    for citation in citations:
        cid = citation.get("citation_id", "")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            unique.append(citation)

    return unique


def format_citations_for_prompt(citations: List[Dict[str, str]]) -> str:
    """
    Format citations for inclusion in LLM prompt.

    Args:
        citations: List of citation dictionaries

    Returns:
        Formatted string for prompt
    """
    if not citations:
        return "(No citations found in retrieved content)"

    formatted_lines = []
    for i, cite in enumerate(citations[:10], 1):  # Limit to top 10
        title = cite.get("title", "Unknown")
        context = cite.get("context", "")[:150]
        relevance = cite.get("relevance", "")

        formatted_lines.append(
            f"{i}. **{title}**\n"
            f"   Topic: {context}\n"
            f"   Relevance: {relevance}"
        )

    if len(citations) > 10:
        formatted_lines.append(f"\n... and {len(citations) - 10} more citations")

    return "\n\n".join(formatted_lines)


def create_citation_summary(citations: List[Dict[str, str]]) -> str:
    """
    Create a brief summary of extracted citations.

    Args:
        citations: List of citation dictionaries

    Returns:
        Summary string
    """
    if not citations:
        return "No citations extracted"

    total = len(citations)
    total = len(citations)
    titles = [cite.get("title", "Unknown") for cite in citations[:3]]

    summary = f"{total} citation(s) found"
    if titles:
        preview = ", ".join(titles)
        summary += f" ({preview}{'...' if total > 3 else ''})"

    return summary
