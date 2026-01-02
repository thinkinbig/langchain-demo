"""
Citation extraction and processing utilities.

This module handles citation extraction from text using LLM-based extraction.
"""

from typing import Dict, List


def extract_citations(text: str, llm=None) -> List[Dict[str, str]]:
    """
    Extract academic citations from text using LLM.

    Args:
        text: Text to extract citations from
        llm: Optional LLM instance (defaults to subagent LLM)

    Returns:
        List of citation dictionaries with keys:
        - title: The citation title or reference
        - context: What the paper discusses
        - relevance: Why it's relevant
        - citation_id: Unique ID for deduplication
    """
    from extraction_service import extract_with_llm
    from prompts import CITATION_EXTRACTION
    from schemas import CitationExtractionResult

    if not text or len(text.strip()) < 20:
        return []

    results = extract_with_llm(
        text=text[:2000],  # Limit text length
        prompt_template=CITATION_EXTRACTION,
        result_schema=CitationExtractionResult,
        llm=llm
    )

    # Add citation_id for deduplication
    for item in results:
        # Use title for ID generation
        if 'title' in item:
            item['citation_id'] = str(hash(item['title']))

    return results


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
