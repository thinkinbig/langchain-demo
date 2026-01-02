"""
Unified extraction utility for LLM-based content analysis.

This module provides a generic pattern for extracting structured information
from unstructured text using LLM agents. It can be used for:
- Citation extraction from research papers
- Key insights from web search results
- Any custom extraction task

Design principle: Let the agent decide what's important rather than
hardcoding patterns.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


def extract_with_llm(
    text: str,
    prompt_template: str,
    result_schema: type[BaseModel],
    llm=None,
    **prompt_kwargs
) -> List[Dict[str, Any]]:
    """
    Generic LLM-based extraction function.

    This is the unified pattern for all extraction tasks:
    - Citations from PDFs
    - Insights from web search
    - Custom extractions

    Args:
        text: Text to extract from
        prompt_template: Prompt template with {text} and any custom placeholders
        result_schema: Pydantic schema for structured output
        llm: Optional LLM instance (defaults to subagent LLM)
        **prompt_kwargs: Additional variables for prompt template

    Returns:
        List of extracted items as dictionaries

    Example:
        >>> from prompts import CITATION_EXTRACTION
        >>> from schemas import CitationExtractionResult
        >>> citations = extract_with_llm(
        ...     text="Song et al., 2023 proposed...",
        ...     prompt_template=CITATION_EXTRACTION,
        ...     result_schema=CitationExtractionResult
        ... )
    """
    if not text or len(text.strip()) < 20:
        return []

    # Import here to avoid circular dependency
    if llm is None:
        try:
            from llm.factory import get_subagent_llm
            llm = get_subagent_llm()
        except Exception:
            # Fallback: if LLM not available, return empty
            return []

    # Format prompt with text and any additional kwargs
    prompt = prompt_template.format(text=text, **prompt_kwargs)

    try:
        # Use structured output for reliable parsing
        structured_llm = llm.with_structured_output(result_schema)
        result = structured_llm.invoke(prompt)

        # Convert to list of dicts
        # Handle both single object and list of objects
        if hasattr(result, 'citations'):
            # Citation extraction result
            return [item.model_dump() for item in result.citations]
        elif hasattr(result, 'findings'):
            # Web content extraction result
            return [item.model_dump() for item in result.findings]
        elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
            # List of items
            return [item.model_dump() if hasattr(item, 'model_dump') else item
                    for item in result]
        elif hasattr(result, 'model_dump'):
            # Single Pydantic model
            return [result.model_dump()]
        else:
            # Unknown format, return as-is
            return [result] if result else []

    except Exception as e:
        # Graceful degradation
        print(f"  ⚠️ [Extraction] LLM extraction failed: {e}")
        return []


# Specialized extraction functions using the unified pattern
def extract_citations(text: str, llm=None) -> List[Dict[str, str]]:
    """Extract academic citations from text using LLM."""
    from prompts import CITATION_EXTRACTION
    from schemas import CitationExtractionResult

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


def extract_web_insights(text: str, query: str, llm=None) -> List[Dict[str, str]]:
    """
    Extract key insights from web search results using LLM.

    Args:
        text: Web search results text
        query: The research question being answered
        llm: Optional LLM instance

    Returns:
        List of extracted insights with source and relevance
    """
    from prompts import WEB_CONTENT_EXTRACTION

    # Define schema for web content extraction
    class WebInsight(BaseModel):
        """Single insight from web content"""
        insight: str = Field(..., description="The main fact or finding")
        source: str = Field(..., description="URL or source title")
        relevance: str = Field(..., description="Why this is relevant")

    class WebExtractionResult(BaseModel):
        """Result of web content extraction"""
        findings: List[WebInsight] = Field(default_factory=list)

    results = extract_with_llm(
        text=text[:3000],  # Limit text length
        prompt_template=WEB_CONTENT_EXTRACTION,
        result_schema=WebExtractionResult,
        llm=llm,
        query=query
    )

    return results


def extract_custom(
    text: str,
    extraction_type: str,
    instructions: str,
    context_info: str = "",
    llm=None
) -> List[Dict[str, Any]]:
    """
    Custom extraction for any use case.

    Args:
        text: Text to extract from
        extraction_type: What to extract (e.g., "key metrics", "action items")
        instructions: Specific instructions for what to look for
        context_info: Additional context for the agent
        llm: Optional LLM instance

    Returns:
        List of extracted items

    Example:
        >>> metrics = extract_custom(
        ...     text="Revenue was $5M with 20% growth...",
        ...     extraction_type="financial metrics",
        ...     instructions="Extract metrics with their values and time periods"
        ... )
    """
    from prompts import GENERIC_EXTRACTION

    # Define a generic schema
    class GenericItem(BaseModel):
        """Generic extracted item"""
        content: str = Field(..., description="The extracted content")
        context: str = Field(default="", description="Context or explanation")

    class GenericExtractionResult(BaseModel):
        """Generic extraction result"""
        items: List[GenericItem] = Field(default_factory=list)

    results = extract_with_llm(
        text=text,
        prompt_template=GENERIC_EXTRACTION,
        result_schema=GenericExtractionResult,
        llm=llm,
        extraction_type=extraction_type,
        extraction_instructions=instructions,
        context_info=context_info
    )

    return results
