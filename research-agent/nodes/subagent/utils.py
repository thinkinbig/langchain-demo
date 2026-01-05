"""Utility functions for subagent analysis node refactoring"""

import asyncio
from typing import List, Tuple

import context_manager
from context_formatter import ContextFormatter
from langchain_core.messages import HumanMessage
from prompts import SUBAGENT_ANALYSIS_RETRY
from retrieval import RetrievalResult, RetrievalSource, Source
from schemas import (
    AnalysisOutput,
    Finding,
    SubagentState,
    VisitedSource,
    content_metadata_to_string,
    create_content_metadata,
)


async def invoke_llm_with_retry(
    structured_llm,
    messages: List,
    max_retries: int = 2,
) -> AnalysisOutput:
    """
    Invoke LLM with structured output and retry logic.

    Handles timeouts, exceptions, and parsing errors with automatic retries.

    Args:
        structured_llm: LLM instance with structured output configured
        messages: List of messages for the LLM
        max_retries: Maximum number of retry attempts (default: 2)

    Returns:
        AnalysisOutput: Parsed structured output from LLM
    """
    from config import settings

    for attempt in range(max_retries):
        try:
            # Async invoke with timeout
            response = await asyncio.wait_for(
                structured_llm.ainvoke(messages),
                timeout=settings.TIMEOUT_LLM_CALL
            )
        except asyncio.TimeoutError:
            error_msg = f"LLM call timeout ({settings.TIMEOUT_LLM_CALL}s)"
            print(f"  ⚠️  [Analysis] Timeout during invoke: {error_msg}")
            if attempt < max_retries - 1:  # Only retry if not last attempt
                messages.append(HumanMessage(content=SUBAGENT_ANALYSIS_RETRY.format(
                    previous_prompt="",
                    error=error_msg
                ).content))
                continue
            else:
                # Max retries reached, return fallback
                break
        except Exception as e:
            # Manually handle Pydantic validation errors that might bubble up
            error_msg = f"Validation Error: {str(e)}"
            print(f"  ⚠️  [Analysis] Exception during invoke: {error_msg}")
            if attempt < max_retries - 1:
                messages.append(HumanMessage(content=SUBAGENT_ANALYSIS_RETRY.format(
                    previous_prompt="",
                    error=error_msg
                ).content))
                continue
            else:
                break

        # Check for parsing error
        if response.get("parsing_error"):
            error_msg = str(response["parsing_error"])
            print(f"  ⚠️  [Analysis] Validation error: {error_msg}")
            if attempt < max_retries - 1:
                # Construct Retry Prompt (Error Prompt Pattern)
                # Maximize KV cache: Append error to history instead of rewriting
                messages.append(HumanMessage(content=SUBAGENT_ANALYSIS_RETRY.format(
                    previous_prompt="",  # Context already in history
                    error=error_msg
                ).content))
                continue
            else:
                break

        return response["parsed"]

    # Fallback if retries failed
    print("  ❌ [Analysis] Failed to parse structured output after retries.")
    return AnalysisOutput(
        summary="Analysis failed due to validation errors.",
        citations=[],
        reasoning="Failed validation"
    )


def enhance_context_with_references(all_sources: List[Source]) -> str:
    """
    Enhance context by retrieving "References" sections from internal papers.

    Args:
        all_sources: List of sources identified in the context

    Returns:
        String containing retrieved References sections
    """
    internal_sources = [
        s.identifier for s in all_sources
        if s.source_type == RetrievalSource.INTERNAL
    ]

    if not internal_sources:
        return ""

    print(
        f"  📚 [Analysis] Enhancing context with Reference sections "
        f"from {len(internal_sources)} papers..."
    )

    enhanced_content_parts = []

    # Process each internal source
    for source_id in internal_sources:
        # Heavily biased retrieval query to find the references section
        # We assume the user has ingested papers where "Refeferences" or "Bibliography"
        # is a distinct section.
        query = f"References Bibliography Citations section in {source_id}"

        try:
            # We use a direct retrieval call focused on this specific source
            # Note: The context_manager logic retrieves based on query similarity.
            # To target a specific file, we rely on the source name being in the chunks
            # or the query retrieving those specific chunks.
            # For a more robust solution, we might need a dedicated `retrieve_from_source` method
            # but for now we'll use the query mechanism with high likelihood of hitting the ref section.

            # TODO: Improve source filtering in context_manager if this proves unreliable

            context, _ = context_manager.retrieve_knowledge(
                query,
                k=2  # Just get top 2 chunks which likely contain the refs
            )

            if context and "No relevant internal documents" not in context:
                enhanced_content_parts.append(
                    f"\n--- OFFICIAL REFERENCES SECTION FROM {source_id} ---\n"
                    f"{context}\n"
                    f"--- END REFERENCES FROM {source_id} ---\n"
                )

        except Exception as e:
            print(f"  ⚠️  Failed to retrieve references for {source_id}: {e}")

    if not enhanced_content_parts:
        return ""

    return "\n".join(enhanced_content_parts)


def prepare_analysis_context(
    state: SubagentState
) -> Tuple[str, str, List[Source], str]:
    """
    Prepare context for analysis node from state.

    Handles state deserialization, context formatting, and message preparation.

    Args:
        state: SubagentState containing retrieval results

    Returns:
        Tuple of (formatted_context, citation_instructions, all_sources, task_description)
    """
    # Get task description
    task_description = state.get("task_description", "")
    if not task_description:
        tasks = state.get("subagent_tasks", [])
        if tasks:
            rs_task = tasks[0]
            task_description = rs_task.description

    # Get retrieval results (unified format)
    internal_result = state.get("internal_result")
    web_result = state.get("web_result")
    paper_result = state.get("paper_result")

    # Handle dict serialization - re-hydrating objects if they are dicts
    if internal_result and isinstance(internal_result, dict):
        sources = [Source(**s) for s in internal_result.get("sources", [])]
        internal_result = RetrievalResult(
            content=internal_result.get("content", ""),
            sources=sources,
            source_type=RetrievalSource.INTERNAL,
            has_content=internal_result.get("has_content", False)
        )
    if web_result and isinstance(web_result, dict):
        sources = [Source(**s) for s in web_result.get("sources", [])]
        web_result = RetrievalResult(
            content=web_result.get("content", ""),
            sources=sources,
            source_type=RetrievalSource.WEB,
            has_content=web_result.get("has_content", False)
        )
    if paper_result and isinstance(paper_result, dict):
        sources = [Source(**s) for s in paper_result.get("sources", [])]
        paper_result = RetrievalResult(
            content=paper_result.get("content", ""),
            sources=sources,
            source_type=RetrievalSource.PAPER,
            has_content=paper_result.get("has_content", False)
        )

    visited_sources = state.get("visited_sources", [])
    visited_identifiers = [vs.identifier for vs in visited_sources]

    formatter = ContextFormatter()
    formatted_context, citation_instructions, all_sources = (
        formatter.format_for_analysis(
            task=task_description,
            internal_result=internal_result,
            web_result=web_result,
            paper_result=paper_result,
            visited_identifiers=visited_identifiers
        )
    )

    # ENHANCEMENT: Automatically retrieve References section for internal papers
    references_context = enhance_context_with_references(all_sources)
    if references_context:
        formatted_context += f"\n\n{references_context}"

    return formatted_context, citation_instructions, all_sources, task_description


def build_finding_from_analysis(
    analysis: AnalysisOutput,
    task_description: str,
    all_sources: List[Source],
    formatted_context: str
) -> Finding:
    """
    Build Finding object from analysis output and context.

    Args:
        analysis: AnalysisOutput from LLM
        task_description: Task description string
        all_sources: List of all sources used
        formatted_context: Formatted context string

    Returns:
        Finding: Complete Finding object ready for output
    """
    # Prepare sources list
    final_sources = []
    for source in all_sources:
        if source.source_type.value == "internal":
            final_sources.append({
                "title": source.title or f"Internal Document: {source.identifier}",
                "url": f"internal/{source.identifier}"
            })
        else:
            final_sources.append({
                "title": source.title or source.identifier,
                "url": source.identifier
            })

    # Prepare content metadata
    evidence_content = ""
    if formatted_context:
        content_metadata = create_content_metadata(
            formatted_context, max_preview=200
        )
        evidence_content = content_metadata_to_string(content_metadata)

    # Convert schema Citations to dictionaries
    extracted_citations_dicts = [
        c.model_dump() for c in analysis.citations
    ]

    if extracted_citations_dicts:
        print(f"  📚 [Analysis] Extracted {len(extracted_citations_dicts)} citations")

    finding = Finding(
        task=task_description,
        summary=analysis.summary,
        sources=final_sources,
        content=evidence_content,
        extracted_citations=extracted_citations_dicts
    )

    return finding


def citations_to_visited_sources(citations: List[dict]) -> List[VisitedSource]:
    """
    Convert extracted citations to VisitedSource objects.

    Only processes citations that have a URL (for deduplication purposes).
    Uses source_type="citation" to distinguish from actually visited sources.

    Args:
        citations: List of citation dictionaries with keys:
            - title: Citation title
            - url: Source URL (required for conversion)
            - context: Optional context
            - relevance: Optional relevance info

    Returns:
        List of VisitedSource objects with source_type="citation"
    """
    visited_sources = []
    seen_identifiers = set()

    for citation in citations:
        # Only process citations with URLs (needed for deduplication)
        url = citation.get("url", "").strip()
        if not url:
            continue

        # Skip if we've already seen this identifier
        if url in seen_identifiers:
            continue

        seen_identifiers.add(url)

        # Create VisitedSource with citation type
        visited_source = VisitedSource(
            identifier=url,
            source_type="citation"
        )
        visited_sources.append(visited_source)

    return visited_sources

