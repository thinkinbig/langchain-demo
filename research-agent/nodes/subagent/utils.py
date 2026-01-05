"""Utility functions for subagent analysis node refactoring"""

import asyncio
from typing import List, Tuple

from context_formatter import ContextFormatter
from langchain_core.messages import HumanMessage
from prompts import SUBAGENT_ANALYSIS_RETRY
from retrieval import RetrievalResult, RetrievalSource, Source
from schemas import (
    AnalysisOutput,
    Finding,
    SubagentState,
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

    visited_sources = state.get("visited_sources", [])
    visited_identifiers = [vs.identifier for vs in visited_sources]

    formatter = ContextFormatter()
    formatted_context, citation_instructions, all_sources = (
        formatter.format_for_analysis(
            task=task_description,
            internal_result=internal_result,
            web_result=web_result,
            visited_identifiers=visited_identifiers
        )
    )

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

