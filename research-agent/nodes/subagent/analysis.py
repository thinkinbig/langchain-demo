"""Analysis node: Analyze gathered context (Internal or Web) and call tools"""

from typing import Any, Dict, List

import context_manager
import tools
from citation_parser import (
    create_citation_summary,
    deduplicate_citations,
    extract_citations,
    format_citations_for_prompt,
)
from context_formatter import ContextFormatter
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from llm.factory import get_subagent_llm
from memory_helpers import (
    content_metadata_to_string,
    create_content_metadata,
)
from prompts import SUBAGENT_ANALYSIS, SUBAGENT_SYSTEM
from retrieval import RetrievalResult, RetrievalSource, Source
from schemas import Finding, SubagentState


def analysis_node(state: SubagentState):
    """Node: Analyze gathered context (Internal or Web) and call tools"""
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

    # Convert dict representation to RetrievalResult if needed
    # (from state serialization)
    if internal_result and not isinstance(internal_result, RetrievalResult):
        if isinstance(internal_result, dict):
            sources = [Source(**s) for s in internal_result.get("sources", [])]
            internal_result = RetrievalResult(
                content=internal_result.get("content", ""),
                sources=sources,
                source_type=RetrievalSource.INTERNAL,
                has_content=internal_result.get("has_content", False)
            )
        else:
            internal_result = None

    if web_result and not isinstance(web_result, RetrievalResult):
        if isinstance(web_result, dict):
            sources = [Source(**s) for s in web_result.get("sources", [])]
            web_result = RetrievalResult(
                content=web_result.get("content", ""),
                sources=sources,
                source_type=RetrievalSource.WEB,
                has_content=web_result.get("has_content", False)
            )
        else:
            web_result = None

    # Get visited identifiers for context formatting
    visited_sources = state.get("visited_sources", [])
    visited_identifiers = [vs.identifier for vs in visited_sources]

    # Use ContextFormatter to format context
    formatter = ContextFormatter()
    formatted_context, citation_instructions, all_sources = (
        formatter.format_for_analysis(
            task=task_description,
            internal_result=internal_result,
            web_result=web_result,
            visited_identifiers=visited_identifiers
        )
    )

    # Log context info and source tracking
    if internal_result and not internal_result.is_empty():
        source_ids = [s.identifier for s in internal_result.sources]
        content_len = len(internal_result.content)
        sources_count = len(internal_result.sources)
        print(
            f"     Found {content_len} chars of internal context "
            f"from {sources_count} sources: {source_ids[:3]}"
        )
    if web_result and not web_result.is_empty():
        source_urls = [s.identifier for s in web_result.sources]
        content_len = len(web_result.content)
        sources_count = len(web_result.sources)
        print(
            f"     Found {content_len} chars of web context "
            f"from {sources_count} sources: {source_urls[:3]}"
        )

    # Debug: Log total sources collected
    source_previews = [s.identifier[:30] for s in all_sources[:3]]
    print(
        f"  📚 [Source Tracking] Total sources available: "
        f"{len(all_sources)} ({source_previews})"
    )

    # Create Analysis Prompt
    analysis_prompt_content = SUBAGENT_ANALYSIS.format(
        task=task_description,
        results=formatted_context + citation_instructions
    )

    # --- Agent Loop (Same as before) ---
    system_prompt = context_manager.get_system_context(
        SUBAGENT_SYSTEM, include_knowledge=True
    )

    # Define Tools
    @tool
    def extract_citations_from_text(text: str) -> Dict[str, Any]:
        """
        Extract academic paper citations from the given text.

        Use this tool when you see citation patterns in the retrieved content
        (e.g., "Song et al., 2023", "Zhang et al., 2025", paper titles in quotes).
        These citations can be valuable leads for deeper research.

        Args:
            text: The text content to extract citations from
                (can be a portion of the retrieved content)

        Returns:
            Dictionary with:
            - citations: List of extracted citations with title, context, and relevance
            - count: Number of citations found
        """
        if not text or len(text.strip()) < 50:
            return {
                "citations": [],
                "count": 0,
                "message": "Text too short for citation extraction"
            }

        try:
            citations = extract_citations(text)
            count = len(citations)

            if citations:
                summary = create_citation_summary(citations)
                print(f"  📚 [Tool] Extracted {summary}")
                return {
                    "citations": citations,
                    "count": count,
                    "message": f"Successfully extracted {count} citation(s)"
                }
            else:
                return {
                    "citations": [],
                    "count": 0,
                    "message": "No citations found in the provided text"
                }
        except Exception as e:
            print(f"  ⚠️ [Tool] Citation extraction failed: {e}")
            return {
                "citations": [],
                "count": 0,
                "message": f"Extraction failed: {str(e)}"
            }

    @tool
    def submit_findings(summary: str, sources: List[Dict[str, str]] = None):
        """Submit the final synthesized findings."""
        return {"summary": summary, "sources": sources}

    tool_list = [
        tools.python_repl,
        tools.read_local_file,
        extract_citations_from_text,
        submit_findings
    ]
    formatted_llm = get_subagent_llm().bind_tools(tool_list)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=analysis_prompt_content),
    ]

    # Track citations extracted during agent loop
    agent_extracted_citations = []

    for _ in range(5):
        response = formatted_llm.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            for t_call in response.tool_calls:
                t_name = t_call["name"]
                t_args = t_call["args"]
                t_id = t_call["id"]

                print(f"    🛠️ [CodeAgent] Call: {t_name}")

                if t_name == "python_repl":
                    try:
                        result = tools.python_repl(t_args.get("code", ""))
                        messages.append(
                            ToolMessage(tool_call_id=t_id, content=str(result))
                        )
                    except Exception as e:
                        messages.append(
                            ToolMessage(tool_call_id=t_id, content=f"Error: {e}")
                        )

                elif t_name == "read_local_file":
                    res = tools.read_local_file(t_args.get("file_path", ""))
                    messages.append(ToolMessage(tool_call_id=t_id, content=res))

                elif t_name == "extract_citations_from_text":
                    # Agent requested citation extraction
                    text_to_extract = t_args.get("text", "")
                    if not text_to_extract:
                        # If no text provided, use formatted_context as fallback
                        text_to_extract = formatted_context

                    result = extract_citations_from_text.invoke(
                        {"text": text_to_extract}
                    )
                    citations_found = result.get("citations", [])

                    # Accumulate citations (agent might call multiple times)
                    agent_extracted_citations.extend(citations_found)

                    # Return formatted result to agent
                    if citations_found:
                        citation_text = format_citations_for_prompt(
                            citations_found
                        )
                        count = len(citations_found)
                        response_text = (
                            f"Extracted {count} citation(s):\n"
                            f"{citation_text}"
                        )
                    else:
                        response_text = result.get("message", "No citations found")

                    messages.append(
                        ToolMessage(tool_call_id=t_id, content=response_text)
                    )

                elif t_name == "submit_findings":
                    # Merging logic for sources
                    # Start with sources from retrieval results (ALWAYS include these)
                    final_sources = []
                    for source in all_sources:
                        if source.source_type.value == "internal":
                            doc_title = (
                                source.title
                                or f"Internal Document: {source.identifier}"
                            )
                            final_sources.append({
                                "title": doc_title,
                                "url": f"internal/{source.identifier}"
                            })
                        else:
                            final_sources.append({
                                "title": source.title or source.identifier,
                                "url": source.identifier
                            })

                    # Add tool-provided sources (from LLM)
                    # Merge with retrieval sources
                    tool_sources = t_args.get("sources")
                    if tool_sources:
                        for s in tool_sources:
                            # Check if source already exists by URL
                            url = s.get("url", "")
                            if url and not any(
                                fs.get("url") == url for fs in final_sources
                            ):
                                final_sources.append(s)

                    # Log source collection for debugging
                    source_names = [
                        s.get('title', s.get('url', 'Unknown'))
                        for s in final_sources[:3]
                    ]
                    sources_count = len(final_sources)
                    print(
                        f"  📚 [Sources] Collected {sources_count} "
                        f"sources: {source_names}"
                    )

                    # Store content metadata instead of full content
                    # (token optimization)
                    # Full content is available in formatted_context
                    # but we store only metadata
                    if formatted_context:
                        content_metadata = create_content_metadata(
                            formatted_context, max_preview=200
                        )
                        evidence_content = content_metadata_to_string(content_metadata)
                    else:
                        evidence_content = ""

                    # Use citations extracted by agent (from tool calls)
                    # Also extract from summary as fallback
                    # (in case agent didn't call the tool)
                    summary_text = t_args.get("summary", "")
                    summary_citations = (
                        extract_citations(summary_text)
                        if summary_text else []
                    )

                    # Merge: agent-extracted + summary citations
                    all_citations_list = (
                        agent_extracted_citations + summary_citations
                    )
                    extracted_citations = deduplicate_citations(
                        all_citations_list
                    )

                    # Log citation collection
                    if agent_extracted_citations:
                        agent_summary = create_citation_summary(
                            agent_extracted_citations
                        )
                        print(
                            f"  📚 [Citations] Agent extracted "
                            f"{agent_summary} via tool"
                        )
                    if summary_citations:
                        summary_summary = create_citation_summary(
                            summary_citations
                        )
                        print(
                            f"  📚 [Citations] Additional "
                            f"{summary_summary} from summary"
                        )

                    # Now stores metadata dict as JSON string
                    finding = Finding(
                        task=task_description,
                        summary=t_args.get("summary", ""),
                        sources=final_sources,
                        content=evidence_content,
                        extracted_citations=extracted_citations
                    )
                    print("  ✅ CodeAgent finished via tool.")
                    return {
                        "subagent_findings": [finding],
                        "extracted_citations": extracted_citations
                    }
        else:
            print("  ⚠️  CodeAgent output text but no tool call. Re-prompting...")
            messages.append(
                HumanMessage(content="You generally must call `submit_findings`.")
            )

    # Fail safe
    print("  ❌ CodeAgent failed to submit findings.")
    return {
        "subagent_findings": [
            Finding(task=task_description, summary="Analysis failed", sources=[])
        ]
    }

