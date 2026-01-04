"""Synthesizer node: Aggregate and synthesize all findings"""

import hashlib

from config import settings
from graph.utils import process_structured_response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from memory.temporal_memory import TemporalMemory
from prompts import (
    REFLECTION_MAIN,
    REFLECTION_SYSTEM,
    SYNTHESIZER_MAIN,
    SYNTHESIZER_REFINE,
    SYNTHESIZER_RETRY,
    SYNTHESIZER_SCR_MAIN,
    SYNTHESIZER_SCR_REFINE,
    SYNTHESIZER_SCR_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from schemas import (
    ReflectionResult,
    SCRResult,
    SynthesisResult,
    SynthesizerState,
    extract_findings_metadata,
)


def _compute_finding_hash(finding: dict) -> str:
    """Compute hash for a finding to track if it's been processed"""
    task = finding.get('task', '')
    summary = finding.get('summary', '')
    return hashlib.sha256((task + summary).encode()).hexdigest()[:16]


def _get_recommended_model(state: SynthesizerState, default: str = "plus") -> str:
    """
    Get recommended model from complexity analysis.
    Automatically downgrades max to plus if ENABLE_MAX_MODEL is False.

    Args:
        state: SynthesizerState containing complexity_analysis
        default: Default model if not found (default: "plus")

    Returns:
        Model choice: "turbo", "plus", or "max" (max only if ENABLE_MAX_MODEL=True)
    """
    from config import settings

    complexity_analysis = state.get("complexity_analysis")
    if complexity_analysis:
        if hasattr(complexity_analysis, "recommended_model"):
            model = complexity_analysis.recommended_model
            if model in ["turbo", "plus", "max"]:
                # Downgrade max to plus if not enabled
                if model == "max" and not settings.ENABLE_MAX_MODEL:
                    return "plus"
                return model
        elif isinstance(complexity_analysis, dict):
            model = complexity_analysis.get("recommended_model", default)
            if model in ["turbo", "plus", "max"]:
                # Downgrade max to plus if not enabled
                if model == "max" and not settings.ENABLE_MAX_MODEL:
                    return "plus"
                return model
    return default


def synthesizer_node(state: SynthesizerState):
    """Synthesizer: Aggregate and synthesize all findings"""
    # Prefer filtered findings if available (Guardrail active)
    findings = state.get("filtered_findings", [])
    if not findings:
        # Fallback to raw findings
        findings = state.get("subagent_findings", [])
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")
    existing_messages = state.get("synthesizer_messages", [])
    processed_finding_ids = set(state.get("processed_findings_ids", []))
    previous_synthesis = state.get("synthesized_results", "")

    if not findings:
        return {"synthesized_results": "No findings to synthesize."}

    # Identify new findings (not yet processed)
    new_findings = []
    for f in findings:
        f_hash = _compute_finding_hash(f)
        if f_hash not in processed_finding_ids:
            new_findings.append(f)

    if not new_findings and previous_synthesis:
        # No new findings, return existing synthesis
        print("\n📊 [Synthesizer] No new findings to process.")
        return {"synthesized_results": previous_synthesis}

    print(f"\n📊 [Synthesizer] Processing {len(new_findings)} new findings "
          f"(out of {len(findings)} total)...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    # Build conversation history incrementally
    messages = list(existing_messages) if existing_messages else []

    # Determine if this is incremental or first-time synthesis
    if not messages:
        # First synthesis: Initialize conversation
        findings_to_process = findings  # Process all on first run
        is_incremental = False
    else:
        # Subsequent synthesis: Only process new findings
        findings_to_process = new_findings
        is_incremental = True

    # Determine if we should use SCR structure
    # Only use SCR for first-time synthesis, not incremental updates
    use_scr = settings.USE_SCR_STRUCTURE and not is_incremental

    if not messages:
        # First synthesis: Initialize conversation with appropriate system
        if use_scr:
            messages.append(SystemMessage(content=SYNTHESIZER_SCR_SYSTEM))
        else:
            messages.append(SystemMessage(content=SYNTHESIZER_SYSTEM))

    if not findings_to_process:
        return {
            "synthesized_results": (
                previous_synthesis or "No findings to synthesize."
            )
        }

    # Pre-compute findings metadata (token optimization)
    findings_metadata = extract_findings_metadata(findings_to_process)

    # Format findings with sources and citations for detailed synthesis
    findings_text_parts = []
    for i, f in enumerate(findings_to_process, 1):
        task = f.get('task', 'Unknown')
        summary = f.get('summary', 'No summary')
        sources = f.get('sources', [])
        citations = f.get('extracted_citations', [])

        finding_text = f"{i}. Task: {task}\n   Summary: {summary}"

        # Include sources for context
        if sources:
            source_list = ", ".join([
                s.get('title', 'Unknown')[:60]
                for s in sources[:5]  # Limit to 5 sources per finding
            ])
            if len(sources) > 5:
                source_list += f" (+{len(sources) - 5} more)"
            finding_text += f"\n   Sources: {source_list}"

        # Include extracted citations if available
        if citations:
            citation_titles = [
                c.get('title', 'Unknown')[:60]
                for c in citations[:3]  # Limit to 3 citations per finding
            ]
            if citation_titles:
                finding_text += (
                    f"\n   Mentioned Papers: {', '.join(citation_titles)}"
                )
                if len(citations) > 3:
                    finding_text += f" (+{len(citations) - 3} more)"

        findings_text_parts.append(finding_text)

    findings_text = "\n\n".join(findings_text_parts)

    # Add metadata context to prompt (helps LLM understand scope)
    count = findings_metadata['count']
    total_sources = findings_metadata['total_sources']
    avg_length = findings_metadata['avg_summary_length']
    metadata_context = (
        f"\n\n[Metadata: {count} findings, "
        f"{total_sources} sources, "
        f"avg length: {avg_length} chars]"
    )

    # Build prompt - include previous synthesis if incremental
    if is_incremental and previous_synthesis:
        prompt_content = (
            f"<query>\n{query}\n</query>\n\n"
            f"<previous_synthesis>\n{previous_synthesis}\n</previous_synthesis>\n\n"
            f"<new_findings>\n{findings_text + metadata_context}\n</new_findings>\n\n"
            "<instructions>\n"
            "Please synthesize the new findings above and integrate them into "
            "the previous synthesis. Expand and update the synthesis to include "
            "the new information while maintaining coherence with the existing "
            "content.\n"
            "</instructions>"
        )
    else:
        if use_scr:
            prompt_content = SYNTHESIZER_SCR_MAIN.format(
                query=query,
                findings=findings_text + metadata_context
            )
        else:
            prompt_content = SYNTHESIZER_MAIN.format(
                query=query,
                findings=findings_text + metadata_context
            )

    if last_error:
        prompt_content = SYNTHESIZER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    # Use appropriate schema based on SCR setting
    if use_scr:
        output_schema = SCRResult
    else:
        output_schema = SynthesisResult

    # Select model based on complexity analysis recommendation
    recommended_model = _get_recommended_model(state)
    llm = get_llm_by_model_choice(recommended_model)
    structured_llm = llm.with_structured_output(output_schema, include_raw=True)
    messages.append(HumanMessage(content=prompt_content))
    response = structured_llm.invoke(messages)

    # Use helper to process retry logic
    def fallback(s):
        return {
            "synthesized_results": (
                "Failed to synthesize findings into structured format."
            )
        }

    retry_state = process_structured_response(response, state, fallback)
    if retry_state:
        # Check if we are clearing the state (success/fallback) or looping (error)
        # Note: process_structured_response returns {error: ...} for loop
        # Preserve conversation history on retry
        retry_state["synthesizer_messages"] = messages
        return retry_state

    # Success - Round 1: Initial synthesis
    result = response["parsed"]

    # Extract synthesis text based on schema type
    if use_scr:
        scr_result = result
        initial_synthesis = scr_result.to_formatted_report()
        print(f"  ✅ Initial SCR synthesis complete "
              f"(S: {len(scr_result.situation)}, "
              f"C: {len(scr_result.complication)}, "
              f"R: {len(scr_result.resolution)} chars)")
    else:
        initial_synthesis = result.summary
        scr_result = None
        print(f"  ✅ Initial synthesis complete ({len(initial_synthesis)} chars)")

    # Two-pass synthesis: Reflection and Refinement
    # Only perform reflection for non-incremental synthesis (first time)
    final_synthesis = initial_synthesis
    reflection_result = None

    if not is_incremental:
        # Perform reflection analysis
        print("\n🔍 [Reflection] Analyzing synthesis quality...")
        try:
            # Get recommended model from complexity analysis
            recommended_model = _get_recommended_model(state)
            reflection_llm_instance = get_llm_by_model_choice(recommended_model)
            reflection_llm = reflection_llm_instance.with_structured_output(
                ReflectionResult, include_raw=True
            )

            # Create a concise findings summary for reflection context
            findings_summary = "\n".join([
                f"- {f.get('task', 'Unknown')[:80]}: "
                f"{f.get('summary', 'No summary')[:200]}"
                for f in findings_to_process[:5]  # Limit to 5 for context
            ])
            if len(findings_to_process) > 5:
                findings_summary += (
                    f"\n... and {len(findings_to_process) - 5} more findings"
                )

            reflection_messages = [
                SystemMessage(content=REFLECTION_SYSTEM),
                HumanMessage(content=REFLECTION_MAIN.format(
                    query=query,
                    synthesis=initial_synthesis,
                    findings_summary=findings_summary
                ))
            ]

            reflection_response = reflection_llm.invoke(reflection_messages)

            # Process reflection response
            if reflection_response.get("parsing_error"):
                error_msg = reflection_response['parsing_error']
                print(f"  ⚠️  Reflection analysis failed: {error_msg}")
            else:
                reflection_result = reflection_response["parsed"]
                quality = reflection_result.overall_quality
                print(f"  ✅ Reflection complete - Quality: {quality}")
                if reflection_result.missing_core_insights:
                    count = len(reflection_result.missing_core_insights)
                    print(f"     Missing insights: {count}")
                if reflection_result.logic_issues:
                    count = len(reflection_result.logic_issues)
                    print(f"     Logic issues: {count}")

                # Refinement Round: Improve synthesis based on reflection
                if reflection_result.overall_quality in ['shallow', 'moderate']:
                    print(
                        "\n✨ [Refinement] Improving synthesis based on "
                        "reflection..."
                    )

                    # Format reflection analysis for prompt
                    quality = reflection_result.overall_quality
                    depth = reflection_result.depth_assessment

                    missing_insights = (
                        chr(10).join(
                            '- ' + insight
                            for insight in reflection_result.missing_core_insights
                        )
                        if reflection_result.missing_core_insights
                        else 'None identified'
                    )

                    logic_issues_text = (
                        chr(10).join(
                            '- ' + issue
                            for issue in reflection_result.logic_issues
                        )
                        if reflection_result.logic_issues
                        else 'None identified'
                    )

                    suggestions = (
                        chr(10).join(
                            '- ' + suggestion
                            for suggestion in reflection_result.improvement_suggestions
                        )
                        if reflection_result.improvement_suggestions
                        else 'None provided'
                    )

                    reflection_text = f"""Quality Assessment: {quality}

Depth Assessment:
{depth}

Missing Core Insights:
{missing_insights}

Logic Issues:
{logic_issues_text}

Improvement Suggestions:
{suggestions}
"""

                    # Use appropriate schema and prompt for refinement
                    if use_scr:
                        refine_schema = SCRResult
                        refine_system = SYNTHESIZER_SCR_SYSTEM
                        refine_prompt_template = SYNTHESIZER_SCR_REFINE
                    else:
                        refine_schema = SynthesisResult
                        refine_system = SYNTHESIZER_SYSTEM
                        refine_prompt_template = SYNTHESIZER_REFINE

                    # Use recommended model for refinement
                    recommended_model = _get_recommended_model(state)
                    refine_llm_instance = get_llm_by_model_choice(recommended_model)
                    refine_llm = refine_llm_instance.with_structured_output(
                        refine_schema, include_raw=True
                    )

                    refine_messages = [
                        SystemMessage(content=refine_system),
                        HumanMessage(
                            content=refine_prompt_template.format(
                                query=query,
                                initial_synthesis=initial_synthesis,
                                reflection_analysis=reflection_text,
                                findings=findings_text + metadata_context
                            )
                        )
                    ]

                    refine_response = refine_llm.invoke(refine_messages)

                    if refine_response.get("parsing_error"):
                        error_msg = refine_response['parsing_error']
                        print(f"  ⚠️  Refinement failed: {error_msg}")
                        print("  ⚠️  Using initial synthesis")
                    else:
                        refined_result = refine_response["parsed"]
                        if use_scr:
                            final_synthesis = refined_result.to_formatted_report()
                            s_len = len(refined_result.situation)
                            c_len = len(refined_result.complication)
                            r_len = len(refined_result.resolution)
                            print(
                                f"  ✅ Refinement complete "
                                f"(S: {s_len}, C: {c_len}, R: {r_len} chars)"
                            )
                        else:
                            final_synthesis = refined_result.summary
                            print(
                                f"  ✅ Refinement complete "
                                f"({len(final_synthesis)} chars)"
                            )
                else:
                    print(
                        "  ℹ️  Synthesis quality is already 'deep', "
                        "skipping refinement"
                    )
        except Exception as e:
            print(f"  ⚠️  Reflection/Refinement failed: {e}")
            print("  ⚠️  Using initial synthesis")
            # Continue with initial synthesis if reflection fails

    # Update conversation history with final synthesis
    updated_messages = messages + [AIMessage(content=final_synthesis)]

    # Track processed findings
    new_processed_ids = list(processed_finding_ids)
    for f in findings_to_process:
        f_hash = _compute_finding_hash(f)
        if f_hash not in new_processed_ids:
            new_processed_ids.append(f_hash)

    print(f"  ✅ Final synthesis complete ({len(final_synthesis)} chars)")

    # Store synthesis in long-term memory with conflict detection
    try:
        memory = TemporalMemory()

        # Search for similar existing memories
        similar_memories = memory.retrieve_memories(
            query=query,
            k=3,
            min_relevance=0.7,
            filter_by_tags=["synthesis"]
        )

        # Check for conflicts and supersede old memories
        superseded_ids = []
        for doc, relevance in similar_memories:
            # If very similar (high relevance), mark for superseding
            if relevance > 0.85:
                # Get memory ID from document metadata or try to extract from doc
                # ChromaDB stores IDs separately, so we need to get it from
                # the vector store. For now, we'll use a hash of the content
                # as identifier
                doc_id = doc.metadata.get("id") or doc.metadata.get("memory_id")
                if doc_id:
                    superseded_ids.append(doc_id)

        # Store with temporal tracking, automatically invalidating old versions
        memory_id = memory.store_memory_with_temporal(
            content=final_synthesis,
            metadata={
                "query": query,
                "iteration": iteration_count,
                "findings_count": len(findings),
            },
            priority=2.0,  # High priority for synthesis results
            tags=["synthesis", "research_result"],
            supersedes=superseded_ids if superseded_ids else None
        )

        if superseded_ids:
            print(f"  🔄 Updated {len(superseded_ids)} conflicting memories")
        else:
            print(f"  💾 Stored synthesis to long-term memory (ID: {memory_id[:8]}...)")
    except Exception as e:
        # Don't fail the node if memory storage fails
        print(f"  ⚠️  Failed to store in long-term memory: {e}")

    return {
        "synthesized_results": final_synthesis,
        "reflection_analysis": reflection_result,
        "error": None,
        "retry_count": 0,
        "synthesizer_messages": updated_messages,
        "processed_findings_ids": new_processed_ids
    }

