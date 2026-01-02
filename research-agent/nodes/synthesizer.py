"""Synthesizer node: Aggregate and synthesize all findings"""

import hashlib

from graph.utils import process_structured_response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm.factory import get_lead_llm
from memory.temporal_memory import TemporalMemory
from prompts import SYNTHESIZER_MAIN, SYNTHESIZER_RETRY, SYNTHESIZER_SYSTEM
from schemas import (
    SynthesisResult,
    SynthesizerState,
    extract_findings_metadata,
)


def _compute_finding_hash(finding: dict) -> str:
    """Compute hash for a finding to track if it's been processed"""
    task = finding.get('task', '')
    summary = finding.get('summary', '')
    return hashlib.sha256((task + summary).encode()).hexdigest()[:16]


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

    if not messages:
        # First synthesis: Initialize conversation
        messages.append(SystemMessage(content=SYNTHESIZER_SYSTEM))
        findings_to_process = findings  # Process all on first run
        is_incremental = False
    else:
        # Subsequent synthesis: Only process new findings
        findings_to_process = new_findings
        is_incremental = True

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
        prompt_content = SYNTHESIZER_MAIN.format(
            query=query,
            findings=findings_text + metadata_context
        )

    if last_error:
        prompt_content = SYNTHESIZER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    structured_llm = get_lead_llm().with_structured_output(
        SynthesisResult, include_raw=True
    )
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

    # Success
    result = response["parsed"]
    new_synthesis = result.summary

    # Update conversation history with AI response
    updated_messages = messages + [AIMessage(content=new_synthesis)]

    # Track processed findings
    new_processed_ids = list(processed_finding_ids)
    for f in findings_to_process:
        f_hash = _compute_finding_hash(f)
        if f_hash not in new_processed_ids:
            new_processed_ids.append(f_hash)

    # If incremental, the result is already integrated
    # Otherwise, it's the full synthesis
    final_synthesis = new_synthesis

    print(f"  ✅ Synthesis complete ({len(final_synthesis)} chars)")

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
                # ChromaDB stores IDs separately, so we need to get it from the vector store
                # For now, we'll use a hash of the content as identifier
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
        "error": None,
        "retry_count": 0,
        "synthesizer_messages": updated_messages,
        "processed_findings_ids": new_processed_ids
    }

