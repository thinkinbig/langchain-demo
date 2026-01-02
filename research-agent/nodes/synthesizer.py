"""Synthesizer node: Aggregate and synthesize all findings"""

from graph_helpers import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_lead_llm
from memory_helpers import extract_findings_metadata
from prompts import SYNTHESIZER_MAIN, SYNTHESIZER_RETRY, SYNTHESIZER_SYSTEM
from schemas import SynthesisResult, SynthesizerState


def synthesizer_node(state: SynthesizerState):
    """Synthesizer: Aggregate and synthesize all findings"""
    # Prefer filtered findings if available (Guardrail active)
    findings = state.get("filtered_findings", [])
    if not findings:
        # Fallback to raw findings
        findings = state.get("subagent_findings", [])
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")

    print(f"\n📊 [Synthesizer] Synthesizing {len(findings)} findings...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    if not findings:
        return {"synthesized_results": "No findings to synthesize."}

    # Pre-compute findings metadata (token optimization)
    findings_metadata = extract_findings_metadata(findings)

    # Format findings - only include essential information (already optimized)
    findings_text = "\n\n".join([
        f"{i+1}. {f.get('task', 'Unknown')[:60]}\n"
        f"{f.get('summary', 'No summary')}"
        for i, f in enumerate(findings)
    ])

    # Add metadata context to prompt (helps LLM understand scope)
    count = findings_metadata['count']
    total_sources = findings_metadata['total_sources']
    avg_length = findings_metadata['avg_summary_length']
    metadata_context = (
        f"\n\n[Metadata: {count} findings, "
        f"{total_sources} sources, "
        f"avg length: {avg_length} chars]"
    )

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
    response = structured_llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=prompt_content),
    ])

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
        return retry_state

    # Success
    result = response["parsed"]
    print(f"  ✅ Synthesis complete ({len(result.summary)} chars)")

    return {
        "synthesized_results": result.summary,
        "error": None,
        "retry_count": 0
    }

