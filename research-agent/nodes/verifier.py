"""Verifier node: Cross-check synthesized report against source evidence"""

import context_manager
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from prompts import VERIFIER_MAIN, VERIFIER_SYSTEM
from schemas import ResearchState, VerificationResult, extract_evidence_summaries
from text_utils import clean_report_output


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

    # Invoke Verifier LLM
    prompt_content = VERIFIER_MAIN.format(
        report=report,
        evidence=evidence_text
    )

    # Use lead LLM (stronger model) for verification
    llm = get_llm_by_model_choice("plus")
    structured_llm = llm.with_structured_output(VerificationResult)

    try:
        response = structured_llm.invoke([
            SystemMessage(content=VERIFIER_SYSTEM),
            HumanMessage(content=prompt_content)
        ])

        if response.is_valid:
            print("  ✅ Report verified: Structure and facts appear accurate.")
            return {}  # No change
        else:
            corrections_count = len(response.corrections)
            print(f"  ⚠️  Issues found. {corrections_count} corrections applied.")
            # Clean corrected report to remove any XML tags
            cleaned_report = clean_report_output(response.corrected_report)
            return {
                "synthesized_results": cleaned_report
            }

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return {}

