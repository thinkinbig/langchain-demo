"""Verifier node: Cross-check synthesized report against source evidence"""

import context_manager
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_verifier_llm
from memory_helpers import extract_evidence_summaries
from prompts import VERIFIER_MAIN, VERIFIER_SYSTEM
from schemas import ResearchState, VerificationResult


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
    # Use the first 100 chars of report as proxy query
    verification_query = report[:200]
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
    structured_llm = get_verifier_llm().with_structured_output(VerificationResult)

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
            return {
                "synthesized_results": response.corrected_report
            }

    except Exception as e:
        print(f"  ❌ Verification failed: {e}")
        return {}

