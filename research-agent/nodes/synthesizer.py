"""Synthesizer node: Aggregate and synthesize all findings"""

import hashlib
from typing import Optional

from config import settings
from graph.utils import process_structured_response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from memory import create_findings_memory_manager
from memory.temporal_memory import TemporalMemory
from prompts import (
    REFLECTION_MAIN,
    REFLECTION_SYSTEM,
    SYNTHESIZER_MAIN,
    SYNTHESIZER_REFINE,
    SYNTHESIZER_RETRY,
    SYNTHESIZER_SCR_INCREMENTAL_STEP_1_SITUATION,
    SYNTHESIZER_SCR_INCREMENTAL_STEP_2_COMPLICATION,
    SYNTHESIZER_SCR_INCREMENTAL_STEP_3_RESOLUTION,
    SYNTHESIZER_SCR_REFINE_STEP_1_SITUATION,
    SYNTHESIZER_SCR_REFINE_STEP_2_COMPLICATION,
    SYNTHESIZER_SCR_REFINE_STEP_3_RESOLUTION,
    SYNTHESIZER_SCR_STEP_1_SITUATION,
    SYNTHESIZER_SCR_STEP_2_COMPLICATION,
    SYNTHESIZER_SCR_STEP_3_RESOLUTION,
    SYNTHESIZER_SCR_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from schemas import (
    ComplicationResult,
    ReflectionResult,
    ResolutionResult,
    SCRResult,
    SituationResult,
    SynthesisResult,
    SynthesizerState,
    extract_findings_metadata,
)
from text_utils import clean_report_output


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


def _generate_scr_chained(
    llm,
    query: str,
    findings_text: str,
    system_message: SystemMessage,
    retry_prompt: str | None = None,
    findings_memory_manager=None,
    stop_after_complication: bool = False
) -> SCRResult:
    """
    Generate SCR synthesis using 3-step chaining approach.

    Args:
        llm: LLM instance to use
        query: User query
        findings_text: Formatted findings text
        system_message: System message for the synthesizer
        retry_prompt: Optional retry prompt if previous attempt failed
        findings_memory_manager: Optional FindingsMemoryManager for semantic
                                 retrieval in Step 2/3
        stop_after_complication: If True, stop after Complication and return partial result

    Returns:
        SCRResult with all three sections (or partial if stop_after_complication=True)
    """
    # Step 1: Generate Situation (use full findings_text)
    print("  📝 [Step 1/3] Generating Situation section...")
    step1_prompt = SYNTHESIZER_SCR_STEP_1_SITUATION.format(
        query=query,
        findings=findings_text
    )
    if retry_prompt:
        step1_prompt = SYNTHESIZER_RETRY.format(
            previous_prompt=step1_prompt,
            error=retry_prompt
        )

    step1_llm = llm.with_structured_output(SituationResult, include_raw=True)
    step1_messages = [system_message, HumanMessage(content=step1_prompt)]
    step1_response = step1_llm.invoke(step1_messages)

    if step1_response.get("parsing_error"):
        raise ValueError(f"Step 1 (Situation) failed: {step1_response['parsing_error']}")

    situation_result = step1_response["parsed"]
    situation_draft = situation_result.situation
    print(f"    ✅ Situation complete ({len(situation_draft)} chars)")

    # Step 2: Generate Complication (with Situation context)
    # Use LangMem retrieval if available, otherwise fallback to full findings
    print("  📝 [Step 2/3] Generating Complication section...")
    step2_findings = findings_text
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context = f"{query}\n\nSituation: {situation_draft}"
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=6
        )
        if retrieved:
            step2_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step2_prompt = SYNTHESIZER_SCR_STEP_2_COMPLICATION.format(
        query=query,
        findings=step2_findings,
        situation_draft=situation_draft
    )

    step2_llm = llm.with_structured_output(ComplicationResult, include_raw=True)
    step2_messages = [system_message, HumanMessage(content=step2_prompt)]
    step2_response = step2_llm.invoke(step2_messages)

    if step2_response.get("parsing_error"):
        raise ValueError(f"Step 2 (Complication) failed: {step2_response['parsing_error']}")

    complication_result = step2_response["parsed"]
    complication_draft = complication_result.complication
    print(f"    ✅ Complication complete ({len(complication_draft)} chars)")

    # Early decision optimization: return partial result if enabled
    if stop_after_complication:
        print("  ⚡ Early decision: stopping after Complication, skipping Resolution")
        return SCRResult(
            situation=situation_draft,
            complication=complication_draft,
            resolution=""  # Empty resolution indicates partial synthesis
        )

    # Step 3: Generate Resolution (with Situation + Complication context)
    # Use LangMem retrieval if available, otherwise fallback to full findings
    print("  📝 [Step 3/3] Generating Resolution section...")
    step3_findings = findings_text
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context = (
            f"{query}\n\nSituation: {situation_draft}\n\n"
            f"Complication: {complication_draft}"
        )
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=6
        )
        if retrieved:
            step3_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step3_prompt = SYNTHESIZER_SCR_STEP_3_RESOLUTION.format(
        query=query,
        findings=step3_findings,
        situation_draft=situation_draft,
        complication_draft=complication_draft
    )

    step3_llm = llm.with_structured_output(ResolutionResult, include_raw=True)
    step3_messages = [system_message, HumanMessage(content=step3_prompt)]
    step3_response = step3_llm.invoke(step3_messages)

    if step3_response.get("parsing_error"):
        raise ValueError(f"Step 3 (Resolution) failed: {step3_response['parsing_error']}")

    resolution_result = step3_response["parsed"]
    resolution_draft = resolution_result.resolution
    print(f"    ✅ Resolution complete ({len(resolution_draft)} chars)")

    # Combine into SCRResult
    return SCRResult(
        situation=situation_draft,
        complication=complication_draft,
        resolution=resolution_draft
    )


def _generate_scr_chained_incremental(
    llm,
    query: str,
    previous_synthesis: str,
    new_findings: str,
    system_message: SystemMessage,
    findings_memory_manager=None
) -> SCRResult:
    """
    Generate incremental SCR synthesis using 3-step chaining approach.

    Args:
        llm: LLM instance to use
        query: User query
        previous_synthesis: Previous SCR synthesis (formatted report)
        new_findings: New findings to integrate
        system_message: System message for the synthesizer
        findings_memory_manager: Optional FindingsMemoryManager for semantic
                                retrieval

    Returns:
        SCRResult with updated sections
    """
    # Parse previous synthesis to extract sections
    # Previous synthesis is formatted as "# Situation\n\n...\n\n# Complication\n\n...\n\n# Resolution\n\n..."
    sections = previous_synthesis.split("# ")
    prev_situation = ""
    prev_complication = ""
    prev_resolution = ""

    for section in sections:
        if section.startswith("Situation"):
            prev_situation = section.replace("Situation\n\n", "").split("\n\n# ")[0].strip()
        elif section.startswith("Complication"):
            prev_complication = section.replace("Complication\n\n", "").split("\n\n# ")[0].strip()
        elif section.startswith("Resolution"):
            prev_resolution = section.replace("Resolution\n\n", "").strip()

    # Step 1: Update Situation
    # Use LangMem retrieval if available
    print("  📝 [Incremental Step 1/3] Updating Situation section...")
    step1_new_findings = new_findings
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context = f"{query}\n\nPrevious Situation: {prev_situation}\n\nNew findings: {new_findings[:200]}"
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=5
        )
        if retrieved:
            step1_new_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step1_prompt = SYNTHESIZER_SCR_INCREMENTAL_STEP_1_SITUATION.format(
        query=query,
        previous_situation=prev_situation,
        new_findings=step1_new_findings
    )

    step1_llm = llm.with_structured_output(SituationResult, include_raw=True)
    step1_messages = [system_message, HumanMessage(content=step1_prompt)]
    step1_response = step1_llm.invoke(step1_messages)

    if step1_response.get("parsing_error"):
        raise ValueError(f"Incremental Step 1 (Situation) failed: {step1_response['parsing_error']}")

    updated_situation = step1_response["parsed"].situation
    print(f"    ✅ Situation updated ({len(updated_situation)} chars)")

    # Step 2: Update Complication
    # Use LangMem retrieval if available
    print("  📝 [Incremental Step 2/3] Updating Complication section...")
    step2_new_findings = new_findings
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context = (
            f"{query}\n\nUpdated Situation: {updated_situation}\n\n"
            f"Previous Complication: {prev_complication}\n\n"
            f"New findings: {new_findings[:200]}"
        )
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=5
        )
        if retrieved:
            step2_new_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step2_prompt = SYNTHESIZER_SCR_INCREMENTAL_STEP_2_COMPLICATION.format(
        query=query,
        previous_complication=prev_complication,
        updated_situation=updated_situation,
        new_findings=step2_new_findings
    )

    step2_llm = llm.with_structured_output(ComplicationResult, include_raw=True)
    step2_messages = [system_message, HumanMessage(content=step2_prompt)]
    step2_response = step2_llm.invoke(step2_messages)

    if step2_response.get("parsing_error"):
        raise ValueError(f"Incremental Step 2 (Complication) failed: {step2_response['parsing_error']}")

    updated_complication = step2_response["parsed"].complication
    print(f"    ✅ Complication updated ({len(updated_complication)} chars)")

    # Step 3: Update Resolution
    # Use LangMem retrieval if available
    print("  📝 [Incremental Step 3/3] Updating Resolution section...")
    step3_new_findings = new_findings
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context = (
            f"{query}\n\nUpdated Situation: {updated_situation}\n\n"
            f"Updated Complication: {updated_complication}\n\n"
            f"Previous Resolution: {prev_resolution}\n\n"
            f"New findings: {new_findings[:200]}"
        )
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=5
        )
        if retrieved:
            step3_new_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step3_prompt = SYNTHESIZER_SCR_INCREMENTAL_STEP_3_RESOLUTION.format(
        query=query,
        previous_resolution=prev_resolution,
        updated_situation=updated_situation,
        updated_complication=updated_complication,
        new_findings=step3_new_findings
    )

    step3_llm = llm.with_structured_output(ResolutionResult, include_raw=True)
    step3_messages = [system_message, HumanMessage(content=step3_prompt)]
    step3_response = step3_llm.invoke(step3_messages)

    if step3_response.get("parsing_error"):
        raise ValueError(f"Incremental Step 3 (Resolution) failed: {step3_response['parsing_error']}")

    updated_resolution = step3_response["parsed"].resolution
    print(f"    ✅ Resolution updated ({len(updated_resolution)} chars)")

    return SCRResult(
        situation=updated_situation,
        complication=updated_complication,
        resolution=updated_resolution
    )


def _generate_scr_chained_refine(
    llm,
    query: str,
    initial_synthesis: str,
    reflection_analysis: str,
    findings_text: str,
    system_message: SystemMessage,
    findings_memory_manager=None,
    decision_reasoning: Optional[str] = None,
    decision_key_factors: Optional[list] = None
) -> SCRResult:
    """
    Refine SCR synthesis using 3-step chaining approach based on reflection feedback
    and decision reasoning from previous iteration.

    Args:
        llm: LLM instance to use
        query: User query
        initial_synthesis: Initial SCR synthesis (formatted report)
        reflection_analysis: Reflection analysis feedback
        findings_text: Formatted findings text
        system_message: System message for the synthesizer
        findings_memory_manager: Optional FindingsMemoryManager for semantic
                                retrieval
        decision_reasoning: Optional reasoning from decision node about why more
                          research was needed
        decision_key_factors: Optional list of key factors from decision node

    Returns:
        Refined SCRResult
    """

    # Parse initial synthesis to extract sections
    sections = initial_synthesis.split("# ")
    initial_situation = ""
    initial_complication = ""
    initial_resolution = ""

    for section in sections:
        if section.startswith("Situation"):
            initial_situation = section.replace("Situation\n\n", "").split("\n\n# ")[0].strip()
        elif section.startswith("Complication"):
            initial_complication = section.replace("Complication\n\n", "").split("\n\n# ")[0].strip()
        elif section.startswith("Resolution"):
            initial_resolution = section.replace("Resolution\n\n", "").strip()

    # Format decision context if available
    decision_context = ""
    decision_task_guidance = ""
    decision_guidance = ""
    decision_improvements = ""

    if decision_reasoning or decision_key_factors:
        decision_parts = []
        if decision_reasoning:
            decision_parts.append(f"<decision_reasoning>\n{decision_reasoning}\n</decision_reasoning>")
        if decision_key_factors:
            factors_text = "\n".join([f"- {factor}" for factor in decision_key_factors])
            decision_parts.append(f"<decision_key_factors>\n{factors_text}\n</decision_key_factors>")
        decision_context = "\n\n".join(decision_parts)
        decision_task_guidance = "Review the decision reasoning from the previous iteration to understand what gaps were identified."
        decision_guidance = " and decision reasoning from previous iteration"
        decision_improvements = "\n- Address specific gaps or areas identified in the decision reasoning."

    # Step 1: Refine Situation
    # Use LangMem retrieval if available
    print("  📝 [Refine Step 1/3] Refining Situation section...")
    step1_findings = findings_text
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context_parts = [
            f"{query}\n\nInitial Situation: {initial_situation}\n\n",
            f"Reflection Analysis: {reflection_analysis[:300]}"
        ]
        if decision_reasoning:
            query_context_parts.append(f"Decision Reasoning: {decision_reasoning[:200]}")
        query_context = "\n".join(query_context_parts)
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=6
        )
        if retrieved:
            step1_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step1_prompt = SYNTHESIZER_SCR_REFINE_STEP_1_SITUATION.format(
        query=query,
        initial_situation=initial_situation,
        reflection_analysis=reflection_analysis,
        decision_context=decision_context,
        decision_task_guidance=decision_task_guidance,
        decision_guidance=decision_guidance,
        decision_improvements=decision_improvements,
        findings=step1_findings
    )

    step1_llm = llm.with_structured_output(SituationResult, include_raw=True)
    step1_messages = [system_message, HumanMessage(content=step1_prompt)]
    step1_response = step1_llm.invoke(step1_messages)

    if step1_response.get("parsing_error"):
        raise ValueError(f"Refine Step 1 (Situation) failed: {step1_response['parsing_error']}")

    refined_situation = step1_response["parsed"].situation
    print(f"    ✅ Situation refined ({len(refined_situation)} chars)")

    # Step 2: Refine Complication
    # Use LangMem retrieval if available
    print("  📝 [Refine Step 2/3] Refining Complication section...")
    step2_findings = findings_text
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context_parts = [
            f"{query}\n\nRefined Situation: {refined_situation}\n\n",
            f"Initial Complication: {initial_complication}\n\n",
            f"Reflection Analysis: {reflection_analysis[:300]}"
        ]
        if decision_reasoning:
            query_context_parts.append(f"Decision Reasoning: {decision_reasoning[:200]}")
        query_context = "\n".join(query_context_parts)
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=6
        )
        if retrieved:
            step2_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step2_prompt = SYNTHESIZER_SCR_REFINE_STEP_2_COMPLICATION.format(
        query=query,
        initial_complication=initial_complication,
        refined_situation=refined_situation,
        reflection_analysis=reflection_analysis,
        decision_context=decision_context,
        decision_task_guidance=decision_task_guidance,
        decision_guidance=decision_guidance,
        decision_improvements=decision_improvements,
        findings=step2_findings
    )

    step2_llm = llm.with_structured_output(ComplicationResult, include_raw=True)
    step2_messages = [system_message, HumanMessage(content=step2_prompt)]
    step2_response = step2_llm.invoke(step2_messages)

    if step2_response.get("parsing_error"):
        raise ValueError(f"Refine Step 2 (Complication) failed: {step2_response['parsing_error']}")

    refined_complication = step2_response["parsed"].complication
    print(f"    ✅ Complication refined ({len(refined_complication)} chars)")

    # Step 3: Refine Resolution
    # Use LangMem retrieval if available
    print("  📝 [Refine Step 3/3] Refining Resolution section...")
    step3_findings = findings_text
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context_parts = [
            f"{query}\n\nRefined Situation: {refined_situation}\n\n",
            f"Refined Complication: {refined_complication}\n\n",
            f"Initial Resolution: {initial_resolution}\n\n",
            f"Reflection Analysis: {reflection_analysis[:300]}"
        ]
        if decision_reasoning:
            query_context_parts.append(f"Decision Reasoning: {decision_reasoning[:200]}")
        query_context = "\n".join(query_context_parts)
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=6
        )
        if retrieved:
            step3_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step3_prompt = SYNTHESIZER_SCR_REFINE_STEP_3_RESOLUTION.format(
        query=query,
        initial_resolution=initial_resolution,
        refined_situation=refined_situation,
        refined_complication=refined_complication,
        reflection_analysis=reflection_analysis,
        decision_context=decision_context,
        decision_task_guidance=decision_task_guidance,
        decision_guidance=decision_guidance,
        decision_improvements=decision_improvements,
        findings=step3_findings
    )

    step3_llm = llm.with_structured_output(ResolutionResult, include_raw=True)
    step3_messages = [system_message, HumanMessage(content=step3_prompt)]
    step3_response = step3_llm.invoke(step3_messages)

    if step3_response.get("parsing_error"):
        raise ValueError(f"Refine Step 3 (Resolution) failed: {step3_response['parsing_error']}")

    refined_resolution = step3_response["parsed"].resolution
    print(f"    ✅ Resolution refined ({len(refined_resolution)} chars)")

    return SCRResult(
        situation=refined_situation,
        complication=refined_complication,
        resolution=refined_resolution
    )


def _complete_partial_synthesis(
    llm,
    query: str,
    partial_synthesis: str,
    findings_text: str,
    system_message: SystemMessage,
    findings_memory_manager=None
) -> SCRResult:
    """
    Complete a partial synthesis (S+C) by generating the Resolution section.

    Args:
        llm: LLM instance to use
        query: User query
        partial_synthesis: Partial synthesis containing only Situation and Complication
        findings_text: Formatted findings text
        system_message: System message for the synthesizer
        findings_memory_manager: Optional FindingsMemoryManager for semantic retrieval

    Returns:
        Complete SCRResult with all three sections
    """
    # Parse partial synthesis to extract sections
    sections = partial_synthesis.split("# ")
    situation_draft = ""
    complication_draft = ""

    for section in sections:
        if section.startswith("Situation"):
            situation_draft = section.replace("Situation\n\n", "").split("\n\n# ")[0].strip()
        elif section.startswith("Complication"):
            complication_draft = section.replace("Complication\n\n", "").split("\n\n# ")[0].strip()

    if not situation_draft or not complication_draft:
        raise ValueError("Partial synthesis must contain both Situation and Complication sections")

    print("  📝 [Completing] Generating Resolution section for partial synthesis...")

    # Use LangMem retrieval if available
    step3_findings = findings_text
    if findings_memory_manager and findings_memory_manager.is_available():
        query_context = (
            f"{query}\n\nSituation: {situation_draft}\n\n"
            f"Complication: {complication_draft}"
        )
        retrieved = findings_memory_manager.retrieve_relevant_findings(
            query_context, top_k=6
        )
        if retrieved:
            step3_findings = retrieved
            print("    🔍 Using LangMem semantic retrieval for findings")

    step3_prompt = SYNTHESIZER_SCR_STEP_3_RESOLUTION.format(
        query=query,
        findings=step3_findings,
        situation_draft=situation_draft,
        complication_draft=complication_draft
    )

    step3_llm = llm.with_structured_output(ResolutionResult, include_raw=True)
    step3_messages = [system_message, HumanMessage(content=step3_prompt)]
    step3_response = step3_llm.invoke(step3_messages)

    if step3_response.get("parsing_error"):
        raise ValueError(f"Resolution completion failed: {step3_response['parsing_error']}")

    resolution_result = step3_response["parsed"]
    resolution_draft = resolution_result.resolution
    print(f"    ✅ Resolution complete ({len(resolution_draft)} chars)")

    return SCRResult(
        situation=situation_draft,
        complication=complication_draft,
        resolution=resolution_draft
    )


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

    # Get early decision settings from state or use defaults
    # Don't modify state directly, include in return dict if needed
    early_decision_enabled_default = settings.ENABLE_EARLY_DECISION
    early_decision_after_default = settings.EARLY_DECISION_AFTER

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

    # Initialize FindingsMemoryManager using Factory pattern
    # Check if manager already exists in state, otherwise create new one
    findings_memory_manager = state.get("_findings_memory_manager")
    if not findings_memory_manager:
        try:
            # Get thread_id from state if available (for namespace isolation)
            thread_id = state.get("_thread_id") or None
            # Use Factory to create manager with auto strategy selection
            findings_memory_manager = create_findings_memory_manager(
                thread_id=thread_id,
                strategy_type="auto"
            )
            # Store all findings (not just new ones) for semantic retrieval
            if findings:
                # Convert findings to dict format if needed
                findings_dicts = []
                for f in findings:
                    if isinstance(f, dict):
                        findings_dicts.append(f)
                    else:
                        # Convert Pydantic model to dict
                        findings_dicts.append(
                            f.model_dump() if hasattr(f, 'model_dump') else f.dict()
                        )

                if findings_dicts:
                    stored = findings_memory_manager.store_findings(findings_dicts)
                    if stored:
                        print("  💾 Stored findings for semantic retrieval")
                    else:
                        print("  ⚠️  Failed to store findings, using fallback")
                        findings_memory_manager = None
        except Exception as e:
            print(f"  ⚠️  Failed to initialize FindingsMemoryManager: {e}")
            findings_memory_manager = None

    # Build conversation history incrementally
    messages = list(existing_messages) if existing_messages else []

    # Default to incremental mode: use previous synthesis if available
    # If no previous synthesis, process all findings (first time)
    if previous_synthesis:
        # Incremental update: Only process new findings
        findings_to_process = new_findings
        is_incremental = True
    else:
        # First synthesis: Process all findings
        findings_to_process = findings
        is_incremental = False

    # Determine if we should use SCR structure (always support if enabled)
    use_scr = settings.USE_SCR_STRUCTURE

    if not messages:
        # Initialize conversation with appropriate system
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

    # Select model based on complexity analysis recommendation
    recommended_model = _get_recommended_model(state)
    llm = get_llm_by_model_choice(recommended_model)

    # Get system message
    system_message = messages[0] if messages and isinstance(messages[0], SystemMessage) else (
        SystemMessage(content=SYNTHESIZER_SCR_SYSTEM) if use_scr
        else SystemMessage(content=SYNTHESIZER_SYSTEM)
    )

    # Check for partial synthesis that needs completion
    # This happens when early decision was "finish research" but we only had S+C
    has_partial_synthesis = state.get("has_partial_synthesis", False)
    needs_completion = (
        has_partial_synthesis and
        previous_synthesis and
        not previous_synthesis.strip().endswith("# Resolution") and
        "# Resolution" not in previous_synthesis
    )

    # Check if early decision is enabled
    early_decision_enabled = state.get("early_decision_enabled", early_decision_enabled_default)
    early_decision_after = state.get("early_decision_after", early_decision_after_default)

    # Build prompt - use incremental update if previous synthesis exists
    if is_incremental and previous_synthesis:
        if use_scr:
            # Check if previous synthesis is partial (S+C only)
            is_previous_partial = (
                has_partial_synthesis and
                "# Resolution" not in previous_synthesis
            )

            # Use chained incremental update
            try:
                scr_result = _generate_scr_chained_incremental(
                    llm=llm,
                    query=query,
                    previous_synthesis=previous_synthesis,
                    new_findings=findings_text + metadata_context,
                    system_message=system_message,
                    findings_memory_manager=findings_memory_manager
                )
                initial_synthesis = scr_result.to_formatted_report()

                # If previous was partial and decision is to continue, keep it partial
                # (don't generate Resolution yet)
                if is_previous_partial and early_decision_enabled:
                    # Check if decision is to continue (needs_more_research)
                    needs_more = state.get("needs_more_research", False)
                    if needs_more:
                        # Decision is to continue, keep partial (skip Resolution)
                        print(f"  ✅ Incremental partial SCR synthesis complete "
                              f"(S: {len(scr_result.situation)}, "
                              f"C: {len(scr_result.complication)}, "
                              f"R: [skipped for early decision])")
                        # Mark as partial by clearing resolution
                        scr_result = SCRResult(
                            situation=scr_result.situation,
                            complication=scr_result.complication,
                            resolution=""
                        )
                        initial_synthesis = scr_result.to_formatted_report()
                    else:
                        # Decision is to finish, generate full synthesis
                        print(f"  ✅ Incremental SCR synthesis complete "
                              f"(S: {len(scr_result.situation)}, "
                              f"C: {len(scr_result.complication)}, "
                              f"R: {len(scr_result.resolution)} chars)")
                else:
                    print(f"  ✅ Incremental SCR synthesis complete "
                          f"(S: {len(scr_result.situation)}, "
                          f"C: {len(scr_result.complication)}, "
                          f"R: {len(scr_result.resolution)} chars)")
            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️  Chained incremental synthesis failed: {error_msg}")
                return {
                    "error": error_msg,
                    "retry_count": retry_count + 1,
                    "synthesizer_messages": messages
                }
        else:
            # Use non-SCR incremental update
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

            if last_error:
                prompt_content = SYNTHESIZER_RETRY.format(
                    previous_prompt=prompt_content,
                    error=last_error
                )

            output_schema = SynthesisResult
            structured_llm = llm.with_structured_output(output_schema, include_raw=True)
            messages.append(HumanMessage(content=prompt_content))
            response = structured_llm.invoke(messages)

            retry_state = process_structured_response(response, state)
            if retry_state:
                retry_state["synthesizer_messages"] = messages
                return retry_state

            result = response["parsed"]
            initial_synthesis = result.summary
            scr_result = None
            print(f"  ✅ Incremental synthesis complete ({len(initial_synthesis)} chars)")
    elif needs_completion and use_scr:
        # Complete partial synthesis (S+C -> S+C+R)
        try:
            scr_result = _complete_partial_synthesis(
                llm=llm,
                query=query,
                partial_synthesis=previous_synthesis,
                findings_text=findings_text + metadata_context,
                system_message=system_message,
                findings_memory_manager=findings_memory_manager
            )
            initial_synthesis = scr_result.to_formatted_report()
            print(f"  ✅ Completed partial synthesis "
                  f"(S: {len(scr_result.situation)}, "
                  f"C: {len(scr_result.complication)}, "
                  f"R: {len(scr_result.resolution)} chars)")
        except Exception as e:
            error_msg = str(e)
            print(f"  ⚠️  Partial synthesis completion failed: {error_msg}")
            return {
                "error": error_msg,
                "retry_count": retry_count + 1,
                "synthesizer_messages": messages
            }
    else:
        # First-time synthesis
        if use_scr:
            # Check if we should stop after complication for early decision
            stop_after_complication = (
                early_decision_enabled and
                early_decision_after == "complication" and
                not is_incremental
            )

            # Use chained synthesis
            try:
                scr_result = _generate_scr_chained(
                    llm=llm,
                    query=query,
                    findings_text=findings_text + metadata_context,
                    system_message=system_message,
                    retry_prompt=last_error,
                    findings_memory_manager=findings_memory_manager,
                    stop_after_complication=stop_after_complication
                )
                initial_synthesis = scr_result.to_formatted_report()

                # Check if this is a partial synthesis
                is_partial = not scr_result.resolution or scr_result.resolution.strip() == ""

                if is_partial:
                    print(f"  ✅ Initial partial SCR synthesis complete "
                          f"(S: {len(scr_result.situation)}, "
                          f"C: {len(scr_result.complication)}, "
                          f"R: [skipped for early decision])")
                else:
                    print(f"  ✅ Initial SCR synthesis complete "
                          f"(S: {len(scr_result.situation)}, "
                          f"C: {len(scr_result.complication)}, "
                          f"R: {len(scr_result.resolution)} chars)")
            except Exception as e:
                error_msg = str(e)
                print(f"  ⚠️  Chained synthesis failed: {error_msg}")
                return {
                    "error": error_msg,
                    "retry_count": retry_count + 1,
                    "synthesizer_messages": messages
                }
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

            output_schema = SynthesisResult
            structured_llm = llm.with_structured_output(output_schema, include_raw=True)
            messages.append(HumanMessage(content=prompt_content))
            response = structured_llm.invoke(messages)

            retry_state = process_structured_response(response, state)
            if retry_state:
                retry_state["synthesizer_messages"] = messages
                return retry_state

            result = response["parsed"]
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
                        # Use chained refinement
                        try:
                            # Use recommended model for refinement
                            recommended_model = _get_recommended_model(state)
                            refine_llm_instance = get_llm_by_model_choice(recommended_model)
                            refine_system = SystemMessage(content=SYNTHESIZER_SCR_SYSTEM)

                            # Get decision reasoning from state if available
                            decision_reasoning = state.get("decision_reasoning")
                            decision_key_factors = state.get("decision_key_factors", [])

                            refined_result = _generate_scr_chained_refine(
                                llm=refine_llm_instance,
                                query=query,
                                initial_synthesis=initial_synthesis,
                                reflection_analysis=reflection_text,
                                findings_text=findings_text + metadata_context,
                                system_message=refine_system,
                                findings_memory_manager=findings_memory_manager,
                                decision_reasoning=decision_reasoning,
                                decision_key_factors=decision_key_factors
                            )

                            final_synthesis = refined_result.to_formatted_report()
                            s_len = len(refined_result.situation)
                            c_len = len(refined_result.complication)
                            r_len = len(refined_result.resolution)
                            print(
                                f"  ✅ Refinement complete "
                                f"(S: {s_len}, C: {c_len}, R: {r_len} chars)"
                            )
                        except Exception as e:
                            error_msg = str(e)
                            print(f"  ⚠️  Chained refinement failed: {error_msg}")
                            print("  ⚠️  Using initial synthesis")
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

                        # Process response with retry logic
                        refine_retry_state = process_structured_response(
                            refine_response, state
                        )
                        if refine_retry_state:
                            # Retry needed, use initial synthesis
                            print("  ⚠️  Refinement failed, using initial synthesis")
                            refine_retry_state["synthesizer_messages"] = messages
                            return refine_retry_state

                        refined_result = refine_response["parsed"]
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

    # Clean final synthesis to remove any XML tags that might have been included
    cleaned_final_synthesis = clean_report_output(final_synthesis)

    # Check if this is a partial synthesis (missing Resolution)
    is_partial = (
        use_scr and
        "# Resolution" not in cleaned_final_synthesis and
        not cleaned_final_synthesis.strip().endswith("# Resolution")
    )

    # Prepare return state
    return_state = {
        "synthesized_results": cleaned_final_synthesis,
        "reflection_analysis": reflection_result,
        "error": None,
        "retry_count": 0,
        "synthesizer_messages": updated_messages,
        "processed_findings_ids": new_processed_ids,
        "has_partial_synthesis": is_partial,
        "partial_synthesis_done": is_partial,  # Mark that partial synthesis is complete
        "synthesis_mode": "partial" if is_partial else "full",
        # Ensure early decision settings are in state for next iteration
        "early_decision_enabled": early_decision_enabled,
        "early_decision_after": early_decision_after,
    }

    # Save findings_memory_manager to state for reuse in next iteration
    # Use underscore prefix as this is an internal implementation detail
    if findings_memory_manager:
        return_state["_findings_memory_manager"] = findings_memory_manager

    return return_state

