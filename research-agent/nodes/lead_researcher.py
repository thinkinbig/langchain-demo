"""Lead researcher node: Analyze query, create plan, generate subagent tasks"""

import hashlib
from typing import Optional

import context_manager
from citation_parser import format_citations_for_prompt
from graph.utils import process_structured_response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm.factory import get_llm_by_model_choice
from memory.temporal_memory import TemporalMemory
from prompts import (
    LEAD_RESEARCHER_APPROACH_MAIN,
    LEAD_RESEARCHER_APPROACH_SYSTEM,
    LEAD_RESEARCHER_INITIAL,
    LEAD_RESEARCHER_REFINE,
    LEAD_RESEARCHER_RETRY,
    LEAD_RESEARCHER_SYSTEM,
    LEAD_RESEARCHER_TASK_GENERATION,
)
from schemas import (
    ApproachEvaluation,
    LeadResearcherState,
    ResearchTasks,
)


def _retrieve_planning_history(query: str, k: int = 5) -> str:
    """
    Retrieve relevant planning history from long-term memory.

    Args:
        query: Current research query
        k: Number of memories to retrieve

    Returns:
        Formatted string with planning history, or empty string if none found
    """
    try:
        memory = TemporalMemory()
        memories = memory.retrieve_valid_memories(
            query=query,
            k=k,
            min_relevance=0.6,
            filter_by_tags=["lead_researcher", "planning"],
            include_invalid=False
        )

        if not memories:
            return ""

        # Format memories for prompt
        history_parts = []
        for doc, relevance in memories[:k]:
            content = doc.page_content[:300]  # Limit each memory
            metadata = doc.metadata
            stored_at = metadata.get("stored_at", "unknown")
            history_parts.append(
                f"[{stored_at[:10]}] (relevance: {relevance:.2f})\n{content}"
            )

        return "\n\n".join(history_parts)
    except Exception as e:
        # Graceful fallback if memory retrieval fails
        print(f"  ⚠️  Memory retrieval failed: {e}")
        return ""


def _store_planning_memory(
    query: str,
    content: str,
    priority: float = 1.0,
    metadata: Optional[dict] = None
) -> Optional[str]:
    """
    Store important planning information to long-term memory.

    Args:
        query: Research query
        content: Content to store (should be concise, < 500 chars)
        priority: Priority score (1.0 = normal, 2.0 = important)
        metadata: Additional metadata

    Returns:
        Memory ID if successful, None otherwise
    """
    try:
        memory = TemporalMemory()
        memory_metadata = metadata or {}
        memory_metadata.update({
            "query": query,
            "type": "planning"
        })

        memory_id = memory.store_memory_with_temporal(
            content=content[:500],  # Limit content length
            metadata=memory_metadata,
            priority=priority,
            tags=["lead_researcher", "planning", "decision"]
        )
        return memory_id
    except Exception as e:
        # Graceful fallback if memory storage fails
        print(f"  ⚠️  Memory storage failed: {e}")
        return None


def _compute_finding_hash(finding: dict) -> str:
    """Compute hash for a finding to track if it's been processed"""
    task = finding.get('task', '')
    summary = finding.get('summary', '')
    return hashlib.sha256((task + summary).encode()).hexdigest()[:16]


def _get_new_findings(existing_findings: list, sent_finding_hashes: set) -> list:
    """Get findings that haven't been sent yet"""
    new_findings = []
    for f in existing_findings:
        f_hash = _compute_finding_hash(f)
        if f_hash not in sent_finding_hashes:
            new_findings.append(f)
    return new_findings


def lead_researcher_node(state: LeadResearcherState):
    """LeadResearcher: Analyze query, create plan, generate subagent tasks"""
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")
    existing_messages = state.get("lead_researcher_messages", [])
    rag_cache = state.get("rag_cache", {})
    sent_finding_hashes = set(state.get("sent_finding_hashes", []))

    print(f"\n🔍 [LeadResearcher] Analyzing query (iteration {iteration_count + 1})...")
    print(f"   Query: {query[:80]}...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    # RAG Caching: Check cache first
    if query in rag_cache:
        print("  🧠 [RAG] Using cached context for query...")
        retrieved_context = rag_cache[query]
    else:
        print(f"  🧠 [RAG] Retrieving context for: {query[:50]}...")
        retrieved_context, _ = context_manager.retrieve_knowledge(query, k=2)
        rag_cache[query] = retrieved_context

    # Build conversation history incrementally
    messages = list(existing_messages) if existing_messages else []

    # Retrieve planning history from long-term memory (for iterations > 0)
    memory_context = ""
    if iteration_count > 0:
        print("  🧠 [Memory] Retrieving planning history...")
        memory_context = _retrieve_planning_history(query, k=5)
        if memory_context:
            line_count = len(memory_context.split("\n"))
            print(f"  ✅ Retrieved {line_count} lines of planning history")
        else:
            print("  ℹ️  No relevant planning history found")

    # Get decision reasoning from previous iteration (if available)
    decision_reasoning = state.get("decision_reasoning")
    decision_key_factors = state.get("decision_key_factors", [])
    decision_guidance = ""
    if iteration_count > 0 and (decision_reasoning or decision_key_factors):
        decision_parts = []
        if decision_reasoning:
            decision_parts.append(f"Previous iteration identified gaps: {decision_reasoning[:300]}")
        if decision_key_factors:
            factors_text = ", ".join(decision_key_factors[:5])
            decision_parts.append(f"Key areas needing more research: {factors_text}")
        decision_guidance = "\n\n".join(decision_parts)
        print("  📋 Using decision guidance from previous iteration")

    # Get complexity analysis for task creation guidance
    complexity_analysis = state.get("complexity_analysis")
    complexity_info = ""
    if complexity_analysis:
        if hasattr(complexity_analysis, "complexity_level"):
            # Pydantic model
            recommended = complexity_analysis.recommended_workers
            complexity_info = (
                f"Complexity Level: {complexity_analysis.complexity_level}\n"
                f"Recommended Workers: {recommended}\n"
                f"Max Iterations: {complexity_analysis.max_iterations}\n"
                f"Rationale: {complexity_analysis.rationale}\n\n"
                f"Use this complexity assessment to guide task breakdown. "
                f"Aim to create approximately {recommended} "
                f"parallel research tasks that can be executed simultaneously."
            )
        elif isinstance(complexity_analysis, dict):
            # Dict format
            level = complexity_analysis.get('complexity_level', 'unknown')
            workers = complexity_analysis.get('recommended_workers', 2)
            iterations = complexity_analysis.get('max_iterations', 2)
            rationale = complexity_analysis.get('rationale', '')
            complexity_info = (
                f"Complexity Level: {level}\n"
                f"Recommended Workers: {workers}\n"
                f"Max Iterations: {iterations}\n"
                f"Rationale: {rationale}\n\n"
                f"Use this complexity assessment to guide task breakdown. "
                f"Aim to create approximately {workers} "
                f"parallel research tasks that can be executed simultaneously."
            )
        else:
            msg = (
                "No complexity analysis available. "
                "Use your judgment for task breakdown."
            )
            complexity_info = msg

    # Check if we need to do approach evaluation (first iteration only)
    selected_approach = state.get("selected_approach")
    selection_reasoning = state.get("selection_reasoning")
    need_approach_evaluation = (
        iteration_count == 0 and not selected_approach
    )

    if need_approach_evaluation:
        # Phase 1: Approach Evaluation
        print("  📋 [Phase 1] Evaluating research approaches...")

        system_prompt = context_manager.get_system_context(
            LEAD_RESEARCHER_APPROACH_SYSTEM, include_knowledge=False
        )
        approach_messages = [SystemMessage(content=system_prompt)]

        memory_ctx = (
            memory_context if memory_context
            else "No previous planning history."
        )
        scratchpad = state.get("scratchpad", "")
        max_scratchpad_length = 200
        if len(scratchpad) > max_scratchpad_length:
            scratchpad = scratchpad[:max_scratchpad_length] + "..."

        approach_prompt = LEAD_RESEARCHER_APPROACH_MAIN.format(
            query=query,
            complexity_info=complexity_info,
            memory_context=memory_ctx,
            internal_knowledge=retrieved_context
        )
        approach_messages.append(HumanMessage(content=approach_prompt))

        # Get recommended model
        from config import settings
        complexity_analysis = state.get("complexity_analysis")
        recommended_model = "plus"  # Default
        if complexity_analysis:
            if hasattr(complexity_analysis, "recommended_model"):
                model = complexity_analysis.recommended_model
                if model in ["turbo", "plus", "max"]:
                    if model == "max" and not settings.ENABLE_MAX_MODEL:
                        recommended_model = "plus"
                    else:
                        recommended_model = model
            elif isinstance(complexity_analysis, dict):
                model = complexity_analysis.get("recommended_model", "plus")
                if model in ["turbo", "plus", "max"]:
                    if model == "max" and not settings.ENABLE_MAX_MODEL:
                        recommended_model = "plus"
                    else:
                        recommended_model = model

        # Invoke LLM for approach evaluation
        llm = get_llm_by_model_choice(recommended_model)
        approach_llm = llm.with_structured_output(
            ApproachEvaluation, include_raw=True
        )

        approach_response = approach_llm.invoke(approach_messages)

        # Process approach evaluation response
        approach_retry_state = process_structured_response(
            approach_response, state
        )
        if approach_retry_state:
            # Retry needed, return state update
            return approach_retry_state

        approach_eval = approach_response["parsed"]

        # Display approach evaluation results
        print("\n  📊 [Approach Evaluation Results]")
        for i, approach in enumerate(approach_eval.approaches):
            is_selected = i == approach_eval.selected_approach_index
            marker = "✅ SELECTED" if is_selected else "  "
            print(f"\n     {marker} Approach {i + 1}:")
            print(f"        Description: {approach.description[:100]}...")
            if approach.advantages:
                adv_str = ', '.join(approach.advantages[:3])
                print(f"        Advantages: {adv_str}")
            if approach.disadvantages:
                dis_str = ', '.join(approach.disadvantages[:3])
                print(f"        Disadvantages: {dis_str}")
            if approach.suitability:
                print(f"        Suitability: {approach.suitability[:80]}...")

        selected_idx = approach_eval.selected_approach_index
        selected_approach_obj = approach_eval.approaches[selected_idx]
        print(f"\n     Selected: Approach {selected_idx + 1}")
        print(f"     Reasoning: {approach_eval.selection_reasoning[:150]}...")

        # Store selected approach for Phase 2
        selected_approach = selected_approach_obj.description
        selection_reasoning = approach_eval.selection_reasoning

        # Now proceed to Phase 2: Task Generation
        print("\n  📝 [Phase 2] Generating tasks based on selected approach...")

    if iteration_count == 0:
        # Phase 2: Task Generation (first iteration, after approach evaluation)
        system_prompt = context_manager.get_system_context(
            LEAD_RESEARCHER_SYSTEM, include_knowledge=False
        )
        messages.append(SystemMessage(content=system_prompt))

        scratchpad = state.get("scratchpad", "")
        max_scratchpad_length = 200
        if len(scratchpad) > max_scratchpad_length:
            scratchpad = scratchpad[:max_scratchpad_length] + "..."

        memory_ctx = (
            memory_context if memory_context
            else "No previous planning history."
        )

        if need_approach_evaluation or selected_approach:
            # Get graph context for task generation (if GraphRAG enabled)
            graph_context = ""
            try:
                from config import settings
                if settings.GRAPH_ENABLED:
                    from memory.graph_rag import GraphRAGManager
                    graph_rag_manager = GraphRAGManager()
                    graph_context = graph_rag_manager.retrieve_with_ppr(
                        query,
                        top_k_nodes=getattr(settings, 'PPR_TOP_K_NODES', 20),
                        top_k_docs=getattr(settings, 'PPR_TOP_K_DOCS', 10),
                        alpha=getattr(settings, 'PPR_ALPHA', 0.85)
                    )
            except Exception:
                # Graceful degradation: continue without graph context
                pass

            # Use task generation prompt with selected approach
            task_prompt = LEAD_RESEARCHER_TASK_GENERATION.format(
                query=query,
                selected_approach=selected_approach or "Standard research approach",
                selection_reasoning=selection_reasoning or "Based on query analysis",
                complexity_info=complexity_info,
                memory_context=memory_ctx,
                internal_knowledge=retrieved_context,
                graph_context=graph_context or "(No graph context available)"
            )
            messages.append(HumanMessage(content=task_prompt))
        else:
            # Fallback to original initial prompt
            prompt_content = LEAD_RESEARCHER_INITIAL.format(
                query=query,
                complexity_info=complexity_info,
                scratchpad=scratchpad,
                memory_context=memory_ctx
            )
            rag_instructions = (
                f"\n\n<internal_knowledge>\n{retrieved_context}\n"
                "</internal_knowledge>\n"
                "Note: Use the above internal knowledge if relevant "
                "to the task breakdown."
            )
            messages.append(HumanMessage(content=prompt_content + rag_instructions))
    else:
        # Subsequent iterations: Only send new information
        existing_findings = state.get("subagent_findings", [])

        # Identify new findings (not yet sent)
        new_findings = _get_new_findings(existing_findings, sent_finding_hashes)

        # Build findings summary only for new findings
        if new_findings:
            findings_summary = "\n".join([
                f"- {f.get('task', 'Unknown')[:50]}: {f.get('summary', '')[:80]}"
                for f in new_findings[:3]
            ])
        else:
            findings_summary = "(No new findings since last iteration)"

        # Get new citations (compare with previous)
        all_citations = state.get("all_extracted_citations", [])
        previous_citation_count = state.get("previous_citation_count", 0)
        new_citations = all_citations[previous_citation_count:]
        citations_formatted = (
            format_citations_for_prompt(new_citations) if new_citations else ""
        )

        # Scratchpad: Keep only current iteration notes (simplified)
        scratchpad = state.get("scratchpad", "")
        max_scratchpad_length = 200  # Keep it short, history in memory
        if len(scratchpad) > max_scratchpad_length:
            scratchpad = scratchpad[:max_scratchpad_length] + "..."

        memory_ctx = (
            memory_context if memory_context
            else "No previous planning history."
        )

        # Format decision guidance for prompt
        decision_context = decision_guidance if decision_guidance else ""
        decision_task_guidance = ""
        if decision_key_factors:
            decision_task_guidance = (
                f"Pay special attention to the key areas identified in the decision "
                f"reasoning: {', '.join(decision_key_factors[:3])}. "
                "Focus new tasks on addressing these specific gaps."
            )
        else:
            decision_task_guidance = ""

        prompt_content = LEAD_RESEARCHER_REFINE.format(
            query=query,
            findings_summary=findings_summary,
            citations_from_previous_round=citations_formatted,
            decision_guidance=decision_context,
            decision_task_guidance=decision_task_guidance,
            scratchpad=scratchpad,
            memory_context=memory_ctx
        )

        # Add RAG context (cached, so minimal cost)
        rag_instructions = (
            f"\n\n<internal_knowledge>\n{retrieved_context}\n</internal_knowledge>\n"
            "Note: Use the above internal knowledge if relevant to the task breakdown."
        )
        messages.append(HumanMessage(content=prompt_content + rag_instructions))

    # Add feedback if retrying - Standard Pattern
    if last_error:
        retry_prompt = LEAD_RESEARCHER_RETRY.format(
            previous_prompt=messages[-1].content if messages else "",
            error=last_error
        )
        # Replace last human message with retry version
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(content=retry_prompt)
        else:
            messages.append(HumanMessage(content=retry_prompt))

    # Get recommended model from complexity analysis (if not already done in Phase 1)
    if not need_approach_evaluation:
        from config import settings

        complexity_analysis = state.get("complexity_analysis")
        recommended_model = "plus"  # Default
        if complexity_analysis:
            if hasattr(complexity_analysis, "recommended_model"):
                model = complexity_analysis.recommended_model
                if model in ["turbo", "plus", "max"]:
                    # Downgrade max to plus if not enabled
                    if model == "max" and not settings.ENABLE_MAX_MODEL:
                        recommended_model = "plus"
                    else:
                        recommended_model = model
            elif isinstance(complexity_analysis, dict):
                model = complexity_analysis.get("recommended_model", "plus")
                if model in ["turbo", "plus", "max"]:
                    # Downgrade max to plus if not enabled
                    if model == "max" and not settings.ENABLE_MAX_MODEL:
                        recommended_model = "plus"
                    else:
                        recommended_model = model

    # Invoke LLM with conversation history for task generation
    llm = get_llm_by_model_choice(recommended_model)
    structured_llm = llm.with_structured_output(
        ResearchTasks, include_raw=True
    )

    response = structured_llm.invoke(messages)

    # Use helper to process retry logic
    retry_state = process_structured_response(response, state)
    if retry_state:
        # Retry needed, preserve state and return for retry
        preserved_scratchpad = state.get("scratchpad", "")
        parsed_result = response.get("parsed")
        if parsed_result and hasattr(parsed_result, "scratchpad"):
            # If LLM generated a scratchpad even with parsing error, use it
            preserved_scratchpad = parsed_result.scratchpad

        # Include scratchpad, conversation history, and cache in retry state
        retry_state["scratchpad"] = preserved_scratchpad
        retry_state["lead_researcher_messages"] = messages
        retry_state["rag_cache"] = rag_cache
        # Preserve selected approach if we're in Phase 2
        if selected_approach:
            retry_state["selected_approach"] = selected_approach
            retry_state["selection_reasoning"] = selection_reasoning
        return retry_state

    # Success case
    parsed_result = response["parsed"]
    tasks = parsed_result.tasks

    plan = f"Research plan for: {query}\nTasks: {len(tasks)} sub-tasks"
    if need_approach_evaluation:
        print(f"\n  ✅ Created {len(tasks)} sub-tasks based on selected approach")
    else:
        print(f"  ✅ Created {len(tasks)} sub-tasks")
    for i, task in enumerate(tasks, 1):
        print(f"     {i}. {task.description[:60]}... (ID: {task.id})")

    # Store important planning information to long-term memory
    if tasks and len(tasks) > 0:
        task_summary = "; ".join(
            [f"{t.id}: {t.description[:50]}" for t in tasks[:3]]
        )
        plan_note = (
            parsed_result.scratchpad[:200]
            if parsed_result.scratchpad else 'N/A'
        )
        approach_info = ""
        if selected_approach:
            approach_info = f"Selected Approach: {selected_approach[:100]}. "
        planning_content = (
            f"Iteration {iteration_count + 1}: {approach_info}"
            f"Created {len(tasks)} tasks. "
            f"Key tasks: {task_summary}. "
            f"Plan: {plan_note}"
        )
        # First iteration is more important
        priority = 2.0 if iteration_count == 0 else 1.5
        memory_id = _store_planning_memory(
            query=query,
            content=planning_content,
            priority=priority,
            metadata={
                "iteration": iteration_count + 1,
                "task_count": len(tasks),
                "task_ids": ",".join([t.id for t in tasks])
            }
        )
        if memory_id:
            print(f"  💾 Stored planning memory (ID: {memory_id[:8]}...)")

    # Update conversation history with AI response
    # Note: We don't store the full AI response, just track that we got one
    # The actual response is in parsed_result
    updated_messages = messages + [AIMessage(content="Tasks generated successfully")]

    # Track which findings we've sent
    existing_findings = state.get("subagent_findings", [])
    new_sent_hashes = set(sent_finding_hashes)
    for f in existing_findings:
        f_hash = _compute_finding_hash(f)
        new_sent_hashes.add(f_hash)

    # Track citation count for next iteration
    all_citations = state.get("all_extracted_citations", [])
    new_citation_count = len(all_citations)

    # Simplify scratchpad: Keep only current iteration's key notes
    # Historical info is in memory, so scratchpad stays short
    new_scratchpad = (
        parsed_result.scratchpad[:200]
        if parsed_result.scratchpad else ""
    )

    result = {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "iteration_count": iteration_count + 1,
        "error": None,
        "retry_count": 0,
        "scratchpad": new_scratchpad,  # Simplified, history in memory
        "lead_researcher_messages": updated_messages,
        "rag_cache": rag_cache,
        "sent_finding_hashes": list(new_sent_hashes),
        "previous_citation_count": new_citation_count
    }

    # Preserve selected approach for future reference
    if selected_approach:
        result["selected_approach"] = selected_approach
        result["selection_reasoning"] = selection_reasoning

    return result

