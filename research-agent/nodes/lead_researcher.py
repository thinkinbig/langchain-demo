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
    LEAD_RESEARCHER_INITIAL,
    LEAD_RESEARCHER_REFINE,
    LEAD_RESEARCHER_RETRY,
    LEAD_RESEARCHER_SYSTEM,
    STRATEGY_GENERATOR_MAIN,
    STRATEGY_GENERATOR_SYSTEM,
)
from schemas import (
    LeadResearcherState,
    ResearchStrategies,
    ResearchStrategy,
    ResearchTask,
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
    """LeadResearcher: Analyze query, create plan, generate subagent tasks
    
    Supports ToT mode: If selected_strategy_index is None and ToT mode is enabled,
    generates 3 strategies. Otherwise, uses selected strategy or generates tasks directly.
    """
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")
    existing_messages = state.get("lead_researcher_messages", [])
    rag_cache = state.get("rag_cache", {})
    sent_finding_hashes = set(state.get("sent_finding_hashes", []))

    # Check if we're in ToT mode and need to generate strategies
    selected_strategy_index = state.get("selected_strategy_index")
    strategies = state.get("strategies", [])

    # Check if we should use ToT mode
    # ToT mode is enabled when complexity is "complex" and first iteration
    complexity_analysis = state.get("complexity_analysis")
    complexity_level = None
    if complexity_analysis:
        if hasattr(complexity_analysis, "complexity_level"):
            complexity_level = complexity_analysis.complexity_level
        elif isinstance(complexity_analysis, dict):
            complexity_level = complexity_analysis.get("complexity_level")

    use_tot_mode = (
        complexity_level == "complex" and
        iteration_count == 0 and
        selected_strategy_index is None
    )

    # Determine if we need to generate strategies (first time in ToT mode)
    need_strategy_generation = (
        use_tot_mode and
        len(strategies) == 0
    )

    if need_strategy_generation:
        print("\n🌳 [LeadResearcher] ToT Mode: Generating 3 research strategies...")
        print(f"   Query: {query[:80]}...")
    else:
        print(f"\n🔍 [LeadResearcher] Analyzing query (iteration {iteration_count + 1})...")
        print(f"   Query: {query[:80]}...")
        if selected_strategy_index is not None:
            print(f"   📌 Using selected strategy {selected_strategy_index}")
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

    # Get complexity analysis for task creation guidance
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

    # Handle ToT mode: Generate strategies if needed
    if need_strategy_generation:
        # Generate 3 strategies using ToT approach
        system_prompt = context_manager.get_system_context(
            STRATEGY_GENERATOR_SYSTEM, include_knowledge=False
        )
        messages.append(SystemMessage(content=system_prompt))

        memory_ctx = (
            _retrieve_planning_history(query, k=3) if iteration_count > 0
            else "No previous planning history."
        )

        prompt_content = STRATEGY_GENERATOR_MAIN.format(
            query=query,
            complexity_info=complexity_info,
            memory_context=memory_ctx
        )

        # Add RAG context
        rag_instructions = (
            f"\n\n<internal_knowledge>\n{retrieved_context}\n</internal_knowledge>\n"
            "Note: Use the above internal knowledge if relevant to strategy generation."
        )
        messages.append(HumanMessage(content=prompt_content + rag_instructions))

        # Get recommended model
        recommended_model = "plus"  # Use plus for strategy generation
        if complexity_analysis:
            if hasattr(complexity_analysis, "recommended_model"):
                model = complexity_analysis.recommended_model
                if model in ["turbo", "plus"]:
                    recommended_model = model
            elif isinstance(complexity_analysis, dict):
                model = complexity_analysis.get("recommended_model", "plus")
                if model in ["turbo", "plus"]:
                    recommended_model = model

        # Invoke LLM to generate strategies
        llm = get_llm_by_model_choice(recommended_model)
        structured_llm = llm.with_structured_output(
            ResearchStrategies, include_raw=True
        )

        response = structured_llm.invoke(messages)

        # Process response
        def fallback_strategies(s):
            # Fallback: Create 3 simple strategies
            return {
                "strategies": [
                    ResearchStrategy(
                        strategy_id="strategy_1",
                        description="Comprehensive overview approach",
                        tasks=[
                            ResearchTask(
                                id="task_1",
                                description=f"Research overview: {s['query']}",
                                rationale="Fallback strategy"
                            )
                        ],
                        rationale="Fallback strategy"
                    ),
                    ResearchStrategy(
                        strategy_id="strategy_2",
                        description="Detailed analysis approach",
                        tasks=[
                            ResearchTask(
                                id="task_2",
                                description=f"Deep dive: {s['query']}",
                                rationale="Fallback strategy"
                            )
                        ],
                        rationale="Fallback strategy"
                    ),
                    ResearchStrategy(
                        strategy_id="strategy_3",
                        description="Comparative approach",
                        tasks=[
                            ResearchTask(
                                id="task_3",
                                description=f"Compare aspects: {s['query']}",
                                rationale="Fallback strategy"
                            )
                        ],
                        rationale="Fallback strategy"
                    )
                ],
                "scratchpad": "Fallback strategies due to error"
            }

        retry_state = process_structured_response(response, state, fallback_strategies)
        if retry_state:
            preserved_scratchpad = state.get("scratchpad", "")
            parsed_result = response.get("parsed")
            if parsed_result and hasattr(parsed_result, "scratchpad"):
                preserved_scratchpad = parsed_result.scratchpad
            retry_state["scratchpad"] = preserved_scratchpad
            retry_state["lead_researcher_messages"] = messages
            retry_state["rag_cache"] = rag_cache
            return retry_state

        # Success: Return strategies for evaluation
        parsed_result = response["parsed"]
        generated_strategies = parsed_result.strategies

        print(f"  ✅ Generated {len(generated_strategies)} strategies")
        for i, strategy in enumerate(generated_strategies):
            print(f"     Strategy {i}: {strategy.description[:60]}... ({len(strategy.tasks)} tasks)")

        return {
            "strategies": generated_strategies,
            "scratchpad": parsed_result.scratchpad[:200] if parsed_result.scratchpad else "",
            "lead_researcher_messages": messages + [AIMessage(content="Strategies generated")],
            "rag_cache": rag_cache,
            "error": None,
            "retry_count": 0
        }

    # Normal mode or using selected strategy
    if iteration_count == 0:
        # First iteration: Initialize conversation
        system_prompt = context_manager.get_system_context(
            LEAD_RESEARCHER_SYSTEM, include_knowledge=False
        )
        messages.append(SystemMessage(content=system_prompt))

        # Scratchpad: Keep only current iteration notes (simplified)
        # Historical info is retrieved from memory
        scratchpad = state.get("scratchpad", "")
        max_scratchpad_length = 200  # Keep it short, history in memory
        if len(scratchpad) > max_scratchpad_length:
            scratchpad = scratchpad[:max_scratchpad_length] + "..."

        memory_ctx = (
            memory_context if memory_context
            else "No previous planning history."
        )
        prompt_content = LEAD_RESEARCHER_INITIAL.format(
            query=query,
            complexity_info=complexity_info,
            scratchpad=scratchpad,
            memory_context=memory_ctx
        )

        # Add RAG context to initial prompt
        rag_instructions = (
            f"\n\n<internal_knowledge>\n{retrieved_context}\n</internal_knowledge>\n"
            "Note: Use the above internal knowledge if relevant to the task breakdown."
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
        prompt_content = LEAD_RESEARCHER_REFINE.format(
            query=query,
            findings_summary=findings_summary,
            citations_from_previous_round=citations_formatted,
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

    # Get recommended model from complexity analysis
    complexity_analysis = state.get("complexity_analysis")
    recommended_model = "plus"  # Default
    if complexity_analysis:
        if hasattr(complexity_analysis, "recommended_model"):
            model = complexity_analysis.recommended_model
            if model in ["turbo", "plus"]:
                recommended_model = model
        elif isinstance(complexity_analysis, dict):
            model = complexity_analysis.get("recommended_model", "plus")
            if model in ["turbo", "plus"]:
                recommended_model = model

    # If we have a selected strategy, use its tasks directly
    if selected_strategy_index is not None and len(strategies) > selected_strategy_index:
        selected_strategy = strategies[selected_strategy_index]
        tasks = selected_strategy.tasks
        print(f"  ✅ Using strategy {selected_strategy_index}: {selected_strategy.description[:60]}...")
        print(f"  ✅ Extracted {len(tasks)} tasks from selected strategy")

        plan = f"Research plan using strategy {selected_strategy_index}: {selected_strategy.description}\nTasks: {len(tasks)} sub-tasks"

        # Store planning memory
        if tasks and len(tasks) > 0:
            task_summary = "; ".join(
                [f"{t.id}: {t.description[:50]}" for t in tasks[:3]]
            )
            planning_content = (
                f"Iteration {iteration_count + 1}: Using strategy {selected_strategy_index}. "
                f"Key tasks: {task_summary}."
            )
            priority = 2.0 if iteration_count == 0 else 1.5
            memory_id = _store_planning_memory(
                query=query,
                content=planning_content,
                priority=priority,
                metadata={
                    "iteration": iteration_count + 1,
                    "task_count": len(tasks),
                    "strategy_index": selected_strategy_index
                }
            )
            if memory_id:
                print(f"  💾 Stored planning memory (ID: {memory_id[:8]}...)")

        # Update conversation history
        updated_messages = messages + [AIMessage(content=f"Using strategy {selected_strategy_index}")]

        # Track findings
        existing_findings = state.get("subagent_findings", [])
        new_sent_hashes = set(sent_finding_hashes)
        for f in existing_findings:
            f_hash = _compute_finding_hash(f)
            new_sent_hashes.add(f_hash)

        all_citations = state.get("all_extracted_citations", [])
        new_citation_count = len(all_citations)

        new_scratchpad = selected_strategy.rationale[:200] if selected_strategy.rationale else ""

        return {
            "research_plan": plan,
            "subagent_tasks": tasks,
            "iteration_count": iteration_count + 1,
            "error": None,
            "retry_count": 0,
            "scratchpad": new_scratchpad,
            "lead_researcher_messages": updated_messages,
            "rag_cache": rag_cache,
            "sent_finding_hashes": list(new_sent_hashes),
            "previous_citation_count": new_citation_count
        }

    # Invoke LLM with conversation history (normal mode)
    llm = get_llm_by_model_choice(recommended_model)
    structured_llm = llm.with_structured_output(
        ResearchTasks, include_raw=True
    )

    response = structured_llm.invoke(messages)

    # Use helper to process retry logic
    def fallback(s):
        return {
            "research_plan": f"Research plan for: {s['query']} (Fallback)",
            "subagent_tasks": [
                ResearchTask(
                    id="task_1",
                    description=f"Research: {s['query']}",
                    rationale="Fallback"
                ),
                ResearchTask(
                    id="task_2",
                    description=f"Info: {s['query']}",
                    rationale="Fallback"
                )
            ],
            "iteration_count": s.get("iteration_count", 0) + 1,
            "scratchpad": s.get("scratchpad", "Fallback due to error")
        }

    retry_state = process_structured_response(response, state, fallback)
    if retry_state:
        # If retry_state is returned, it means we either failed (and are looping)
        # or we hit max retries and are returning the fallback.
        # Preserve scratchpad: try to extract from parsed result,
        # otherwise keep existing
        preserved_scratchpad = state.get("scratchpad", "")
        parsed_result = response.get("parsed")
        if parsed_result and hasattr(parsed_result, "scratchpad"):
            # If LLM generated a scratchpad even with parsing error, use it
            preserved_scratchpad = parsed_result.scratchpad

        # Include scratchpad, conversation history, and cache in retry state
        retry_state["scratchpad"] = preserved_scratchpad
        retry_state["lead_researcher_messages"] = messages
        retry_state["rag_cache"] = rag_cache
        return retry_state

    # Success case
    parsed_result = response["parsed"]
    tasks = parsed_result.tasks

    plan = f"Research plan for: {query}\nTasks: {len(tasks)} sub-tasks"
    print(f"  ✅ Created {len(tasks)} sub-tasks")
    for i, task in enumerate(tasks, 1):
        print(f"     {i}. {task.description[:60]}... (ID: {task.id})")

    # Store important planning information to long-term memory
    # Only store if there's meaningful content (not just fallback)
    if tasks and len(tasks) > 0:
        task_summary = "; ".join(
            [f"{t.id}: {t.description[:50]}" for t in tasks[:3]]
        )
        plan_note = (
            parsed_result.scratchpad[:200]
            if parsed_result.scratchpad else 'N/A'
        )
        planning_content = (
            f"Iteration {iteration_count + 1}: Created {len(tasks)} tasks. "
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

    return {
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

