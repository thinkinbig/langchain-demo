"""Lead researcher node: Analyze query, create plan, generate subagent tasks"""

import hashlib

import context_manager
from citation_parser import format_citations_for_prompt
from graph.utils import process_structured_response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm.factory import get_lead_llm
from prompts import (
    LEAD_RESEARCHER_INITIAL,
    LEAD_RESEARCHER_REFINE,
    LEAD_RESEARCHER_RETRY,
    LEAD_RESEARCHER_SYSTEM,
)
from schemas import LeadResearcherState, ResearchTask, ResearchTasks


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

    if iteration_count == 0:
        # First iteration: Initialize conversation
        system_prompt = context_manager.get_system_context(
            LEAD_RESEARCHER_SYSTEM, include_knowledge=False
        )
        messages.append(SystemMessage(content=system_prompt))

        prompt_content = LEAD_RESEARCHER_INITIAL.format(
            query=query,
            scratchpad=state.get("scratchpad", "")
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

        prompt_content = LEAD_RESEARCHER_REFINE.format(
            query=query,
            findings_summary=findings_summary,
            citations_from_previous_round=citations_formatted,
            scratchpad=state.get("scratchpad", "")
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

    # Invoke LLM with conversation history
    structured_llm = get_lead_llm().with_structured_output(
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

    return {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "iteration_count": iteration_count + 1,
        "error": None,
        "retry_count": 0,
        "scratchpad": parsed_result.scratchpad,
        "lead_researcher_messages": updated_messages,
        "rag_cache": rag_cache,
        "sent_finding_hashes": list(new_sent_hashes),
        "previous_citation_count": new_citation_count
    }

