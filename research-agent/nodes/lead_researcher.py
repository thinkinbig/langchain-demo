"""Lead researcher node: Analyze query, create plan, generate subagent tasks"""

import context_manager
from citation_parser import format_citations_for_prompt
from graph.utils import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_lead_llm
from prompts import (
    LEAD_RESEARCHER_INITIAL,
    LEAD_RESEARCHER_REFINE,
    LEAD_RESEARCHER_RETRY,
    LEAD_RESEARCHER_SYSTEM,
)
from schemas import LeadResearcherState, ResearchTask, ResearchTasks


def lead_researcher_node(state: LeadResearcherState):
    """LeadResearcher: Analyze query, create plan, generate subagent tasks"""
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")

    print(f"\n🔍 [LeadResearcher] Analyzing query (iteration {iteration_count + 1})...")
    print(f"   Query: {query[:80]}...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    if iteration_count == 0:
        prompt_content = LEAD_RESEARCHER_INITIAL.format(
            query=query,
            scratchpad=state.get("scratchpad", "")
        )
    else:
        # REFINE mode - inject citations
        existing_findings = state.get("subagent_findings", [])
        findings_summary = "\n".join([
            f"- {f.get('task', 'Unknown')[:50]}: {f.get('summary', '')[:80]}"
            for f in existing_findings[:3]
        ])

        # Format extracted citations for prompt
        citations = state.get("all_extracted_citations", [])
        citations_formatted = format_citations_for_prompt(citations)

        prompt_content = LEAD_RESEARCHER_REFINE.format(
            query=query,
            findings_summary=findings_summary,
            citations_from_previous_round=citations_formatted,
            scratchpad=state.get("scratchpad", "")
        )

    # Add feedback if retrying - Standard Pattern
    if last_error:
        prompt_content = LEAD_RESEARCHER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    # Invoke LLM
    structured_llm = get_lead_llm().with_structured_output(
        ResearchTasks, include_raw=True
    )

    # RAG: Retrieve context dynamically (Limited to k=2 for planner efficiency)
    print(f"  🧠 [RAG] Retrieving context for: {query[:50]}...")
    retrieved_context, _ = context_manager.retrieve_knowledge(query, k=2)

    # Inject retrieved context into prompt
    rag_instructions = (
        f"\n\n<internal_knowledge>\n{retrieved_context}\n</internal_knowledge>\n"
        "Note: Use the above internal knowledge if relevant to the task breakdown."
    )

    # Invoke LLM without static knowledge base in system prompt
    system_prompt = context_manager.get_system_context(
        LEAD_RESEARCHER_SYSTEM, include_knowledge=False
    )

    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content + rag_instructions)
    ])

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
        # In both cases, we return deeply.
        return retry_state

    # Success case
    parsed_result = response["parsed"]
    tasks = parsed_result.tasks

    plan = f"Research plan for: {query}\nTasks: {len(tasks)} sub-tasks"
    print(f"  ✅ Created {len(tasks)} sub-tasks")
    for i, task in enumerate(tasks, 1):
        print(f"     {i}. {task.description[:60]}... (ID: {task.id})")

    return {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "iteration_count": iteration_count + 1,
        "error": None,
        "retry_count": 0,
        "scratchpad": parsed_result.scratchpad
    }

