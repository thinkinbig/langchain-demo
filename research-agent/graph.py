"""Multi-agent research system graph"""

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from schemas import (
    Citation,
    CitationAgentState,
    DecisionState,
    Finding,
    LeadResearcherState,
    ResearchState,
    ResearchTasks,
    SubagentOutput,
    SubagentState,
    SynthesisResult,
    SynthesizerState,
)
import tools

# LLM Configuration - Lazy Loading Pattern
# LLMs are initialized on first use, not at module import time
# This allows tests to mock LLMs without needing API keys

_lead_llm = None
_subagent_llm = None

# Import tracing for local logging (optional)
try:
    from tracing import get_callbacks
except ImportError:
    def get_callbacks():
        return []


def get_lead_llm():
    """Get or create the lead LLM instance (lazy loading)"""
    global _lead_llm
    if _lead_llm is None:
        _lead_llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.3,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _lead_llm


def get_subagent_llm():
    """Get or create the subagent LLM instance (lazy loading)"""
    global _subagent_llm
    if _subagent_llm is None:
        _subagent_llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
            temperature=0.3,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _subagent_llm


def process_structured_response(response, state, fallback_func=None):
    """
    Standardized validation and state update for retry loop.
    
    Args:
        response: Output from .with_structured_output(..., include_raw=True)
        state: Current state dict
        fallback_func: Optional callable to generate fallback state if max retries reached
    
    Returns:
        dict: State update
    """
    retry_count = state.get("retry_count", 0)
    parsing_error = response.get("parsing_error")
    
    if parsing_error:
        print(f"  ❌ Validation failed: {parsing_error}")
        
        # Check max retries (e.g., 3 attempts total: 0, 1, 2)
        if retry_count >= 2:
            print(f"  ❌ Max retries ({retry_count + 1}) reached")
            if fallback_func:
                print("  ⚠️  Using fallback logic")
                fallback_update = fallback_func(state)
                # Ensure fallback clears the error state
                return {**fallback_update, "error": None, "retry_count": 0}
            
            # If no fallback, just propagate the error or decide to end
            return {"error": str(parsing_error), "retry_count": retry_count + 1}
        
        # Route back for retry
        return {
            "error": str(parsing_error),
            "retry_count": retry_count + 1
        }
    
    # Success
    return None  # Signal to caller that retrieval was successful, caller handles "parsed"


def should_retry(state: ResearchState) -> bool:
    """Common condition to check if we should loop back"""
    return bool(state.get("error"))


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

    # Optimized prompt for query analysis and task generation
    from prompts import (
        LEAD_RESEARCHER_INITIAL, 
        LEAD_RESEARCHER_REFINE, 
        LEAD_RESEARCHER_RETRY, 
        LEAD_RESEARCHER_SYSTEM
    )

    if iteration_count == 0:
        prompt_content = LEAD_RESEARCHER_INITIAL.format(query=query)
    else:
        existing_findings = state.get("subagent_findings", [])
        findings_summary = "\n".join([
            f"- {f.get('task', 'Unknown')[:50]}: {f.get('summary', '')[:80]}"
            for f in existing_findings[:3]
        ])
        prompt_content = LEAD_RESEARCHER_REFINE.format(
            query=query, 
            findings_summary=findings_summary
        )

    # Add feedback if retrying - Standard Pattern
    if last_error:
        prompt_content = LEAD_RESEARCHER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    # Invoke LLM
    structured_llm = get_lead_llm().with_structured_output(ResearchTasks, include_raw=True)
    response = structured_llm.invoke([
        SystemMessage(content=LEAD_RESEARCHER_SYSTEM),
        HumanMessage(content=prompt_content)
    ])
    
    # Use helper to process retry logic
    def fallback(s):
        return {
            "research_plan": f"Research plan for: {s['query']} (Fallback)",
            "subagent_tasks": [f"Research: {s['query']}", f"Info: {s['query']}"],
            "iteration_count": s.get("iteration_count", 0) + 1
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
        print(f"     {i}. {str(task)[:60]}...")

    return {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "iteration_count": iteration_count + 1,
        "error": None,
        "retry_count": 0,
    }


def subagent_node(state: SubagentState):
    """Subagent: Perform web search and analyze results"""
    from prompts import (
        SUBAGENT_ANALYSIS, 
        SUBAGENT_RETRY, 
        SUBAGENT_SYSTEM
    )
    
    tasks = state.get("subagent_tasks", [])
    if not tasks:
        return {}

    # Get the first task (each worker gets one task via Send)
    task = tasks[0]
    task = str(task)

    print(f"\n🔎 [Subagent] Researching: {task[:60]}...")

    # Perform web search
    search_results = tools.search_web(task, max_results=5)

    if not search_results:
        print("  ⚠️  No search results found")
        finding = Finding(
            task=task,
            summary="No information found",
            sources=[],
        )
        return {"subagent_findings": [finding]}

    # Analyze and summarize findings - optimize content length
    sources_text = "\n\n".join([
        f"{i+1}. {r['title']}\n{r['content'][:150]}"
        for i, r in enumerate(search_results)
    ])

    analysis_prompt_content = SUBAGENT_ANALYSIS.format(
        task=task,
        results=sources_text
    )
    
    # Internal retry loop for parallel subagent
    structured_llm = get_subagent_llm().with_structured_output(SubagentOutput, include_raw=True)
    current_prompt = analysis_prompt_content
    
    for attempt in range(3):
        response = structured_llm.invoke([
            SystemMessage(content=SUBAGENT_SYSTEM),
            HumanMessage(content=current_prompt),
        ])
        
        if not response.get("parsing_error"):
            # Success
            output = response["parsed"]
            # Ensure sources are preserved from search
            sources_list = [
                 {"title": r["title"], "url": r["url"]}
                 for r in search_results
            ]
            finding = Finding(
                task=task,
                summary=output.summary,
                sources=sources_list
            )
            
            print(f"  ✅ Found {len(sources_list)} sources, summary created")
            return {"subagent_findings": [finding]}
            
        # Failure
        error = response["parsing_error"]
        print(f"  ⚠️  Subagent validation failed (attempt {attempt+1}/3): {error}")
        
        if attempt < 2:
            current_prompt = SUBAGENT_RETRY.format(
                previous_prompt=analysis_prompt_content,
                error=error
            )

    # Fallback if all retries fail
    print("  ❌ All subagent retries failed, using raw fallback")
    finding = Finding(
        task=task,
        summary="Failed to generate structured summary.",
        sources=[{"title": r["title"], "url": r["url"]} for r in search_results]
    )
    return {"subagent_findings": [finding]}


def assign_subagents(state: ResearchState):
    """Fan-out: Assign each task to a subagent OR retry"""
    # Check for retry condition
    if should_retry(state):
        print(f"  ↺ Routing back to lead_researcher for retry...")
        return "lead_researcher"

    tasks = state.get("subagent_tasks", [])
    print(f"\n🔀 [Fan-out] Distributing {len(tasks)} tasks to subagents...")

    # Ensure all tasks are strings before sending
    return [
        Send(
            "subagent",
            {
                "subagent_tasks": [
                    str(task) if not isinstance(task, str) else task
                ]
            },
        )
        for task in tasks
    ]


def synthesizer_node(state: SynthesizerState):
    """Synthesizer: Aggregate and synthesize all findings"""
    from prompts import (
        SYNTHESIZER_MAIN, 
        SYNTHESIZER_RETRY, 
        SYNTHESIZER_SYSTEM
    )
    
    findings = state.get("subagent_findings", [])
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")

    print(f"\n📊 [Synthesizer] Synthesizing {len(findings)} findings...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    if not findings:
        return {"synthesized_results": "No findings to synthesize."}

    # Format findings - only include essential information
    findings_text = "\n\n".join([
        f"{i+1}. {f.get('task', 'Unknown')[:60]}\n{f.get('summary', 'No summary')}"
        for i, f in enumerate(findings)
    ])

    prompt_content = SYNTHESIZER_MAIN.format(
        query=query,
        findings=findings_text
    )

    if last_error:
        prompt_content = SYNTHESIZER_RETRY.format(
            previous_prompt=prompt_content,
            error=last_error
        )

    structured_llm = get_lead_llm().with_structured_output(SynthesisResult, include_raw=True)
    response = structured_llm.invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=prompt_content),
    ])

    # Use helper to process retry logic
    def fallback(s):
        return {
            "synthesized_results": "Failed to synthesize findings into structured format."
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


def decision_node(state: DecisionState):
    """Decision: Determine if more research is needed"""
    # Only use metadata, not full content - reduces token usage
    iteration_count = state.get("iteration_count", 0)
    findings_count = len(state.get("subagent_findings", []))
    synthesized_length = len(state.get("synthesized_results", ""))

    print("\n🤔 [Decision] Evaluating if more research is needed...")
    print(
        f"   Iteration: {iteration_count}, "
        f"Findings: {findings_count}, "
        f"Synthesis length: {synthesized_length}"
    )

    # Simple decision logic using only metadata
    # Continue if: iteration < 3 AND (few findings OR short synthesis)
    needs_more = (
        iteration_count < 3 and
        (findings_count < 3 or synthesized_length < 500)
    )

    if needs_more:
        print("  ✅ Decision: Continue research")
        return {"needs_more_research": True}
    else:
        print("  ✅ Decision: Finish research")
        return {"needs_more_research": False}


def citation_agent_node(state: CitationAgentState):
    """CitationAgent: Extract citations and create final report"""
    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")
    query = state["query"]

    print("\n📝 [CitationAgent] Extracting citations and creating final report...")

    # Collect all sources - only extract necessary fields
    all_sources = []
    seen_urls = set()
    for finding in findings:
        sources = finding.get("sources", [])
        for source in sources:
            # Handle both dict and Pydantic model
            url = (
                source.get("url", "")
                if isinstance(source, dict)
                else source.get("url", "")
            )
            if url and url not in seen_urls:
                seen_urls.add(url)
                title = (
                    source.get("title", "Unknown")
                    if isinstance(source, dict)
                    else source.get("title", "Unknown")
                )
                all_sources.append(Citation(title=title, url=url))

    # Format citations
    citations = all_sources

    # Create final report - optimized format
    citations_text = "\n".join([
        f"{i+1}. {c.get('title', 'Unknown')} - {c.get('url', '')}"
        for i, c in enumerate(citations)
    ])

    final_report = f"""# Research Report

## Query
{query}

## Findings
{synthesized}

## Sources
{citations_text}
"""

    print(f"  ✅ Final report created with {len(citations)} citations")

    return {
        "citations": citations,
        "final_report": final_report,
    }


def route_decision(
    state: ResearchState,
) -> Literal["lead_researcher", "citation_agent"]:
    """Route based on decision node result"""
    if state.get("needs_more_research", False):
        return "lead_researcher"
    return "citation_agent"


def route_synthesizer(state: ResearchState) -> Literal["synthesizer", "decision"]:
    """Route synthesizer retry or success"""
    if should_retry(state):
        print(f"  ↺ Routing back to synthesizer for retry...")
        return "synthesizer"
    return "decision"


# Build graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("lead_researcher", lead_researcher_node)
workflow.add_node("subagent", subagent_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("decision", decision_node)
workflow.add_node("citation_agent", citation_agent_node)

# Add edges
workflow.add_edge(START, "lead_researcher")
workflow.add_conditional_edges(
    "lead_researcher",
    assign_subagents,
    ["subagent", "lead_researcher"],
)
workflow.add_edge("subagent", "synthesizer")
workflow.add_conditional_edges(
    "synthesizer",
    route_synthesizer,
    ["synthesizer", "decision"],
)
workflow.add_conditional_edges(
    "decision",
    route_decision,
    {
        "lead_researcher": "lead_researcher",
        "citation_agent": "citation_agent",
    },
)
workflow.add_edge("citation_agent", END)

# Compile app
app = workflow.compile()

