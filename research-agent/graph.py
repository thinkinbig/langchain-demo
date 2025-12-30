"""Multi-agent research system graph"""

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from schemas import ResearchState
from tools import search_web

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.3,
    max_retries=2,
)


def lead_researcher_node(state: ResearchState):
    """LeadResearcher: Analyze query, create plan, generate subagent tasks"""
    query = state["query"]
    iteration_count = state.get("iteration_count", 0)

    print(f"\n🔍 [LeadResearcher] Analyzing query (iteration {iteration_count + 1})...")
    print(f"   Query: {query[:80]}...")

    # Simple prompt for query analysis and task generation
    if iteration_count == 0:
        prompt = (
            f"Analyze this research query and break it into 2-3 independent sub-tasks "
            f"that can be researched in parallel:\n\n"
            f'Query: "{query}"\n\n'
            f"Return a JSON array of task descriptions, e.g.:\n"
            f'["Task 1 description", "Task 2 description", ...]\n\n'
            f"Return ONLY the JSON array, no markdown."
        )
    else:
        # Refine strategy based on existing findings
        existing_findings = state.get("subagent_findings", [])
        findings_summary = "\n".join([
            f"- {f.get('task', 'Unknown')}: {f.get('summary', '')[:100]}..."
            for f in existing_findings[:3]
        ])

        prompt = (
            f"Based on these existing findings, refine the research strategy:\n\n"
            f"Original Query: {query}\n\n"
            f"Existing Findings:\n{findings_summary}\n\n"
            f"Generate 1-2 additional research tasks to fill gaps.\n\n"
            f"Return a JSON array of task descriptions."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Parse tasks
    content = re.sub(r"```json\n?", "", content)
    content = re.sub(r"```\n?", "", content).strip()

    try:
        tasks = json.loads(content)
        if not isinstance(tasks, list):
            tasks = [tasks]
    except json.JSONDecodeError:
        # Fallback: create default tasks
        tasks = [
            f"Research: {query}",
            f"Find information about: {query}",
        ]

    # Create research plan
    plan = f"Research plan for: {query}\nTasks: {len(tasks)} sub-tasks"

    print(f"  ✅ Created {len(tasks)} sub-tasks")
    for i, task in enumerate(tasks, 1):
        print(f"     {i}. {task[:60]}...")

    return {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "iteration_count": iteration_count + 1,
    }


def subagent_node(state: ResearchState):
    """Subagent: Perform web search and analyze results"""
    tasks = state.get("subagent_tasks", [])
    if not tasks:
        return {}

    # Get the first task (each worker gets one task via Send)
    task = tasks[0]

    print(f"\n🔎 [Subagent] Researching: {task[:60]}...")

    # Perform web search
    search_results = search_web(task, max_results=5)

    if not search_results:
        print("  ⚠️  No search results found")
        return {
            "subagent_findings": [{
                "task": task,
                "summary": "No information found",
                "sources": [],
            }]
        }

    # Analyze and summarize findings
    sources_text = "\n\n".join([
        f"Source {i+1}: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:200]}..."
        for i, r in enumerate(search_results)
    ])

    analysis_prompt = (
        f"Research this specific task and summarize the key findings:\n\n"
        f"Task: {task}\n\n"
        f"Search Results:\n{sources_text}\n\n"
        f"Provide a concise summary of the key findings (2-3 sentences)."
    )

    response = llm.invoke([
        SystemMessage(content="You are a research assistant. Summarize findings clearly."),
        HumanMessage(content=analysis_prompt),
    ])

    # Extract sources
    sources = [
        {"title": r["title"], "url": r["url"]}
        for r in search_results
    ]

    finding = {
        "task": task,
        "summary": response.content,
        "sources": sources,
    }

    print(f"  ✅ Found {len(sources)} sources, summary created")

    return {"subagent_findings": [finding]}


def assign_subagents(state: ResearchState):
    """Fan-out: Assign each task to a subagent"""
    tasks = state.get("subagent_tasks", [])
    print(f"\n🔀 [Fan-out] Distributing {len(tasks)} tasks to subagents...")

    return [Send("subagent", {"subagent_tasks": [task]}) for task in tasks]


def synthesizer_node(state: ResearchState):
    """Synthesizer: Aggregate and synthesize all findings"""
    findings = state.get("subagent_findings", [])
    query = state["query"]

    print(f"\n📊 [Synthesizer] Synthesizing {len(findings)} findings...")

    if not findings:
        return {"synthesized_results": "No findings to synthesize."}

    # Format findings for synthesis
    findings_text = "\n\n".join([
        f"Task: {f.get('task', 'Unknown')}\n"
        f"Summary: {f.get('summary', 'No summary')}\n"
        f"Sources: {len(f.get('sources', []))} sources"
        for f in findings
    ])

    synthesis_prompt = (
        f"Synthesize these research findings into a comprehensive answer:\n\n"
        f"Original Query: {query}\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Provide a well-structured, comprehensive answer that integrates all findings."
    )

    response = llm.invoke([
        SystemMessage(content="You are a synthesis expert. Combine insights effectively."),
        HumanMessage(content=synthesis_prompt),
    ])

    print(f"  ✅ Synthesis complete ({len(response.content)} chars)")

    return {"synthesized_results": response.content}


def decision_node(state: ResearchState):
    """Decision: Determine if more research is needed"""
    iteration_count = state.get("iteration_count", 0)
    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")

    print("\n🤔 [Decision] Evaluating if more research is needed...")
    print(f"   Iteration: {iteration_count}, Findings: {len(findings)}")

    # Simple decision logic
    # Continue if: iteration < 3 AND (few findings OR short synthesis)
    needs_more = (
        iteration_count < 3 and
        (len(findings) < 3 or len(synthesized) < 500)
    )

    if needs_more:
        print("  ✅ Decision: Continue research")
        return {"needs_more_research": True}
    else:
        print("  ✅ Decision: Finish research")
        return {"needs_more_research": False}


def citation_agent_node(state: ResearchState):
    """CitationAgent: Extract citations and create final report"""
    findings = state.get("subagent_findings", [])
    synthesized = state.get("synthesized_results", "")
    query = state["query"]

    print("\n📝 [CitationAgent] Extracting citations and creating final report...")

    # Collect all sources
    all_sources = []
    for finding in findings:
        for source in finding.get("sources", []):
            if source not in all_sources:
                all_sources.append(source)

    # Format citations
    citations = [
        {
            "title": src.get("title", "Unknown"),
            "url": src.get("url", ""),
        }
        for src in all_sources
    ]

    # Create final report
    citations_text = "\n".join([
        f"{i+1}. {c['title']} - {c['url']}"
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


def route_decision(state: ResearchState) -> Literal["lead_researcher", "citation_agent"]:
    """Route based on decision node result"""
    if state.get("needs_more_research", False):
        return "lead_researcher"
    return "citation_agent"


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
    ["subagent"],
)
workflow.add_edge("subagent", "synthesizer")
workflow.add_edge("synthesizer", "decision")
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

