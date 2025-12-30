"""Sectioning Pattern: Break task into independent subtasks run in parallel

This demonstrates the Sectioning parallelization pattern:
- Break a complex task into independent sections
- Process each section in parallel
- Aggregate results at the end
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from schemas import SectioningState

# Initialize LLM with async support for true parallelization
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.3,
    max_retries=2,
)


def section_planner_node(state: SectioningState):
    """Break task into independent sections that can be processed in parallel"""
    task = state["task"]
    print("\n📋 [Section Planner] Breaking task into sections...")

    prompt = (
        "Break this task into 3-4 independent sections that can be "
        "processed in parallel:\n\n"
        f'Task: "{task}"\n\n'
        "Each section should:\n"
        "- Be independent (can be processed separately)\n"
        "- Focus on a specific aspect\n"
        "- Be clear and actionable\n\n"
        "Return a JSON array of section descriptions, e.g.:\n"
        '["Section 1 description", "Section 2 description", ...]\n\n'
        "Return ONLY the JSON array, no markdown."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # Parse sections
    import json
    import re
    content = re.sub(r"```json\n?", "", content)
    content = re.sub(r"```\n?", "", content).strip()

    try:
        sections = json.loads(content)
        if not isinstance(sections, list):
            sections = [sections]
    except json.JSONDecodeError:
        # Fallback: create sections manually
        sections = [
            f"Analyze the technical aspects of: {task}",
            f"Evaluate the practical implications of: {task}",
            f"Consider the strategic considerations of: {task}",
        ]

    print(f"  ✅ Created {len(sections)} independent sections")
    for i, section in enumerate(sections, 1):
        print(f"     {i}. {section[:60]}...")

    return {"sections": sections}


def section_worker_node(state: SectioningState):
    """Process a single section independently"""
    import time
    sections = state["sections"]
    if not sections:
        return {}

    # Get the first section (each worker gets one section via Send)
    section = sections[0]
    worker_id = id(state) % 10000  # Unique worker ID
    start_time = time.time()
    print(
        f"\n⚙️  [Section Worker {worker_id}] Processing: {section[:50]}... "
        f"(started at {start_time:.2f})"
    )

    prompt = (
        "Process this section independently and thoroughly:\n\n"
        f"{section}\n\n"
        "Provide a detailed, focused response for this specific section."
    )
    # Use async invoke for true parallelization
    # Note: LangGraph handles async automatically when using Send()
    response = llm.invoke([
        SystemMessage(
            content="You are a focused specialist. Provide detailed analysis."
        ),
        HumanMessage(content=prompt),
    ])

    end_time = time.time()
    elapsed = end_time - start_time
    result = f"**Section: {section}**\n\n{response.content}"
    print(
        f"  ✅ Section Worker {worker_id} completed in {elapsed:.2f}s "
        f"({len(response.content)} chars)"
    )

    return {"section_results": [result]}


def assign_sections(state: SectioningState):
    """Fan-out: Assign each section to a worker"""
    sections = state.get("sections", [])
    print(f"\n🔀 [Fan-out] Distributing {len(sections)} sections to workers...")
    return [Send("section_worker", {"sections": [section]}) for section in sections]


def aggregator_node(state: SectioningState):
    """Aggregate all section results into final summary"""
    section_results = state.get("section_results", [])
    task = state["task"]
    print(f"\n📊 [Aggregator] Combining {len(section_results)} section results...")

    all_results = "\n\n".join(section_results)

    prompt = (
        "Synthesize these parallel section results into a comprehensive "
        "final answer:\n\n"
        f"Original Task: {task}\n\n"
        f"Section Results:\n{all_results}\n\n"
        "Provide a well-structured, comprehensive summary that integrates "
        "all sections."
    )
    response = llm.invoke([
        SystemMessage(
            content="You are a synthesis expert. Combine insights effectively."
        ),
        HumanMessage(content=prompt),
    ])

    print(f"  ✅ Final summary created ({len(response.content)} chars)")
    return {"final_summary": response.content}


# Build graph
workflow = StateGraph(SectioningState)

# Add nodes
workflow.add_node("section_planner", section_planner_node)
workflow.add_node("section_worker", section_worker_node)
workflow.add_node("aggregator", aggregator_node)

# Add edges
workflow.add_edge(START, "section_planner")
workflow.add_conditional_edges(
    "section_planner",
    assign_sections,
    ["section_worker"],
)
workflow.add_edge("section_worker", "aggregator")
workflow.add_edge("aggregator", END)

# Compile app
sectioning_app = workflow.compile()

