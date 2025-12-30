"""Router Pattern: Intelligent task routing to specialized experts

This demonstrates the Router Pattern from Building Effective Agents:
- A router node analyzes the task and determines the best expert
- Multiple specialized expert nodes handle different task types
- All experts eventually converge to a final formatter
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from schemas import RouterState

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.3,
)


def router_node(state: RouterState):
    """Router: Analyzes task and determines which expert to use"""
    task = state["task"]
    print(f"\n🔀 [Router] Analyzing task: {task[:60]}...")

    # Use LLM to classify the task type
    prompt = f"""Analyze this task and determine the best expert to handle it:

Task: "{task}"

Available experts:
1. code_expert - For programming, code generation, debugging, technical tasks
2. analysis_expert - For data analysis, research, insights, statistical work
3. writing_expert - For writing, content creation, editing, documentation
4. calculation_expert - For mathematical calculations, formulas, computations
5. general_expert - For general questions, explanations, or unclear tasks

Respond with ONLY the expert name (e.g., "code_expert"), no explanation."""
    response = llm.invoke([HumanMessage(content=prompt)])
    task_type = response.content.strip().lower()

    # Validate and normalize task type
    valid_types = [
        "code_expert",
        "analysis_expert",
        "writing_expert",
        "calculation_expert",
        "general_expert",
    ]
    if task_type not in valid_types:
        task_type = "general_expert"

    # Generate routing reason
    reason_prompt = (
        f'Briefly explain why "{task_type}" is the best choice for this task:\n'
        f'"{task}"\n\nOne sentence only.'
    )
    reason_response = llm.invoke([HumanMessage(content=reason_prompt)])
    routing_reason = reason_response.content.strip()

    print(f"  ✅ Routed to: {task_type}")
    print(f"  📝 Reason: {routing_reason}")

    return {
        "task_type": task_type,
        "routed_to": task_type,
        "routing_reason": routing_reason,
    }


def code_expert_node(state: RouterState):
    """Code Expert: Handles programming and technical tasks"""
    task = state["task"]
    print("\n💻 [Code Expert] Processing code-related task...")

    prompt = f"""You are an expert software developer. Help with this task:

{task}

Provide:
1. Clear, well-commented code
2. Explanation of the approach
3. Any important considerations

If the task is not code-related, adapt your response accordingly."""
    response = llm.invoke([
        SystemMessage(content="You are an expert software developer."),
        HumanMessage(content=prompt),
    ])

    print("  ✅ Code expert completed")
    return {"expert_output": response.content}


def analysis_expert_node(state: RouterState):
    """Analysis Expert: Handles data analysis and research tasks"""
    task = state["task"]
    print("\n📊 [Analysis Expert] Processing analysis task...")

    prompt = f"""You are an expert data analyst and researcher. Help with this task:

{task}

Provide:
1. Structured analysis
2. Key insights and findings
3. Data-driven recommendations

Focus on analytical thinking and evidence-based conclusions."""
    response = llm.invoke([
        SystemMessage(content="You are an expert data analyst and researcher."),
        HumanMessage(content=prompt),
    ])

    print("  ✅ Analysis expert completed")
    return {"expert_output": response.content}


def writing_expert_node(state: RouterState):
    """Writing Expert: Handles writing and content creation"""
    task = state["task"]
    print("\n✍️ [Writing Expert] Processing writing task...")

    prompt = f"""You are an expert writer and content creator. Help with this task:

{task}

Provide:
1. Well-structured, engaging content
2. Clear communication
3. Appropriate tone and style

Focus on clarity, engagement, and effective communication."""
    response = llm.invoke([
        SystemMessage(content="You are an expert writer and content creator."),
        HumanMessage(content=prompt),
    ])

    print("  ✅ Writing expert completed")
    return {"expert_output": response.content}


def calculation_expert_node(state: RouterState):
    """Calculation Expert: Handles mathematical and computational tasks"""
    task = state["task"]
    print("\n🔢 [Calculation Expert] Processing calculation task...")

    prompt = (
        "You are an expert mathematician and computational specialist. "
        f"Help with this task:\n\n{task}\n\n"
        "Provide:\n1. Step-by-step calculations\n"
        "2. Clear mathematical reasoning\n"
        "3. Final answer with explanation\n\n"
        "Show your work and explain each step."
    )
    response = llm.invoke([
        SystemMessage(content="You are an expert mathematician."),
        HumanMessage(content=prompt),
    ])

    print("  ✅ Calculation expert completed")
    return {"expert_output": response.content}


def general_expert_node(state: RouterState):
    """General Expert: Handles general questions and unclear tasks"""
    task = state["task"]
    print("\n🤔 [General Expert] Processing general task...")

    prompt = (
        f"You are a helpful AI assistant. Help with this task:\n\n{task}\n\n"
        "Provide a comprehensive, well-thought-out response that addresses "
        "all aspects of the request."
    )
    response = llm.invoke([
        SystemMessage(content="You are a helpful and knowledgeable AI assistant."),
        HumanMessage(content=prompt),
    ])

    print("  ✅ General expert completed")
    return {"expert_output": response.content}


def formatter_node(state: RouterState):
    """Formatter: Formats the final output based on task type"""
    task_type = state.get("task_type", "general_expert")
    expert_output = state.get("expert_output", "")
    routing_reason = state.get("routing_reason", "")

    print("\n📝 [Formatter] Formatting final output...")

    # Format based on task type
    if task_type == "code_expert":
        formatted = f"""# Code Solution

{routing_reason}

## Solution

{expert_output}

---
*Generated by Code Expert*
"""
    elif task_type == "analysis_expert":
        formatted = f"""# Analysis Report

{routing_reason}

## Analysis

{expert_output}

---
*Generated by Analysis Expert*
"""
    elif task_type == "writing_expert":
        formatted = f"""# Content

{routing_reason}

{expert_output}

---
*Generated by Writing Expert*
"""
    elif task_type == "calculation_expert":
        formatted = f"""# Calculation Result

{routing_reason}

## Solution

{expert_output}

---
*Generated by Calculation Expert*
"""
    else:
        formatted = f"""# Response

{routing_reason}

{expert_output}

---
*Generated by General Expert*
"""

    print(f"  ✅ Output formatted for {task_type}")
    return {"expert_output": formatted}


def route_to_expert(state: RouterState) -> Literal[
    "code_expert",
    "analysis_expert",
    "writing_expert",
    "calculation_expert",
    "general_expert",
]:
    """Route function: Determines which expert to use based on router's decision"""
    task_type = state.get("task_type", "general_expert")
    return task_type  # type: ignore


# Build graph
workflow = StateGraph(RouterState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("code_expert", code_expert_node)
workflow.add_node("analysis_expert", analysis_expert_node)
workflow.add_node("writing_expert", writing_expert_node)
workflow.add_node("calculation_expert", calculation_expert_node)
workflow.add_node("general_expert", general_expert_node)
workflow.add_node("formatter", formatter_node)

# Add edges
workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    route_to_expert,
    {
        "code_expert": "code_expert",
        "analysis_expert": "analysis_expert",
        "writing_expert": "writing_expert",
        "calculation_expert": "calculation_expert",
        "general_expert": "general_expert",
    },
)
# All experts converge to formatter
workflow.add_edge("code_expert", "formatter")
workflow.add_edge("analysis_expert", "formatter")
workflow.add_edge("writing_expert", "formatter")
workflow.add_edge("calculation_expert", "formatter")
workflow.add_edge("general_expert", "formatter")
workflow.add_edge("formatter", END)

# Compile app
app = workflow.compile()

