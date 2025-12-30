"""Multi-Stage Decision Agent: Complex workflow with validation and error handling"""

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from schemas import DecisionState

# Initialize LLM
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.3,
)


def input_parser_node(state: DecisionState):
    """Stage 1: Parse and validate input request"""
    request = state["request"]
    print(f"\n[Stage 1: Input Parser] Processing request: {request[:50]}...")

    # Extract key information using structured extraction
    prompt = f"""Extract structured information from this request:
"{request}"

Return a JSON object with these fields:
- intent: main purpose (e.g., "data_analysis", "report_generation", "calculation")
- entities: list of key entities mentioned
- parameters: dict of key-value parameters
- priority: "high", "medium", or "low"
- estimated_complexity: 1-10 scale

Return ONLY valid JSON, no markdown."""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    # Remove markdown code blocks if present
    content = re.sub(r"```json\n?", "", content)
    content = re.sub(r"```\n?", "", content).strip()

    try:
        extracted_data = json.loads(content)
    except json.JSONDecodeError:
        extracted_data = {
            "intent": "unknown",
            "entities": [],
            "parameters": {},
            "priority": "medium",
            "estimated_complexity": 5,
        }

    # Validate extracted data
    validation_errors = []
    if "intent" not in extracted_data:
        validation_errors.append("Missing intent field")
    if "entities" not in extracted_data:
        validation_errors.append("Missing entities field")

    validation_result = (
        "✅ Input parsed successfully"
        if not validation_errors
        else f"⚠️ Validation issues: {', '.join(validation_errors)}"
    )

    print(f"  Extracted: {extracted_data.get('intent', 'unknown')} intent")
    print(f"  Validation: {validation_result}")

    return {
        "extracted_data": extracted_data,
        "validation_results": [validation_result],
        "processing_log": [f"[Stage 1] Parsed input: {extracted_data.get('intent')}"],
        "stage": "validator",
        "error_count": len(validation_errors),
    }


def validator_node(state: DecisionState):
    """Stage 2: Validate extracted data and requirements"""
    extracted_data = state["extracted_data"]
    print("\n[Stage 2: Validator] Validating extracted data...")

    # Check data completeness
    checks = []
    if not extracted_data.get("entities"):
        checks.append("No entities found - may need clarification")
    if extracted_data.get("estimated_complexity", 0) > 8:
        checks.append("High complexity detected - may require additional resources")
    if extracted_data.get("priority") == "high":
        checks.append("High priority request - expediting processing")

    # Validate against business rules
    intent = extracted_data.get("intent", "")
    if intent not in [
        "data_analysis",
        "report_generation",
        "calculation",
        "text_processing",
        "unknown",
    ]:
        checks.append(f"Unknown intent '{intent}' - using default handler")

    validation_result = (
        "✅ All validations passed"
        if not checks
        else f"⚠️ Validation notes: {'; '.join(checks)}"
    )

    print(f"  {validation_result}")

    # Decide next stage based on validation
    next_stage = "processor"
    if state["error_count"] > 0:
        next_stage = "error_handler"
    elif extracted_data.get("estimated_complexity", 0) > 7:
        next_stage = "complex_processor"

    return {
        "validation_results": [validation_result],
        "processing_log": [f"[Stage 2] Validation complete: {len(checks)} checks"],
        "decisions": [f"Next stage: {next_stage}"],
        "stage": next_stage,
    }


def processor_node(state: DecisionState):
    """Stage 3: Process the request based on extracted data"""
    extracted_data = state["extracted_data"]
    request = state["request"]
    print("\n[Stage 3: Processor] Processing request...")

    intent = extracted_data.get("intent", "unknown")
    entities = extracted_data.get("entities", [])

    # Process based on intent
    if intent == "data_analysis":
        prompt = f"""Analyze the following request and provide insights:
Request: {request}
Entities: {entities}

Provide a structured analysis with:
1. Key findings
2. Patterns identified
3. Recommendations"""
    elif intent == "report_generation":
        prompt = f"""Generate a comprehensive report based on:
Request: {request}
Entities: {entities}

Include:
1. Executive summary
2. Detailed findings
3. Conclusions"""
    elif intent == "calculation":
        prompt = f"""Perform calculations or analysis for:
Request: {request}
Entities: {entities}

Show step-by-step calculations and final result."""
    else:
        prompt = f"""Process this request:
{request}

Provide a detailed response addressing all aspects."""

    response = llm.invoke([
        SystemMessage(content="You are a precise assistant."),
        HumanMessage(content=prompt),
    ])

    output = response.content
    print(f"  Generated output ({len(output)} characters)")

    return {
        "processing_log": [f"[Stage 3] Processed with intent: {intent}"],
        "final_output": output,
        "stage": "quality_checker",
    }


def complex_processor_node(state: DecisionState):
    """Stage 3b: Handle complex requests with multi-step processing"""
    extracted_data = state["extracted_data"]
    request = state["request"]
    print("\n[Stage 3b: Complex Processor] Handling complex request...")

    # Multi-step processing
    steps = [
        "Breaking down into sub-tasks",
        "Analyzing each component",
        "Synthesizing results",
    ]

    results = []
    for i, step in enumerate(steps, 1):
        print(f"  Step {i}/{len(steps)}: {step}")
        prompt = f"""Step {i}: {step}

Request: {request}
Context: {extracted_data}

Provide detailed output for this step."""
        response = llm.invoke([HumanMessage(content=prompt)])
        results.append(f"Step {i}: {response.content[:100]}...")

    # Combine results
    results_text = "\n".join(results)
    combined_prompt = (
        "Synthesize these partial results into a comprehensive response:\n\n"
        f"{results_text}\n\n"
        f"Original request: {request}\n"
        "Provide a unified, comprehensive final output."
    )
    final_response = llm.invoke([HumanMessage(content=combined_prompt)])

    output = final_response.content
    print(f"  Complex processing complete ({len(output)} characters)")

    return {
        "processing_log": [f"[Stage 3b] Complex processing: {len(steps)} steps"],
        "final_output": output,
        "stage": "quality_checker",
    }


def quality_checker_node(state: DecisionState):
    """Stage 4: Quality check and final validation"""
    final_output = state.get("final_output", "")
    request = state["request"]
    print("\n[Stage 4: Quality Checker] Checking output quality...")

    # Quality checks
    checks = []
    if len(final_output) < 50:
        checks.append("Output too short - may be incomplete")
    if len(final_output) > 5000:
        checks.append("Output very long - may need summarization")
    is_calculation = "calculation" in request.lower()
    has_numbers = any(char.isdigit() for char in final_output)
    if is_calculation and not has_numbers:
        checks.append("Calculation request but no numbers in output")

    # Check relevance
    output_preview = final_output[:500]
    relevance_prompt = f"""Check if this output adequately addresses the request:

Request: {request}
Output: {output_preview}...

Respond with:
- "PASS" if output is relevant and complete
- "FAIL" with reason if there are issues"""
    relevance_response = llm.invoke([HumanMessage(content=relevance_prompt)])
    relevance_check = relevance_response.content.strip()

    if "PASS" not in relevance_check.upper():
        checks.append(f"Relevance check: {relevance_check}")

    quality_result = (
        "✅ Quality check passed"
        if not checks
        else f"⚠️ Quality issues: {'; '.join(checks)}"
    )

    print(f"  {quality_result}")

    # Decide if output needs refinement
    next_stage = "formatter"
    if checks and state["error_count"] < 2:
        next_stage = "refiner"
    elif state["error_count"] >= 2:
        next_stage = "error_handler"

    return {
        "validation_results": [quality_result],
        "processing_log": [f"[Stage 4] Quality check: {len(checks)} issues"],
        "decisions": [f"Next stage: {next_stage}"],
        "stage": next_stage,
    }


def refiner_node(state: DecisionState):
    """Stage 5: Refine output based on quality check"""
    final_output = state.get("final_output", "")
    request = state["request"]
    validation_results = state.get("validation_results", [])
    print("\n[Stage 5: Refiner] Refining output...")

    # Get quality issues from validation
    issues = "; ".join([r for r in validation_results if "⚠️" in r])

    prompt = f"""Refine this output based on the quality issues identified:

Original request: {request}
Current output: {final_output}
Quality issues: {issues}

Provide an improved version that addresses all issues."""
    response = llm.invoke([HumanMessage(content=prompt)])

    refined_output = response.content
    print(f"  Output refined ({len(refined_output)} characters)")

    return {
        "processing_log": ["[Stage 5] Output refined"],
        "final_output": refined_output,
        "stage": "formatter",
        "error_count": max(0, state["error_count"] - 1),  # Reduce error count
    }


def formatter_node(state: DecisionState):
    """Stage 6: Format final output"""
    final_output = state.get("final_output", "")
    extracted_data = state.get("extracted_data", {})
    print("\n[Stage 6: Formatter] Formatting final output...")

    intent = extracted_data.get("intent", "unknown")
    priority = extracted_data.get("priority", "medium")

    # Format based on intent and priority
    if intent == "report_generation":
        formatted = f"""# Report

{final_output}

---
Generated with priority: {priority}
"""
    elif intent == "data_analysis":
        formatted = f"""## Analysis Results

{final_output}

---
Analysis completed"""
    else:
        formatted = f"""## Response

{final_output}

---
Request processed"""

    print(f"  Output formatted for {intent}")

    return {
        "processing_log": [f"[Stage 6] Formatted as {intent}"],
        "final_output": formatted,
        "stage": END,
    }


def error_handler_node(state: DecisionState):
    """Error handler: Handle errors and decide recovery"""
    error_count = state.get("error_count", 0)
    print(f"\n[Error Handler] Handling errors (count: {error_count})...")

    if error_count >= 3:
        print("  ❌ Too many errors - aborting")
        return {
            "final_output": "Error: Unable to process request after multiple attempts.",
            "stage": END,
        }

    # Retry from a previous stage
    retry_stage = state.get("retry_stage", "input_parser")
    print(f"  🔄 Retrying from stage: {retry_stage}")

    return {
        "processing_log": [f"[Error Handler] Retrying from {retry_stage}"],
        "stage": retry_stage,
        "error_count": error_count + 1,
        "retry_stage": retry_stage,
    }


def route_after_validator(
    state: DecisionState,
) -> Literal["processor", "complex_processor", "error_handler"]:
    """Route after validation stage"""
    if state["error_count"] > 0:
        return "error_handler"
    extracted = state.get("extracted_data", {})
    complexity = extracted.get("estimated_complexity", 0)
    if complexity > 7:
        return "complex_processor"
    return "processor"


def route_after_quality_check(
    state: DecisionState,
) -> Literal["formatter", "refiner", "error_handler"]:
    """Route after quality check"""
    if state["error_count"] >= 2:
        return "error_handler"
    validation_results = state.get("validation_results", [])
    has_issues = any("⚠️" in r for r in validation_results)
    if has_issues:
        return "refiner"
    return "formatter"


# Build graph
workflow = StateGraph(DecisionState)

# Add nodes
workflow.add_node("input_parser", input_parser_node)
workflow.add_node("validator", validator_node)
workflow.add_node("processor", processor_node)
workflow.add_node("complex_processor", complex_processor_node)
workflow.add_node("quality_checker", quality_checker_node)
workflow.add_node("refiner", refiner_node)
workflow.add_node("formatter", formatter_node)
workflow.add_node("error_handler", error_handler_node)

# Add edges
workflow.add_edge(START, "input_parser")
workflow.add_edge("input_parser", "validator")
workflow.add_conditional_edges(
    "validator",
    route_after_validator,
    {
        "processor": "processor",
        "complex_processor": "complex_processor",
        "error_handler": "error_handler",
    },
)
workflow.add_edge("processor", "quality_checker")
workflow.add_edge("complex_processor", "quality_checker")
workflow.add_conditional_edges(
    "quality_checker",
    route_after_quality_check,
    {
        "formatter": "formatter",
        "refiner": "refiner",
        "error_handler": "error_handler",
    },
)
workflow.add_edge("refiner", "formatter")
workflow.add_edge("formatter", END)
workflow.add_edge("error_handler", END)

# Compile app
app = workflow.compile()

