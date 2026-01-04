"""Complexity analyzer: Assess query complexity and recommend resources"""

from graph.utils import process_structured_response
from langchain_core.messages import HumanMessage, SystemMessage
from llm.factory import get_subagent_llm
from prompts import COMPLEXITY_ANALYZER_MAIN, COMPLEXITY_ANALYZER_SYSTEM
from schemas import ComplexityAnalysis, ResearchState


def complexity_analyzer_node(state: ResearchState):
    """ComplexityAnalyzer: Analyze query complexity and recommend workers/iterations"""
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    last_error = state.get("error")

    print("\n📊 [ComplexityAnalyzer] Analyzing query complexity...")
    print(f"   Query: {query[:80]}...")
    if retry_count > 0:
        print(f"   ⚠️  Retry attempt {retry_count}")

    # Build messages
    system_prompt = COMPLEXITY_ANALYZER_SYSTEM
    messages = [SystemMessage(content=system_prompt)]

    prompt_content = COMPLEXITY_ANALYZER_MAIN.format(query=query)
    messages.append(HumanMessage(content=prompt_content))

    # Add retry feedback if needed
    if last_error:
        retry_message = (
            f"\n\n<error>\nThe previous attempt failed validation:\n"
            f"{last_error}\n</error>\n"
            "<instructions>Please correct the JSON structure and try again."
            "</instructions>"
        )
        if messages and isinstance(messages[-1], HumanMessage):
            messages[-1] = HumanMessage(content=prompt_content + retry_message)
        else:
            messages.append(HumanMessage(content=retry_message))

    # Invoke LLM with structured output
    # Use subagent_llm (cheaper model) for this simple classification task
    structured_llm = get_subagent_llm().with_structured_output(
        ComplexityAnalysis, include_raw=True
    )

    response = structured_llm.invoke(messages)

    # Use helper to process retry logic
    def fallback(s):
        # Default to medium complexity if analysis fails
        return {
            "complexity_analysis": ComplexityAnalysis(
                complexity_level="medium",
                recommended_workers=2,
                max_iterations=2,
                rationale=(
                    "Fallback: Defaulting to medium complexity "
                    "due to analysis error"
                ),
                recommended_model="plus"  # Default to plus for fallback
            ),
            "error": None,
            "retry_count": 0
        }

    retry_state = process_structured_response(response, state, fallback)
    if retry_state:
        # If retry_state is returned, it means we either failed (and are looping)
        # or we hit max retries and are returning the fallback.
        return retry_state

    # Success case
    parsed_result = response["parsed"]
    complexity_analysis = parsed_result

    print(f"  ✅ Complexity: {complexity_analysis.complexity_level}")
    print(f"     Recommended workers: {complexity_analysis.recommended_workers}")
    print(f"     Max iterations: {complexity_analysis.max_iterations}")
    print(f"     Recommended model: {complexity_analysis.recommended_model}")
    print(f"     Rationale: {complexity_analysis.rationale[:100]}...")

    return {
        "complexity_analysis": complexity_analysis,
        "error": None,
        "retry_count": 0
    }

