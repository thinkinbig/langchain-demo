"""Human approach selector node: Display approaches and wait for user input"""

from typing import Any, Dict

from schemas import ResearchState


def _get_approaches_from_eval(approach_evaluation):
    """
    Extract approaches list from approach_evaluation.
    Handles both Pydantic model and dict formats.
    """
    if not approach_evaluation:
        return None

    if hasattr(approach_evaluation, "approaches"):
        # Pydantic model
        return approach_evaluation.approaches
    elif isinstance(approach_evaluation, dict) and "approaches" in approach_evaluation:
        # Dict format
        return approach_evaluation["approaches"]
    else:
        return None


def format_approaches_for_display(approach_evaluation) -> str:
    """Format approach evaluation for user display"""
    if not approach_evaluation:
        return "No approaches available."

    approaches = _get_approaches_from_eval(approach_evaluation)
    if not approaches:
        return "No approaches available."

    lines = ["\n" + "=" * 80]
    lines.append("📋 Please Select Research Approach")
    lines.append("=" * 80 + "\n")

    for i, approach in enumerate(approaches):
        # Handle both Pydantic model and dict
        if hasattr(approach, "description"):
            desc = approach.description
            advantages = getattr(approach, "advantages", None) or []
            disadvantages = getattr(approach, "disadvantages", None) or []
            suitability = getattr(approach, "suitability", None) or ""
        elif isinstance(approach, dict):
            desc = approach.get("description", "")
            advantages = approach.get("advantages", [])
            disadvantages = approach.get("disadvantages", [])
            suitability = approach.get("suitability", "")
        else:
            continue

        lines.append(f"Approach {i + 1}:")
        lines.append(f"  Description: {desc}")

        if advantages:
            adv_str = ", ".join(advantages)
            lines.append(f"  Advantages: {adv_str}")

        if disadvantages:
            dis_str = ", ".join(disadvantages)
            lines.append(f"  Disadvantages: {dis_str}")

        if suitability:
            lines.append(f"  Suitability: {suitability}")

        lines.append("")  # Empty line between approaches

    lines.append("=" * 80)
    lines.append("Enter approach number (0-2): ")

    return "\n".join(lines)


def human_approach_selector_node(state: ResearchState) -> Dict[str, Any]:
    """
    Human approach selector node.

    This node processes the user's selection that was provided during the interrupt.
    With interrupt_before configured, execution pauses before this node, allowing
    the main program to collect user input and update state via app.update_state().

    When execution resumes, this node receives the updated state with
    selected_approach_index already set, and it just validates and formats the result.

    Args:
        state: ResearchState containing approach_evaluation and selected_approach_index

    Returns:
        State update with validated selected_approach_index and selection_reasoning
    """
    approach_evaluation = state.get("approach_evaluation")

    if not approach_evaluation:
        msg = (
            "  ⚠️  No approach evaluation found in state. "
            "Skipping human selection."
        )
        print(msg)
        return {
            "awaiting_human_input": False,
            "selected_approach_index": None,
        }

    # Check if user selection was already provided via update_state
    selected_index = state.get("selected_approach_index")

    if selected_index is None:
        # No selection provided - this shouldn't happen if interrupt was handled correctly
        msg = (
            "  ⚠️  No approach selection found in state. "
            "User input may not have been collected during interrupt."
        )
        print(msg)
        return {
            "awaiting_human_input": False,
            "selected_approach_index": None,
        }

    # Validate the selection
    if not (0 <= selected_index <= 2):
        msg = (
            f"  ⚠️  Invalid approach index {selected_index}. "
            "Must be 0, 1, or 2. Skipping selection."
        )
        print(msg)
        return {
            "awaiting_human_input": False,
            "selected_approach_index": None,
        }

    # Valid selection - format the result
    approaches = _get_approaches_from_eval(approach_evaluation)
    if not approaches or selected_index >= len(approaches):
        num_approaches = len(approaches) if approaches else 0
        msg = (
            f"  ⚠️  Invalid approach index {selected_index}. "
            f"Only {num_approaches} approaches available."
        )
        print(msg)
        return {
            "awaiting_human_input": False,
            "selected_approach_index": None,
        }

    selected_approach_obj = approaches[selected_index]

    # Handle both Pydantic model and dict
    if hasattr(selected_approach_obj, "description"):
        selected_approach_desc = selected_approach_obj.description
    elif isinstance(selected_approach_obj, dict):
        selected_approach_desc = selected_approach_obj.get("description", "")
    else:
        selected_approach_desc = "Unknown approach"

    desc = selected_approach_desc[:80]
    print(
        f"\n✅ Processing selected approach "
        f"{selected_index + 1}: {desc}..."
    )

    # Return state update with validated user selection
    # IMPORTANT: Include approach_evaluation to preserve it
    # Convert Pydantic model to dict if needed for LangGraph serialization
    approach_eval_dict = (
        approach_evaluation.model_dump()
        if hasattr(approach_evaluation, "model_dump")
        else approach_evaluation
    )
    result = {
        "selected_approach_index": selected_index,
        "selected_approach": selected_approach_desc,
        "selection_reasoning": (
            f"User selected approach "
            f"{selected_index + 1}: "
            f"{selected_approach_desc[:100]}"
        ),
        "awaiting_human_input": False,
        # Preserve approach_evaluation so lead_researcher can use it
        # Convert to dict for LangGraph serialization
        "approach_evaluation": approach_eval_dict,
    }
    return result

