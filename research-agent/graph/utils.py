"""Graph-level utility functions for structured response processing and retry logic"""

from schemas import ResearchState


def process_structured_response(response, state):
    """
    Standardized validation and state update for retry loop.

    Args:
        response: Output from .with_structured_output(..., include_raw=True)
        state: Current state dict

    Returns:
        dict: State update or None if successful

    Raises:
        ValueError: If max retries reached and parsing still fails
    """
    retry_count = state.get("retry_count", 0)
    parsing_error = response.get("parsing_error")

    if parsing_error:
        print(f"  ❌ Validation failed: {parsing_error}")

        # Check max retries (e.g., 3 attempts total: 0, 1, 2)
        if retry_count >= 2:
            print(f"  ❌ Max retries ({retry_count + 1}) reached")
            error_msg = (
                f"Failed to parse LLM response after {retry_count + 1} attempts: "
                f"{parsing_error}"
            )
            raise ValueError(error_msg)

        # Route back for retry
        return {
            "error": str(parsing_error),
            "retry_count": retry_count + 1
        }

    # Success
    # Signal to caller that retrieval was successful, caller handles "parsed"
    return None


def should_retry(state: ResearchState) -> bool:
    """Common condition to check if we should loop back"""
    return bool(state.get("error"))

