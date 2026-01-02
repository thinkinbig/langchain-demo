"""Graph-level helper functions for structured response processing and retry logic"""

from schemas import ResearchState


def process_structured_response(response, state, fallback_func=None):
    """
    Standardized validation and state update for retry loop.

    Args:
        response: Output from .with_structured_output(..., include_raw=True)
        state: Current state dict
        fallback_func: Optional callable to generate fallback state
            if max retries reached

    Returns:
        dict: State update or None if successful
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
    # Signal to caller that retrieval was successful, caller handles "parsed"
    return None


def should_retry(state: ResearchState) -> bool:
    """Common condition to check if we should loop back"""
    return bool(state.get("error"))

