"""Error classification and checkpointer history management"""

import asyncio

from cost_control import CostLimitExceeded


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is recoverable.

    Recoverable: preserve history, reuse thread_id
    Unrecoverable: discard history, generate new thread_id

    Args:
        error: Exception object

    Returns:
        True if error is recoverable, False otherwise
    """
    # Unrecoverable errors
    if isinstance(error, CostLimitExceeded):
        return False
    if isinstance(error, asyncio.TimeoutError):
        return False
    if isinstance(error, ValueError):
        # ValueError might be a parsing error, need to check if max retries reached
        # If error message contains "Max retries" or "Failed to parse",
        # it's unrecoverable
        error_msg = str(error).lower()
        if "max retries" in error_msg or "failed to parse" in error_msg:
            return False

    # Other errors are considered recoverable by default (can retry)
    return True


def should_discard_history(error: Exception) -> bool:
    """
    Determine if checkpointer history should be discarded.

    Args:
        error: Exception object

    Returns:
        True if history should be discarded, False otherwise
    """
    return not is_recoverable_error(error)


def get_error_category(error: Exception) -> str:
    """
    Get error category for logging and user prompts.

    Args:
        error: Exception object

    Returns:
        Error category string: 'recoverable' or 'unrecoverable'
    """
    if is_recoverable_error(error):
        return "recoverable"
    return "unrecoverable"

