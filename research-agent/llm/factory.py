"""LLM factory with lazy loading pattern"""

from langchain_openai import ChatOpenAI

# LLM Configuration - Lazy Loading Pattern
# LLMs are initialized on first use, not at module import time
# This allows tests to mock LLMs without needing API keys

_turbo_llm = None
_plus_llm = None
_max_llm = None

# Import tracing for local logging (optional)
try:
    from tracing import get_callbacks
except ImportError:
    def get_callbacks():
        return []


def get_llm_by_model_choice(model_choice: str):
    """
    Get LLM instance based on model choice (turbo, plus, or max).
    Uses lazy loading pattern for efficient resource management.

    Args:
        model_choice: "turbo", "plus", or "max"

    Returns:
        ChatOpenAI instance with the selected model

    Raises:
        ValueError: If model_choice is not one of the valid options
    """
    from config import settings

    model_choice = model_choice.lower()

    if model_choice == "turbo":
        global _turbo_llm
        if _turbo_llm is None:
            _turbo_llm = ChatOpenAI(
                base_url=settings.LLM_BASE_URL,
                model=settings.MODEL_TURBO,
                temperature=settings.TEMP_PLANNER,
                max_retries=2,
                callbacks=get_callbacks(),
            )
        return _turbo_llm
    elif model_choice == "plus":
        global _plus_llm
        if _plus_llm is None:
            _plus_llm = ChatOpenAI(
                base_url=settings.LLM_BASE_URL,
                model=settings.MODEL_PLUS,
                temperature=settings.TEMP_PLANNER,
                max_retries=2,
                callbacks=get_callbacks(),
            )
        return _plus_llm
    elif model_choice == "max":
        global _max_llm
        if _max_llm is None:
            _max_llm = ChatOpenAI(
                base_url=settings.LLM_BASE_URL,
                model=settings.MODEL_MAX,
                temperature=settings.TEMP_PLANNER,
                max_retries=2,
                callbacks=get_callbacks(),
            )
        return _max_llm
    else:
        raise ValueError(
            f"Invalid model_choice: {model_choice}. "
            "Must be 'turbo', 'plus', or 'max'"
        )
