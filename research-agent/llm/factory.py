"""LLM factory with lazy loading pattern"""

from langchain_openai import ChatOpenAI

# LLM Configuration - Lazy Loading Pattern
# LLMs are initialized on first use, not at module import time
# This allows tests to mock LLMs without needing API keys

_lead_llm = None
_subagent_llm = None

# Import tracing for local logging (optional)
try:
    from tracing import get_callbacks
except ImportError:
    def get_callbacks():
        return []


def get_lead_llm():
    """
    Get or create the lead LLM instance (lazy loading).
    Used for high-level tasks: planning, synthesis, verification.
    """
    global _lead_llm
    if _lead_llm is None:
        from config import settings
        _lead_llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            model=settings.MODEL_PLANNER,  # Use planner model config as default
            temperature=settings.TEMP_PLANNER,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _lead_llm


def get_subagent_llm():
    """
    Get or create the subagent LLM instance (lazy loading).
    Used for sub-tasks and extraction tasks (citations, web content).
    """
    global _subagent_llm
    if _subagent_llm is None:
        from config import settings
        _subagent_llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            # Use extractor model config (usually turbo)
            model=settings.MODEL_EXTRACTOR,
            temperature=settings.TEMP_EXTRACTOR,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _subagent_llm


# Backward compatibility aliases (deprecated, will be removed)
def get_planner_llm():
    """Deprecated: Use get_lead_llm() instead"""
    return get_lead_llm()


def get_synthesizer_llm():
    """Deprecated: Use get_lead_llm() instead"""
    return get_lead_llm()


def get_verifier_llm():
    """Deprecated: Use get_lead_llm() instead"""
    return get_lead_llm()


def get_extraction_llm():
    """Deprecated: Use get_subagent_llm() instead"""
    return get_subagent_llm()
