"""LLM factory with lazy loading pattern"""

from langchain_openai import ChatOpenAI

# LLM Configuration - Lazy Loading Pattern
# LLMs are initialized on first use, not at module import time
# This allows tests to mock LLMs without needing API keys

_lead_llm = None
_subagent_llm = None
_planner_llm = None
_synthesizer_llm = None
_verifier_llm = None
_extraction_llm = None

# Import tracing for local logging (optional)
try:
    from tracing import get_callbacks
except ImportError:
    def get_callbacks():
        return []


def get_lead_llm():
    """Get or create the lead LLM instance (lazy loading)"""
    global _lead_llm
    if _lead_llm is None:
        _lead_llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.3,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _lead_llm


def get_subagent_llm():
    """Get or create the subagent LLM instance (lazy loading)"""
    global _subagent_llm
    if _subagent_llm is None:
        _subagent_llm = ChatOpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
            temperature=0.3,
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _subagent_llm


# Specialized Accessors with independent instances
def get_planner_llm():
    """LLM for high-level planning and task decomposition."""
    global _planner_llm
    if _planner_llm is None:
        from config import settings
        _planner_llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            model=settings.MODEL_PLANNER,
            temperature=settings.TEMP_PLANNER, # Creativity + Stability
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _planner_llm


def get_synthesizer_llm():
    """LLM for final report synthesis (high context)."""
    global _synthesizer_llm
    if _synthesizer_llm is None:
        from config import settings
        _synthesizer_llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            model=settings.MODEL_SYNTHESIZER,
            temperature=settings.TEMP_SYNTHESIZER, # Flow
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _synthesizer_llm


def get_verifier_llm():
    """LLM for rigorous fact-checking."""
    global _verifier_llm
    if _verifier_llm is None:
        from config import settings
        _verifier_llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            model=settings.MODEL_VERIFIER,
            temperature=settings.TEMP_VERIFIER, # Strictness
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _verifier_llm


def get_extraction_llm():
    """LLM for high-volume extraction tasks (citations, web content)."""
    global _extraction_llm
    if _extraction_llm is None:
        from config import settings
        # Use turbo for speed/cost on high-volume extract actions
        _extraction_llm = ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            model=settings.MODEL_EXTRACTOR,
            temperature=settings.TEMP_EXTRACTOR, # Determinism
            max_retries=2,
            callbacks=get_callbacks(),
        )
    return _extraction_llm

