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

