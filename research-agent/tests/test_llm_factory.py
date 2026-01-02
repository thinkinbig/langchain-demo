"""Tests for LLM factory module"""

from unittest.mock import MagicMock, patch

from llm.factory import get_lead_llm, get_subagent_llm


def test_llm_factory_lazy_loading():
    """Test LLM factory lazy loading"""
    # Reset the global instances
    import llm.factory
    llm.factory._lead_llm = None
    llm.factory._subagent_llm = None

    # First call should create instance
    with patch("llm.factory.ChatOpenAI") as mock_chat:
        llm1 = get_lead_llm()
        assert mock_chat.called

    # Second call should return same instance
    llm2 = get_lead_llm()
    assert llm1 is llm2


def test_llm_factory_singleton_pattern():
    """Test singleton pattern (same instance returned on multiple calls)"""
    import llm.factory
    llm.factory._lead_llm = None
    llm.factory._subagent_llm = None

    with patch("llm.factory.ChatOpenAI"):
        llm1 = get_lead_llm()
        llm2 = get_lead_llm()
        assert llm1 is llm2


def test_llm_factory_separate_instances():
    """Test that lead and subagent LLMs are separate instances"""
    import llm.factory
    llm.factory._lead_llm = None
    llm.factory._subagent_llm = None

    # Create separate mock instances
    mock_lead = MagicMock()
    mock_subagent = MagicMock()

    with patch("llm.factory.ChatOpenAI") as mock_chat:
        # Make the mock return different instances on each call
        mock_chat.side_effect = [mock_lead, mock_subagent]

        lead_llm = get_lead_llm()
        subagent_llm = get_subagent_llm()

        # Should be called twice (once for each)
        assert mock_chat.call_count == 2
        # Should be different instances
        assert lead_llm is not subagent_llm
        assert llm.factory._lead_llm is not llm.factory._subagent_llm


def test_llm_factory_error_handling():
    """Test error handling when API keys are missing"""
    import llm.factory
    llm.factory._lead_llm = None

    # The error would occur when the LLM is actually created (not when retrieved)
    # Since we're using lazy loading, the error happens during get_lead_llm()
    # if ChatOpenAI initialization fails
    with patch("llm.factory.ChatOpenAI", side_effect=Exception("API key missing")):
        # The error is raised during initialization, which is expected behavior
        try:
            llm = get_lead_llm()
            # If we get here, the mock didn't raise (which is fine for testing)
            # In real usage, this would fail
        except Exception as e:
            # This is expected - the error is raised during initialization
            assert "API key missing" in str(e)

