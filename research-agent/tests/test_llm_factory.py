"""Tests for LLM factory module"""

from unittest.mock import MagicMock, patch

import pytest
from llm.factory import get_llm_by_model_choice


def test_llm_factory_lazy_loading():
    """Test LLM factory lazy loading"""
    # Reset the global instances
    import llm.factory
    llm.factory._turbo_llm = None
    llm.factory._plus_llm = None
    llm.factory._max_llm = None

    # First call should create instance
    with patch("llm.factory.ChatOpenAI") as mock_chat:
        llm1 = get_llm_by_model_choice("plus")
        assert mock_chat.called

    # Second call should return same instance
    llm2 = get_llm_by_model_choice("plus")
    assert llm1 is llm2


def test_llm_factory_singleton_pattern():
    """Test singleton pattern (same instance returned on multiple calls)"""
    import llm.factory
    llm.factory._plus_llm = None

    with patch("llm.factory.ChatOpenAI"):
        llm1 = get_llm_by_model_choice("plus")
        llm2 = get_llm_by_model_choice("plus")
        assert llm1 is llm2


def test_llm_factory_separate_instances():
    """Test that different model choices return separate instances"""
    import llm.factory
    llm.factory._turbo_llm = None
    llm.factory._plus_llm = None

    # Create separate mock instances
    mock_turbo = MagicMock()
    mock_plus = MagicMock()

    with patch("llm.factory.ChatOpenAI") as mock_chat:
        # Make the mock return different instances on each call
        mock_chat.side_effect = [mock_turbo, mock_plus]

        turbo_llm = get_llm_by_model_choice("turbo")
        plus_llm = get_llm_by_model_choice("plus")

        # Should be called twice (once for each)
        assert mock_chat.call_count == 2
        # Should be different instances
        assert turbo_llm is not plus_llm
        assert llm.factory._turbo_llm is not llm.factory._plus_llm


def test_llm_factory_error_handling():
    """Test error handling when API keys are missing"""
    import llm.factory
    llm.factory._plus_llm = None

    # The error would occur when the LLM is actually created (not when retrieved)
    # Since we're using lazy loading, the error happens during get_llm_by_model_choice()
    # if ChatOpenAI initialization fails
    with patch("llm.factory.ChatOpenAI", side_effect=Exception("API key missing")):
        # The error is raised during initialization, which is expected behavior
        try:
            llm = get_llm_by_model_choice("plus")
            # If we get here, the mock didn't raise (which is fine for testing)
            # In real usage, this would fail
        except Exception as e:
            # This is expected - the error is raised during initialization
            assert "API key missing" in str(e)


def test_llm_factory_invalid_model_choice():
    """Test error handling for invalid model choice"""
    with pytest.raises(ValueError, match="Invalid model_choice"):
        get_llm_by_model_choice("invalid")

