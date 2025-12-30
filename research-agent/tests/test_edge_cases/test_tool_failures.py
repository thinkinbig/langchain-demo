"""Test handling of tool failures"""

from unittest.mock import patch

import pytest


@pytest.mark.edge_case
class TestToolFailures:
    """Test error handling for tool failures"""

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_search_tool_failure(
        self, mock_llm, mock_search, app, initial_state, mock_lead_researcher_response
    ):
        """Test handling of search tool failures"""
        # Mock search failure
        mock_search.side_effect = Exception("Search API error")

        # Mock LLM responses
        mock_llm.invoke.return_value = mock_lead_researcher_response

        state = initial_state.copy()
        state["query"] = "Test query with search failure"

        # Should not crash, should handle gracefully
        final_state = app.invoke(state)

        # Should still produce some output (graceful degradation)
        assert "final_report" in final_state

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_api_key_missing(
        self, mock_llm, mock_search, app, initial_state, mock_lead_researcher_response
    ):
        """Test handling of missing API key"""
        # Mock search to simulate API key error
        mock_search.side_effect = ValueError(
            "TAVILY_API_KEY environment variable not set"
        )

        # Mock LLM responses
        mock_llm.invoke.return_value = mock_lead_researcher_response

        state = initial_state.copy()
        state["query"] = "Test query"

        # Should handle missing API key gracefully
        final_state = app.invoke(state)
        assert "final_report" in final_state

