"""Test handling of empty search results"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.edge_case
class TestEmptyResults:
    """Test behavior when search returns no results"""

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_empty_search_results(
        self, mock_llm, mock_search, app, initial_state, mock_lead_researcher_response
    ):
        """Test graceful handling of empty search results"""
        # Mock empty search results
        mock_search.return_value = []

        # Mock LLM responses
        mock_llm.invoke.return_value = mock_lead_researcher_response

        state = initial_state.copy()
        state["query"] = "Test query with no results"

        final_state = app.invoke(state)

        # Should still complete (graceful degradation)
        assert "final_report" in final_state

        # Should have findings even if empty
        findings = final_state.get("subagent_findings", [])
        # System should handle empty results gracefully
        assert isinstance(findings, list)

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_partial_empty_results(
        self, mock_llm, mock_search, app, initial_state, mock_lead_researcher_response
    ):
        """Test handling when some subagents get empty results"""
        # Mock: first call returns results, second returns empty
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {
                        "title": "Result 1",
                        "url": "https://example.com",
                        "content": "Content",
                    },
                ]
            return []

        mock_search.side_effect = side_effect

        # Mock LLM responses
        mock_response = MagicMock()
        mock_response.content = "Mock summary"
        mock_llm.invoke.return_value = mock_response

        state = initial_state.copy()
        state["query"] = "Test query with partial results"

        final_state = app.invoke(state)

        # Should still complete
        assert "final_report" in final_state

        # Should handle partial results
        findings = final_state.get("subagent_findings", [])
        assert len(findings) > 0, "Should have at least some findings"

