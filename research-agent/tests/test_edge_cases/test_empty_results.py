"""Test handling of empty search results"""

from unittest.mock import patch

import pytest
from schemas import ResearchTasks, SubagentOutput, SynthesisResult
from tests.test_utils import configure_structured_output_mock


@pytest.mark.edge_case
class TestEmptyResults:
    """Test behavior when search returns no results"""

    @patch("tools.search_web")
    @patch("llm.factory.get_lead_llm")
    @patch("llm.factory.get_subagent_llm")
    def test_empty_search_results(
        self, mock_subagent_llm, mock_lead_llm, mock_search, app, initial_state
    ):
        """Test graceful handling of empty search results"""
        # Mock empty search results
        mock_search.return_value = []

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(tasks=["Research task 1"]),
            SynthesisResult: SynthesisResult(summary="No findings to synthesize.")
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(summary="No information found")
        })

        # Create new state dict (LangGraph accepts dict and converts to Pydantic)
        state = {
            **initial_state,
            "query": "Test query with no results",
        }

        final_state = app.invoke(state)

        # Should still complete (graceful degradation)
        final_report = final_state.get("final_report", "")
        assert final_report is not None

        # Should have findings even if empty
        findings = final_state.get("subagent_findings", [])
        # System should handle empty results gracefully
        assert isinstance(findings, list)

    @patch("tools.search_web")
    @patch("llm.factory.get_lead_llm")
    @patch("llm.factory.get_subagent_llm")
    def test_partial_empty_results(
        self, mock_subagent_llm, mock_lead_llm, mock_search, app, initial_state
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

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(tasks=["Research task 1", "Research task 2"]),
            SynthesisResult: SynthesisResult(summary="Partial findings synthesized.")
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(summary="Mock summary from search results")
        })

        # Create new state dict
        state = {
            **initial_state,
            "query": "Test query with partial results",
        }

        final_state = app.invoke(state)

        # Should still complete
        final_report = final_state.get("final_report", "")
        assert final_report is not None

        # Should handle partial results
        findings = final_state.get("subagent_findings", [])
        assert len(findings) > 0, "Should have at least some findings"
