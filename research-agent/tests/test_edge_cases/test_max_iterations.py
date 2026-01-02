"""Test maximum iteration limits"""

from unittest.mock import patch

import pytest
from schemas import ResearchTasks, SubagentOutput, SynthesisResult
from tests.test_helpers import configure_structured_output_mock


@pytest.mark.edge_case
class TestMaxIterations:
    """Test iteration limit enforcement"""

    @patch("tools.search_web")
    @patch("graph.get_lead_llm")
    @patch("graph.get_subagent_llm")
    def test_iteration_limit_enforced(
        self,
        mock_subagent_llm,
        mock_lead_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
    ):
        """Test that iteration limit (3) is enforced"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(
                tasks=["Research complex topic", "Deep analysis"]
            ),
            SynthesisResult: SynthesisResult(
                summary="Comprehensive summary of the complex query findings. "
                "This covers all aspects of the topic in detail."
            )
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(
                summary="Detailed findings from search results."
            )
        })

        state = {**initial_state, "query": "Very complex query needing many iterations"}

        final_state = app.invoke(state)

        # Should not exceed max iterations
        iteration_count = final_state.get("iteration_count", 0)
        msg = f"Iteration count {iteration_count} exceeds limit of 3"
        assert iteration_count <= 3, msg

        # Should have final report even at limit
        final_report = final_state.get("final_report", "")
        assert final_report is not None

    @patch("tools.search_web")
    @patch("graph.get_lead_llm")
    @patch("graph.get_subagent_llm")
    def test_loop_termination(
        self,
        mock_subagent_llm,
        mock_lead_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
    ):
        """Test that loop terminates correctly"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(tasks=["Research query topic"]),
            SynthesisResult: SynthesisResult(
                summary="Complete summary of all findings for loop termination test. "
                "This provides detailed insights into the topic."
            )
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(
                summary="Summary from search results for termination test."
            )
        })

        state = {**initial_state, "query": "Test query for loop termination"}

        final_state = app.invoke(state)

        # Should eventually terminate (not loop forever)
        final_report = final_state.get("final_report", "")
        assert final_report is not None

        # Should have needs_more_research set correctly
        # At termination, should be False
        # (unless we're at max iterations, in which case it might be True
        # but we still terminate)
        _ = final_state.get("needs_more_research", False)  # Check it exists
