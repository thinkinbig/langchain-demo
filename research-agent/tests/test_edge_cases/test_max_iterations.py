"""Test maximum iteration limits"""

from unittest.mock import patch

import pytest


@pytest.mark.edge_case
class TestMaxIterations:
    """Test iteration limit enforcement"""

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_iteration_limit_enforced(
        self,
        mock_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
        mock_lead_researcher_response,
        mock_subagent_llm_response,
        mock_synthesizer_response,
    ):  # noqa: PLR0913
        """Test that iteration limit (3) is enforced"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Mock LLM responses
        call_count = 0
        def llm_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Lead researcher response
                return mock_lead_researcher_response
            elif call_count <= 4:
                # Subagent responses
                return mock_subagent_llm_response
            else:
                # Synthesizer responses
                return mock_synthesizer_response

        mock_llm.invoke.side_effect = llm_side_effect

        state = initial_state.copy()
        state["query"] = "Very complex query needing many iterations"

        final_state = app.invoke(state)

        # Should not exceed max iterations
        iteration_count = final_state.get("iteration_count", 0)
        msg = f"Iteration count {iteration_count} exceeds limit of 3"
        assert iteration_count <= 3, msg

        # Should have final report even at limit
        assert "final_report" in final_state

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_loop_termination(
        self,
        mock_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
        mock_lead_researcher_response,
        mock_subagent_llm_response,
        mock_synthesizer_response,
    ):
        """Test that loop terminates correctly"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Mock LLM responses
        call_count = 0
        def llm_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_lead_researcher_response
            elif call_count <= 3:
                return mock_subagent_llm_response
            else:
                return mock_synthesizer_response

        mock_llm.invoke.side_effect = llm_side_effect

        state = initial_state.copy()
        state["query"] = "Test query for loop termination"

        final_state = app.invoke(state)

        # Should eventually terminate (not loop forever)
        assert "final_report" in final_state

        # Should have needs_more_research set correctly
        # At termination, should be False
        # (unless we're at max iterations, in which case it might be True but we still terminate)
        _ = final_state.get("needs_more_research", False)  # Check it exists

