"""Integration tests for simple queries

Based on Anthropic's principle: "Simple fact-finding requires just 1 agent with 3-10 tool calls"

NOTE: These tests use REAL API calls. Make sure you have valid API keys in .env file.
"""

import pytest
from dotenv import load_dotenv

# Load environment variables for integration tests
load_dotenv()

from tests.test_helpers import (  # noqa: E402
    assert_complete_workflow,
    assert_content_relevance,
    assert_iteration_reasonable,
)


@pytest.mark.integration
class TestSimpleQueries:
    """Test simple fact-finding queries"""

    def test_what_is_langgraph(self, app, initial_state):
        """Test: What is LangGraph?"""
        state = {**initial_state, "query": "What is LangGraph?"}
        query = state["query"]

        final_state = app.invoke(state)

        # End-state evaluation: Check that workflow completed successfully
        assert_complete_workflow(final_state)

        # Check that final report is relevant to query
        report = final_state.get("final_report", "")
        assert_content_relevance(report, query, min_keyword_matches=1)

        # Iteration count should be reasonable (not too many)
        iteration_count = final_state.get("iteration_count", 0)
        assert_iteration_reasonable(iteration_count, max_iterations=5)

        # V2 Check: Even simple queries should use structured Task objects
        tasks = final_state.get("subagent_tasks", [])
        if tasks:
            assert hasattr(tasks[0], 'id'), "Tasks should be structured"

    def test_who_created_python(self, app, initial_state):
        """Test: Who created Python?"""
        state = {**initial_state, "query": "Who created Python?"}
        query = state["query"]

        final_state = app.invoke(state)

        # End-state evaluation: Check workflow completion and relevance
        assert_complete_workflow(final_state)

        report = final_state.get("final_report", "")
        assert_content_relevance(report, query, min_keyword_matches=1)

        # Check iteration count is reasonable
        iteration_count = final_state.get("iteration_count", 0)
        assert_iteration_reasonable(iteration_count, max_iterations=5)

    def test_capital_of_france(self, app, initial_state):
        """Test: What is the capital of France?"""
        state = {**initial_state, "query": "What is the capital of France?"}
        query = state["query"]

        final_state = app.invoke(state)

        # End-state evaluation: Check workflow completion and relevance
        assert_complete_workflow(final_state)

        report = final_state.get("final_report", "")
        assert_content_relevance(report, query, min_keyword_matches=1)

