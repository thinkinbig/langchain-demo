"""Test agent coordination patterns

Based on Anthropic: "Multi-agent systems have key differences from
single-agent systems, including a rapid growth in coordination complexity."
"""

from unittest.mock import patch

import pytest
from test_helpers import (
    assert_complete_workflow,
    assert_content_relevance,
    assert_findings_structure,
    assert_synthesis_quality,
    assert_tasks_related_to_query,
)  # noqa: E402


@pytest.mark.behavioral
class TestCoordination:
    """Test coordination between agents"""

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_task_coverage(
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
        """Test that tasks cover the query adequately"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Mock LLM - first call returns tasks, subsequent calls return summaries
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
        query = "Compare Python and Rust for web development"
        state["query"] = query

        final_state = app.invoke(state)

        # Should have tasks generated
        tasks = final_state.get("subagent_tasks", [])
        assert len(tasks) > 0, "Should generate tasks"

        # Tasks should be related to the query (semantic check, not length)
        assert_tasks_related_to_query(tasks, query)

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_findings_completeness(
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
        """Test that findings cover the query"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Mock LLM - handle multiple calls correctly
        # Call order: LeadResearcher -> Subagent1 -> Subagent2 -> Synthesizer
        call_count = 0
        def llm_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call: LeadResearcher generates tasks
            if call_count == 1:
                return mock_lead_researcher_response
            # Next 2 calls: Subagents analyze results
            elif call_count <= 3:
                return mock_subagent_llm_response
            # Final call: Synthesizer combines findings
            else:
                return mock_synthesizer_response

        mock_llm.invoke.side_effect = llm_side_effect

        state = initial_state.copy()
        query = "Research pros and cons of microservices"
        state["query"] = query

        final_state = app.invoke(state)

        # Should have findings with proper structure
        findings = final_state.get("subagent_findings", [])
        assert len(findings) > 0, "Should have findings"
        assert_findings_structure(findings)

        # Should have synthesis that is relevant and meaningful
        synthesis = final_state.get("synthesized_results", "")
        assert_synthesis_quality(synthesis, query)

    @patch("tools.search_web")
    @patch("graph.llm")
    def test_citation_quality(
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
        """Test that citations are properly extracted"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Mock LLM
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
        query = "What is LangGraph?"
        state["query"] = query

        final_state = app.invoke(state)

        # Check complete workflow (includes citations and final report)
        assert_complete_workflow(final_state)

        # Final report should be relevant to query
        report = final_state.get("final_report", "")
        assert_content_relevance(report, query, min_keyword_matches=1)

