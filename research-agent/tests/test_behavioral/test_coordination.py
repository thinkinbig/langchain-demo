"""Test agent coordination patterns

Based on Anthropic: "Multi-agent systems have key differences from
single-agent systems, including a rapid growth in coordination complexity."
"""

from unittest.mock import patch

import pytest
from tests.test_helpers import (
    assert_complete_workflow,
    assert_content_relevance,
    assert_findings_structure,
    assert_synthesis_quality,
    assert_tasks_related_to_query,
    configure_structured_output_mock,
)  # noqa: E402
from schemas import ResearchTasks, SynthesisResult, SubagentOutput


@pytest.mark.behavioral
class TestCoordination:
    """Test coordination between agents"""

    @patch("tools.search_web")
    @patch("graph.get_lead_llm")
    @patch("graph.get_subagent_llm")
    def test_task_coverage(
        self,
        mock_subagent_llm,
        mock_lead_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
    ):
        """Test that tasks cover the query adequately"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(tasks=["Research Python features", "Research Rust features"]),
            SynthesisResult: SynthesisResult(
                summary="Mock synthesized results combining Python and Rust findings. "
                "This is a comprehensive summary."
            )
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(summary="Mock summary of findings from search results.")
        })

        state = {**initial_state, "query": "Compare Python and Rust for web development"}
        query = state["query"]

        final_state = app.invoke(state)

        # Should have tasks generated
        tasks = final_state.get("subagent_tasks", [])
        assert len(tasks) > 0, "Should generate tasks"

        # Tasks should be related to the query (semantic check, not length)
        assert_tasks_related_to_query(tasks, query)

    @patch("tools.search_web")
    @patch("graph.get_lead_llm")
    @patch("graph.get_subagent_llm")
    def test_findings_completeness(
        self,
        mock_subagent_llm,
        mock_lead_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
    ):
        """Test that findings cover the query"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(tasks=["Research microservices pros", "Research microservices cons"]),
            SynthesisResult: SynthesisResult(
                summary="Mock synthesized results about microservices pros and cons. "
                "This is a comprehensive summary covering all aspects."
            )
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(summary="Mock summary of microservices findings from search results.")
        })

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
    @patch("graph.get_lead_llm")
    @patch("graph.get_subagent_llm")
    def test_citation_quality(
        self,
        mock_subagent_llm,
        mock_lead_llm,
        mock_search,
        app,
        initial_state,
        mock_search_results,
    ):
        """Test that citations are properly extracted"""
        # Mock search
        mock_search.return_value = mock_search_results

        # Configure structured output mocks
        configure_structured_output_mock(mock_lead_llm, {
            ResearchTasks: ResearchTasks(tasks=["Research LangGraph features"]),
            SynthesisResult: SynthesisResult(
                summary="LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                "It provides graph-based workflow orchestration."
            )
        })
        configure_structured_output_mock(mock_subagent_llm, {
            SubagentOutput: SubagentOutput(summary="LangGraph provides graph-based orchestration for LLM applications.")
        })

        state = initial_state.copy()
        query = "What is LangGraph?"
        state["query"] = query

        final_state = app.invoke(state)

        # Check complete workflow (includes citations and final report)
        assert_complete_workflow(final_state)

        # Final report should be relevant to query
        report = final_state.get("final_report", "")
        assert_content_relevance(report, query, min_keyword_matches=1)
