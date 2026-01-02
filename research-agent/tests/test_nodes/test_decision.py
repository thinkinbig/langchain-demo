"""Tests for decision node"""

from nodes.decision import decision_node


def test_decision_node_continue_few_findings(initial_state):
    """Test decision to continue (few findings)"""
    state = initial_state.copy()
    state["iteration_count"] = 1
    state["subagent_findings"] = []  # Few findings
    state["synthesized_results"] = "Short"

    result = decision_node(state)
    assert result["needs_more_research"] is True


def test_decision_node_continue_short_synthesis(initial_state):
    """Test decision to continue (short synthesis)"""
    state = initial_state.copy()
    state["iteration_count"] = 1
    state["subagent_findings"] = [{"task": "T1", "summary": "S1"}]
    state["synthesized_results"] = "Short"  # Less than 500 chars

    result = decision_node(state)
    assert result["needs_more_research"] is True


def test_decision_node_continue_citations_found(initial_state):
    """Test decision to continue (citations found)"""
    state = initial_state.copy()
    state["iteration_count"] = 1
    state["subagent_findings"] = [{"task": "T1", "summary": "S1"}]
    state["synthesized_results"] = "Some results"
    state["all_extracted_citations"] = [{"title": "Paper 1"}]

    result = decision_node(state)
    assert result["needs_more_research"] is True


def test_decision_node_stop_max_iterations(initial_state):
    """Test decision to stop (max iterations)"""
    state = initial_state.copy()
    state["iteration_count"] = 3

    result = decision_node(state)
    assert result["needs_more_research"] is False


def test_decision_node_stop_sufficient_findings(initial_state):
    """Test decision to stop (sufficient findings)"""
    state = initial_state.copy()
    state["iteration_count"] = 1
    state["subagent_findings"] = [
        {"task": "T1", "summary": "S1"},
        {"task": "T2", "summary": "S2"},
        {"task": "T3", "summary": "S3"},
    ]
    state["synthesized_results"] = "A" * 500  # Long enough

    result = decision_node(state)
    assert result["needs_more_research"] is False

