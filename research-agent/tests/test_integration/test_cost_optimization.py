from unittest.mock import MagicMock, patch

import pytest
from nodes.subagent.analysis import analysis_node
from schemas import AnalysisOutput, Citation, SubagentState


@pytest.fixture
def mock_llm():
    with patch("nodes.subagent.analysis.get_llm_by_model_choice") as mock_get_llm:
        mock_chat = MagicMock()
        def mock_get_llm_side_effect(model_choice):
            if model_choice == "turbo":
                return mock_chat
            return mock_chat
        mock_get_llm.side_effect = mock_get_llm_side_effect

        # Mock structured output bound model
        mock_structured = MagicMock()
        mock_chat.with_structured_output.return_value = mock_structured

        yield mock_structured

def test_analysis_simple_flow(mock_llm):
    """Test standard flow: Single analysis call, no python code"""

    # Setup Mock Response
    mock_output = {
        "parsed": AnalysisOutput(
            summary="Rust memory safety features prevent data races.",
            citations=[
                Citation(title="Rust Manual", context="Safety", relevance="High")
            ],
            python_code=None,
            reasoning="Analyzed text directly."
        )
    }
    mock_llm.invoke.return_value = mock_output

    # Setup State
    state = SubagentState(
        subagent_tasks=[{"id": "1", "description": "Analyze Rust safety"}],
        task_description="Analyze Rust safety",
        subagent_findings=[],
        visited_sources=[]
    )

    # Run Node
    result = analysis_node(state)

    # Verify Logic
    assert mock_llm.invoke.call_count == 1

    findings = result["subagent_findings"]
    assert len(findings) == 1
    assert findings[0].summary == "Rust memory safety features prevent data races."
    assert len(findings[0].extracted_citations) == 1

def test_analysis_with_code_execution(mock_llm):
    """Test flow with Python code: Analyze -> Execute -> Refine"""

    # Setup Mock Responses for TWO calls
    # Call 1: Returns code
    response_1 = {
        "parsed": AnalysisOutput(
            summary="Calculating average...",
            citations=[],
            python_code="print(10 + 20)",
            reasoning="Need to calc"
        )
    }

    # Call 2: Returns final summary (Refine step)
    response_2 = {
        "parsed": AnalysisOutput(
            summary="The average is 30.",
            citations=[],
            python_code=None,
            reasoning="Calculation complete"
        )
    }

    mock_llm.invoke.side_effect = [response_1, response_2]

    # Setup State
    state = SubagentState(
        subagent_tasks=[{"id": "1", "description": "Calc value"}],
        task_description="Calc value",
        subagent_findings=[],
        visited_sources=[]
    )

    # Run Node
    # We strip 'tools.python_repl' call to avoid actual exec issues if
    # environment weird, but analysis_node uses real tools.python_repl.
    # We trust it prints to stdout and returns string. 10+20 is safe.

    with patch("tools.python_repl", return_value="30") as mock_tool:
        result = analysis_node(state)

        # Verify
        assert mock_llm.invoke.call_count == 2
        assert mock_tool.call_count == 1

        findings = result["subagent_findings"]
        assert findings[0].summary == "The average is 30."

def test_analysis_validation_retry(mock_llm):
    """Test validation retry logic (Error Prompt Pattern)"""

    # Call 1: Validation Error
    response_1 = {"parsing_error": "Missing summary field"}
    # Call 2: Success
    response_2 = {
        "parsed": AnalysisOutput(
            summary="Recovered summary",
            citations=[],
            reasoning="Fixed format"
        )
    }

    mock_llm.invoke.side_effect = [response_1, response_2]

    state = SubagentState(
        subagent_tasks=[{"id": "1", "description": "Test Retry"}],
        task_description="Test Retry",
        subagent_findings=[],
        visited_sources=[]
    )

    result = analysis_node(state)

    # Verify we retried
    assert mock_llm.invoke.call_count == 2

    # Check that error prompt was added in second call
    call_args_2 = mock_llm.invoke.call_args_list[1]
    messages = call_args_2[0][0] # First arg, messages list
    assert hasattr(messages[-1], "content")
    assert "failed validation" in messages[-1].content

    findings = result["subagent_findings"]
    assert findings[0].summary == "Recovered summary"
