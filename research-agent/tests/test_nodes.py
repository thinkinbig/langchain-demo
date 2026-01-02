
from unittest.mock import MagicMock, patch

# Import nodes to test
from graph import (
    citation_agent_node,
    decision_node,
    lead_researcher_node,
    subagent_node,
    synthesizer_node,
)
from schemas import Finding, ResearchTasks, SynthesisResult


# 1. Test Lead Researcher Node
def test_lead_researcher_node_initial(initial_state):
    """Test lead researcher generating initial plan"""
    from schemas import ResearchTask

    # Mock LLM response
    mock_tasks = ResearchTasks(
        tasks=[
            ResearchTask(id="task_1", description="Task A"),
            ResearchTask(id="task_2", description="Task B")
        ]
    )

    # Mock the LLM getter and context manager
    with patch("graph.get_lead_llm") as mock_get_llm:
        with patch("graph.context_manager.retrieve_knowledge", return_value=("", [])):
            # Create mock chain
            mock_llm = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = {
                "parsed": mock_tasks,
                "parsing_error": None
            }
            mock_llm.with_structured_output.return_value = mock_structured
            mock_get_llm.return_value = mock_llm

            # Run node
            result = lead_researcher_node(initial_state)

            # Assertions
            assert "research_plan" in result
            assert "subagent_tasks" in result
            assert len(result["subagent_tasks"]) == 2
            assert result["iteration_count"] == 1

def test_lead_researcher_retry_logic(initial_state):
    """Test lead researcher handling validation error and retry"""
    from schemas import ResearchTask

    # Simulate a state that has an error from previous attempt
    state = initial_state.copy()
    state["error"] = "Invalid JSON"
    state["retry_count"] = 1

    with patch("graph.get_lead_llm") as mock_get_llm:
        with patch("graph.context_manager.retrieve_knowledge", return_value=("", [])):
            mock_llm = MagicMock()
            mock_structured = MagicMock()

            # This time it succeeds
            mock_tasks = ResearchTasks(
                tasks=[ResearchTask(id="task_1", description="Task Retry")]
            )
            mock_structured.invoke.return_value = {
                "parsed": mock_tasks,
                "parsing_error": None
            }

            mock_llm.with_structured_output.return_value = mock_structured
            mock_get_llm.return_value = mock_llm

            result = lead_researcher_node(state)

            assert len(result["subagent_tasks"]) == 1
            assert result["error"] is None
            assert result["retry_count"] == 0

# 2. Test Subagent Node
def test_subagent_node_success():
    """Test subagent performing search and summarization"""
    from retrieval import RetrievalResult, RetrievalSource, Source
    from schemas import ResearchTask

    task = ResearchTask(id="task_1", description="Research Task 1")
    state = {"subagent_tasks": [task]}

    # Mock retrieval results
    mock_internal_result = RetrievalResult(
        content="",
        sources=[],
        source_type=RetrievalSource.INTERNAL,
        has_content=False
    )
    mock_web_result = RetrievalResult(
        content="Web content",
        sources=[Source(identifier="u1", title="T1", source_type=RetrievalSource.WEB)],
        source_type=RetrievalSource.WEB,
        has_content=True
    )

    # Mock LLM tool calls

    with patch("graph.RetrievalService.retrieve_internal", return_value=mock_internal_result):
        with patch("graph.RetrievalService.retrieve_web", return_value=mock_web_result):
            with patch("graph.get_subagent_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_response = MagicMock()
                mock_response.tool_calls = [{
                    "name": "submit_findings",
                    "args": {"summary": "Summary 1", "sources": [{"title": "T1", "url": "u1"}]},
                    "id": "call_1"
                }]
                mock_llm.bind_tools.return_value.invoke.return_value = mock_response
                mock_get_llm.return_value = mock_llm

                result = subagent_node(state)

                # Verify finding structure
                assert "subagent_findings" in result
                findings = result["subagent_findings"]
                assert len(findings) == 1
                assert findings[0].summary == "Summary 1"
                assert len(findings[0].sources) > 0

def test_subagent_node_empty_search():
    """Test subagent handling empty search results"""
    from retrieval import RetrievalResult, RetrievalSource
    from schemas import ResearchTask

    task = ResearchTask(id="task_1", description="Empty Task")
    state = {"subagent_tasks": [task]}

    mock_internal_result = RetrievalResult(
        content="",
        sources=[],
        source_type=RetrievalSource.INTERNAL,
        has_content=False
    )
    mock_web_result = RetrievalResult(
        content="",
        sources=[],
        source_type=RetrievalSource.WEB,
        has_content=False
    )

    with patch("graph.RetrievalService.retrieve_internal", return_value=mock_internal_result):
        with patch("graph.RetrievalService.retrieve_web", return_value=mock_web_result):
            with patch("graph.get_subagent_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_response = MagicMock()
                mock_response.tool_calls = [{
                    "name": "submit_findings",
                    "args": {"summary": "No information found", "sources": []},
                    "id": "call_1"
                }]
                mock_llm.bind_tools.return_value.invoke.return_value = mock_response
                mock_get_llm.return_value = mock_llm

                result = subagent_node(state)

                assert len(result["subagent_findings"]) == 1
                assert "No information" in result["subagent_findings"][0].summary.lower()

# 3. Test Synthesizer Node
def test_synthesizer_node_success(initial_state):
    """Test synthesization of findings"""
    state = initial_state.copy()
    state["subagent_findings"] = [
        Finding(task="T1", summary="S1", sources=[])
    ]

    mock_synthesis = SynthesisResult(summary="Final Answer")

    with patch("graph.get_lead_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {
            "parsed": mock_synthesis,
            "parsing_error": None
        }
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        result = synthesizer_node(state)

        assert result["synthesized_results"] == "Final Answer"

# 4. Test Decision Node
def test_decision_node_continue(initial_state):
    """Test decision to continue research"""
    state = initial_state.copy()
    state["iteration_count"] = 1
    state["subagent_findings"] = [] # Few findings
    state["synthesized_results"] = "Short"

    result = decision_node(state)
    assert result["needs_more_research"] is True

def test_decision_node_stop_max_iterations(initial_state):
    """Test decision to stop at max iterations"""
    state = initial_state.copy()
    state["iteration_count"] = 3

    result = decision_node(state)
    assert result["needs_more_research"] is False

# 5. Test Citation Agent
def test_citation_agent_node(initial_state):
    """Test citation extraction"""
    state = initial_state.copy()
    state["synthesized_results"] = "Final Answer"
    state["subagent_findings"] = [
        Finding(
            task="T1",
            summary="S1",
            sources=[{"title": "Source 1", "url": "http://example.com"}]
        )
    ]

    result = citation_agent_node(state)

    assert "final_report" in result
    assert "Research Report" in result["final_report"]
    assert "http://example.com" in result["final_report"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].url == "http://example.com"
