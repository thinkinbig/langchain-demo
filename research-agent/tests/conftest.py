"""Pytest fixtures and shared utilities"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

# ==============================================================================
# MOCK MISSING DEPENDENCIES (Global Patching)
# ==============================================================================
# This is necessary because the test environment lacks some production dependencies
# (langchain_chroma, langgraph). We mock them here so pytest can collect tests.

# Mock langgraph
if "langgraph" not in sys.modules:
    sys.modules["langgraph"] = MagicMock()
if "langgraph.graph" not in sys.modules:
    sys.modules["langgraph.graph"] = MagicMock()
if "langgraph.checkpoint.memory" not in sys.modules:
    sys.modules["langgraph.checkpoint.memory"] = MagicMock()
if "langgraph.types" not in sys.modules:
    sys.modules["langgraph.types"] = MagicMock()

# Mock langchain_chroma
if "langchain_chroma" not in sys.modules:
    sys.modules["langchain_chroma"] = MagicMock()

# Mock langchain_huggingface
if "langchain_huggingface" not in sys.modules:
    sys.modules["langchain_huggingface"] = MagicMock()

# Mock langchain_community (often needed)
if "langchain_community" not in sys.modules:
    sys.modules["langchain_community"] = MagicMock()
if "langchain_community.vectorstores" not in sys.modules:
    sys.modules["langchain_community.vectorstores"] = MagicMock()

# ==============================================================================

# Only load env for integration tests
# Other tests should use mocks
@pytest.fixture(scope="session", autouse=True)
def load_env_for_integration(request):
    """Load environment variables only for integration tests"""
    # Check if we're running integration tests
    if "test_integration" in str(request.node):
        load_dotenv()
    # Also check command line args
    import sys
    if any("test_integration" in arg for arg in sys.argv):
        load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from schemas import ResearchState  # noqa: E402


@pytest.fixture
def initial_state() -> ResearchState:
    """Standard initial state for tests"""
    # LangGraph accepts dict and converts to Pydantic model automatically
    # But we can also return dict directly - LangGraph handles the conversion
    return {
        "query": "test query",
        "research_plan": "",
        "subagent_tasks": [],
        "subagent_findings": [],
        "iteration_count": 0,
        "needs_more_research": False,
        "synthesized_results": "",
        "citations": [],
        "final_report": "",
        "error": None,
        "retry_count": 0,
    }


@pytest.fixture
def mock_search_results():
    """Mock search results for testing"""
    return [
        {
            "title": "Test Result 1",
            "url": "https://example.com/1",
            "content": (
                "This is test content for result 1. "
                "It contains relevant information about the topic."
            ),
        },
        {
            "title": "Test Result 2",
            "url": "https://example.com/2",
            "content": (
                "This is test content for result 2. "
                "It provides additional context and details."
            ),
        },
        {
            "title": "Test Result 3",
            "url": "https://example.com/3",
            "content": (
                "This is test content for result 3. "
                "It offers a different perspective on the topic."
            ),
        },
    ]


@pytest.fixture
def app():
    """Research agent app instance (uses real API)"""
    from graph import app
    return app


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    mock_response = MagicMock()
    mock_response.content = "Mock LLM response content"
    return mock_response


@pytest.fixture
def mock_app_with_mocks(mock_search_results, mock_llm_response):
    """App with mocked LLM and search - for non-integration tests"""
    # Create app
    from graph import app
    from schemas import ResearchTasks, SubagentOutput, SynthesisResult

    # Helper to create structured output mock based on schema
    def create_structured_output_mock(schema_class, **kwargs):
        """Create a mock that returns the proper include_raw=True format"""
        mock_structured = MagicMock()

        if schema_class == ResearchTasks:
            parsed = ResearchTasks(tasks=["Research Task 1", "Research Task 2"])
        elif schema_class == SynthesisResult:
            parsed = SynthesisResult(
                summary="Mock synthesized results combining all findings. "
                "This is a comprehensive summary that integrates multiple research "
                "findings into a coherent answer. It covers all aspects of the query "
                "and provides detailed insights based on the collected information."
            )
        elif schema_class == SubagentOutput:
            msg = "Mock summary of findings from search results."
            parsed = SubagentOutput(summary=msg)
        else:
            parsed = MagicMock()

        mock_structured.invoke.return_value = {
            "parsed": parsed,
            "parsing_error": None,
            "raw": MagicMock()
        }
        return mock_structured

    # Create mock LLM objects
    mock_lead_llm = MagicMock()
    mock_lead_llm.invoke.return_value = mock_llm_response
    mock_lead_llm.with_structured_output.side_effect = create_structured_output_mock

    mock_subagent_llm = MagicMock()
    mock_subagent_llm.invoke.return_value = mock_llm_response
    mock_subagent_llm.with_structured_output.side_effect = create_structured_output_mock

    # Mock the getter function to return our mock LLMs based on model choice
    def mock_get_llm_by_model_choice(model_choice):
        if model_choice == "plus":
            return mock_lead_llm
        elif model_choice == "turbo":
            return mock_subagent_llm
        else:
            return mock_lead_llm  # Default fallback

    with patch("llm.factory.get_llm_by_model_choice", side_effect=mock_get_llm_by_model_choice):

        # Mock search
        with patch("tools.search_web") as mock_search:
            mock_search.return_value = mock_search_results

            yield app


@pytest.fixture
def mock_lead_researcher_response():
    """Mock response for lead researcher"""
    mock_response = MagicMock()
    mock_response.content = '["Research Python features", "Research Rust features"]'
    return mock_response


@pytest.fixture
def mock_subagent_llm_response():
    """Mock LLM response for subagent analysis"""
    mock_response = MagicMock()
    mock_response.content = "Mock summary of findings from search results."
    return mock_response


@pytest.fixture
def mock_synthesizer_response():
    """Mock LLM response for synthesizer"""
    mock_response = MagicMock()
    mock_response.content = (
        "Mock synthesized results combining all findings. "
        "This is a comprehensive summary that integrates multiple research "
        "findings into a coherent answer. It covers all aspects of the query "
        "and provides detailed insights based on the collected information. "
        "The synthesis demonstrates thorough analysis and integration of "
        "diverse sources and perspectives."
    )
    return mock_response

