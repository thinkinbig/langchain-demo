# Test Strategy Summary

## Mock vs Real API Usage

### ✅ Integration Tests (Use Real API)
**Location**: `tests/test_integration/`

- `test_simple_queries.py` - Uses real API
- `test_medium_queries.py` - Uses real API
- `test_complex_queries.py` - Uses real API
- `test_parallel_execution.py` - Uses real API

**How it works**: Each integration test file loads `.env` with `load_dotenv()` at the top.

### ✅ All Other Tests (Use Mocks)
**Location**: `tests/test_edge_cases/` and `tests/test_behavioral/`

- All edge case tests use `@patch` to mock:
  - `tools.search_web` - Mock search results
  - `graph.llm` - Mock LLM responses
- All behavioral tests use `@patch` to mock APIs

**How it works**: Tests use `@patch` decorators to mock API calls, no real API keys needed.

## Running Tests

```bash
# Run only integration tests (uses real API)
pytest research-agent/tests/test_integration/ -v

# Run only mocked tests (no API needed)
pytest research-agent/tests/test_edge_cases/ -v
pytest research-agent/tests/test_behavioral/ -v

# Run all tests (integration will use API, others use mocks)
pytest research-agent/tests/ -v

# Skip integration tests (all mocked)
pytest research-agent/tests/ -m "not integration"
```

## Mock Fixtures Available

In `conftest.py`:
- `mock_search_results` - Mock search results
- `mock_lead_researcher_response` - Mock LLM response for lead researcher
- `mock_subagent_llm_response` - Mock LLM response for subagents
- `mock_synthesizer_response` - Mock LLM response for synthesizer

## Example: Mocked Test

```python
@patch("tools.search_web")
@patch("graph.llm")
def test_something(mock_llm, mock_search, app, initial_state, mock_lead_researcher_response):
    # Mock search
    mock_search.return_value = mock_search_results
    
    # Mock LLM
    mock_llm.invoke.return_value = mock_lead_researcher_response
    
    # Run test - no real API calls!
    final_state = app.invoke(state)
    assert "final_report" in final_state
```

## Example: Integration Test

```python
from dotenv import load_dotenv
load_dotenv()  # Loads real API keys

def test_real_query(app, initial_state):
    # Uses real API - make sure .env has valid keys
    final_state = app.invoke(state)
    assert "final_report" in final_state
```

