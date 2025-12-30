# Research Agent Test Suite

Comprehensive pytest test suite based on [Anthropic's Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) principles.

## Test Philosophy

Following Anthropic's approach: **End-state evaluation** - We test whether the system achieves the correct final outcome, not whether it followed a specific process.

## Running Tests

```bash
# Run all tests
pytest research-agent/tests/

# Run specific category
pytest research-agent/tests/test_integration/
pytest research-agent/tests/test_edge_cases/
pytest research-agent/tests/test_behavioral/

# Run with markers
pytest -m integration
pytest -m edge_case
pytest -m behavioral
pytest -m "not slow"  # Skip slow tests

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=research-agent --cov-report=html
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_integration/        # Integration tests
│   ├── test_simple_queries.py
│   ├── test_medium_queries.py
│   ├── test_complex_queries.py
│   └── test_parallel_execution.py
├── test_edge_cases/         # Edge case tests
│   ├── test_empty_results.py
│   ├── test_tool_failures.py
│   └── test_max_iterations.py
└── test_behavioral/         # Behavioral tests
    └── test_coordination.py
```

## Test Categories

### Integration Tests
- **Simple queries**: Fact-finding (1 subagent, 3-10 tool calls)
- **Medium queries**: Comparisons (2-4 subagents, 10-15 calls each)
- **Complex queries**: Breadth-first (5+ subagents, multiple iterations)
- **Parallel execution**: Verify subagents run concurrently

### Edge Case Tests
- Empty search results handling
- Tool failure recovery
- Maximum iteration limits
- Error handling

### Behavioral Tests
- Agent coordination
- Task coverage
- Citation quality

## Test Markers

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.edge_case` - Edge case tests
- `@pytest.mark.behavioral` - Behavioral tests
- `@pytest.mark.slow` - Slow-running tests

## Requirements

- pytest >= 9.0.2
- Environment variables:
  - `OPENAI_API_KEY` (or configured base_url)
  - `TAVILY_API_KEY` (for search tests)

## Notes

- Some tests use mocking to avoid API calls
- Integration tests require actual API access
- Slow tests are marked and can be skipped

