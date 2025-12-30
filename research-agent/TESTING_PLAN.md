# Testing Plan for Research Agent

Based on [Anthropic's Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) principles.

## Testing Philosophy

### End-State Evaluation
As mentioned in the Anthropic article: "We found success focusing on end-state evaluation rather than turn-by-turn analysis. Instead of judging whether the agent followed a specific process, evaluate whether it achieved the correct final state."

**Key Principle**: Test whether the system achieves the correct final outcome, not whether it followed a specific process.

### Testing Categories

1. **Functional Tests** - Does the system work?
2. **Integration Tests** - Do components work together?
3. **Behavioral Tests** - Do agents behave correctly?
4. **Edge Case Tests** - How does it handle failures?
5. **Performance Tests** - Does it meet performance targets?

## Test Structure

```
tests/
├── conftest.py          # Pytest fixtures and shared utilities
├── test_unit/           # Unit tests for individual nodes
│   ├── test_lead_researcher.py
│   ├── test_subagent.py
│   ├── test_synthesizer.py
│   ├── test_decision.py
│   └── test_citation_agent.py
├── test_integration/    # Integration tests
│   ├── test_simple_queries.py
│   ├── test_medium_queries.py
│   ├── test_complex_queries.py
│   └── test_parallel_execution.py
├── test_edge_cases/     # Edge case tests
│   ├── test_empty_results.py
│   ├── test_tool_failures.py
│   ├── test_max_iterations.py
│   └── test_error_handling.py
└── test_behavioral/     # Behavioral pattern tests
    ├── test_coordination.py
    ├── test_task_delegation.py
    └── test_iteration_loop.py
```

## Test Cases

### 1. Unit Tests

#### test_lead_researcher.py
- Test query analysis
- Test plan creation
- Test subagent task generation
- Test iteration refinement
- Test JSON parsing fallback

#### test_subagent.py
- Test web search integration
- Test result analysis
- Test empty results handling
- Test source extraction
- Test finding structure

#### test_synthesizer.py
- Test findings aggregation
- Test synthesis quality
- Test empty findings handling

#### test_decision.py
- Test iteration limit enforcement
- Test decision logic (continue/finish)
- Test coverage assessment

#### test_citation_agent.py
- Test citation extraction
- Test report formatting
- Test source deduplication

### 2. Integration Tests

#### test_simple_queries.py
**Query Types** (from Anthropic: "Simple fact-finding requires just 1 agent with 3-10 tool calls")
- "What is LangGraph?"
- "Who created Python?"
- "What is the capital of France?"

**Assertions**:
- Final state has synthesized_results
- Citations are present
- Iteration count <= 1
- Subagents count <= 2

#### test_medium_queries.py
**Query Types** (from Anthropic: "Direct comparisons might need 2-4 subagents with 10-15 calls each")
- "Compare Python and Rust for web development"
- "Research the pros and cons of microservices"
- "Analyze differences between REST and GraphQL"

**Assertions**:
- Final state has comprehensive synthesis
- Multiple subagents executed
- Citations from diverse sources
- Iteration count <= 2

#### test_complex_queries.py
**Query Types** (from Anthropic: "Complex research might use more than 10 subagents")
- "Find all board members of top 10 AI companies"
- "Research history, current state, and future of quantum computing"
- "Compare top 5 cloud providers across pricing, features, reliability"

**Assertions**:
- Multiple iterations executed
- Multiple subagents (3+)
- Comprehensive final report
- Iteration count <= 3

#### test_parallel_execution.py
**Test**: Verify subagents execute in parallel
- Measure execution time
- Compare sequential vs parallel
- Verify all subagents complete

### 3. Edge Case Tests

#### test_empty_results.py
- Test behavior when search returns no results
- Test graceful degradation
- Test partial results handling

#### test_tool_failures.py
- Test Tavily API failures
- Test LLM API failures
- Test error recovery
- Test partial completion

#### test_max_iterations.py
- Test iteration limit (3) enforcement
- Test behavior at max iterations
- Test final state at limit

#### test_error_handling.py
- Test invalid queries
- Test malformed JSON responses
- Test network timeouts
- Test API key errors

### 4. Behavioral Tests

#### test_coordination.py
**From Anthropic**: "Multi-agent systems have key differences from single-agent systems, including a rapid growth in coordination complexity."

- Test task overlap (should be minimal)
- Test task coverage (should be complete)
- Test delegation clarity

#### test_task_delegation.py
**From Anthropic**: "Without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information."

- Test task uniqueness
- Test task clarity
- Test no duplication

#### test_iteration_loop.py
- Test iteration decision logic
- Test strategy refinement
- Test loop termination

## Test Fixtures

### conftest.py
```python
@pytest.fixture
def initial_state():
    """Standard initial state for tests"""
    
@pytest.fixture
def mock_search_results():
    """Mock search results for testing"""
    
@pytest.fixture
def app():
    """Research agent app instance"""
```

## Test Metrics

Based on Anthropic's evaluation approach:

1. **Success Rate**: % of queries that complete successfully
2. **Completeness**: Does answer cover all aspects?
3. **Source Diversity**: Number of unique sources
4. **Citation Accuracy**: Do citations match claims?
5. **Coordination Quality**: Task overlap, coverage
6. **Performance**: Latency, token usage

## Test Data

### Query Complexity Classification

**Simple** (1 subagent, 3-10 tool calls):
- Fact-finding queries
- Single-topic questions

**Medium** (2-4 subagents, 10-15 calls each):
- Comparison queries
- Multi-aspect research

**Complex** (5+ subagents, multiple iterations):
- Breadth-first queries
- Multi-directional research

## Running Tests

```bash
# Run all tests
pytest research-agent/tests/

# Run specific category
pytest research-agent/tests/test_integration/

# Run with coverage
pytest research-agent/tests/ --cov=research-agent --cov-report=html

# Run with verbose output
pytest research-agent/tests/ -v

# Run specific test
pytest research-agent/tests/test_integration/test_simple_queries.py::test_what_is_langgraph
```

## Success Criteria

Tests should verify:
1. ✅ System handles queries of all complexity levels
2. ✅ Subagents execute in parallel
3. ✅ Results are synthesized correctly
4. ✅ Citations are properly extracted
5. ✅ Iteration loop works correctly
6. ✅ Error handling is graceful
7. ✅ Coordination is effective (minimal overlap, good coverage)

