# Parallelization Patterns

This directory demonstrates two parallelization patterns from Building Effective Agents:

## 1. Sectioning Pattern

**Concept**: Break a complex task into independent subtasks that run in parallel.

**Implementation**:
- Uses `Send()` to fan-out tasks to multiple workers
- Each worker processes one section independently
- Results are automatically aggregated using `Annotated[List, operator.add]`

**Key Code**:
```python
def assign_sections(state):
    return [Send("section_worker", {"sections": [section]}) for section in sections]

workflow.add_conditional_edges(
    "section_planner",
    assign_sections,
    ["section_worker"],
)
```

## 2. Voting Pattern

**Concept**: Run the same task multiple times in parallel to get diverse outputs.

**Implementation**:
- Uses `Send()` to fan-out the same task to multiple workers
- Each worker uses different LLM configurations (strict, creative, balanced)
- Results are aggregated to reach consensus

**Key Code**:
```python
def assign_voters(state):
    return [
        Send("voting_worker", {"task": task, "worker_id": "strict"}),
        Send("voting_worker", {"task": task, "worker_id": "creative"}),
        Send("voting_worker", {"task": task, "worker_id": "balanced"}),
    ]

workflow.add_conditional_edges(
    START,
    assign_voters,
    ["voting_worker"],
)
```

## How LangGraph Parallelization Works

LangGraph's `Send()` API enables true parallelization:

1. **Fan-out**: A function returns a list of `Send()` objects
2. **Parallel Execution**: LangGraph executes all `Send()` targets concurrently
3. **Fan-in**: Results are automatically aggregated using reducers (e.g., `operator.add`)

## Verifying Parallelization

To verify that tasks are actually running in parallel:

1. **Check timestamps**: Workers should start at similar times
2. **Check execution time**: Parallel execution should be faster than sequential
3. **Check output order**: Results may arrive in any order

Run the test script:
```bash
python parallel-agent/test_parallel.py
```

## Note on Parallel Execution

LangGraph's `Send()` enables parallel execution, but the actual parallelism depends on:
- The execution environment (async/threading support)
- The LLM provider's concurrency limits
- The system's resources

In practice, you may see:
- True parallel execution (multiple LLM calls simultaneously)
- Concurrent execution (overlapping I/O operations)
- Sequential execution with batching (if resources are limited)

The key is that LangGraph's graph structure supports parallelization, and the execution engine will parallelize when possible.

