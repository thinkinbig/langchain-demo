# Subagent Design Documentation

## Overview

The **Subagent** is a sophisticated subgraph within the Research Agent system that handles individual research tasks. It implements a multi-stage workflow with intelligent routing, context isolation, and comprehensive error handling. Each subagent operates as an independent worker that can retrieve from internal knowledge bases, perform web searches and academic paper searches (in parallel), and analyze results to produce structured findings. The LLM in the analysis node intelligently selects which sources to use based on the task requirements.

## Architecture

### Subgraph Structure

The subagent is implemented as a **LangGraph subgraph** (`subagent_workflow`) with three core nodes:

```
START → retrieve_node → [route_source_necessity] → analysis_node → END
                              ↓
                        web_search_node → analysis_node
```

### Node Responsibilities

#### 1. **retrieve_node** (`nodes/subagent/retrieve.py`)
- **Purpose**: Retrieves relevant information from internal knowledge base (RAG)
- **Input**: Task description, visited sources
- **Output**: `internal_result` (RetrievalResult object)
- **Key Features**:
  - Uses `RetrievalService.aretrieve_internal()` for async retrieval
  - Filters out already-visited internal sources
  - Implements timeout protection (`TIMEOUT_RETRIEVAL`)
  - Returns empty result on timeout/failure (graceful degradation)

#### 2. **web_search_node** (`nodes/subagent/web_search.py`)
- **Purpose**: Performs web search and academic paper search in parallel when internal knowledge is insufficient
- **Input**: Task description, visited URLs (both web and paper)
- **Output**: `web_result` and `paper_result` (both RetrievalResult objects)
- **Key Features**:
  - Only executed when `route_source_necessity` determines internal knowledge is insufficient
  - **Parallel Search Execution**: Simultaneously searches:
    - **Web sources**: Uses `RetrievalService.aretrieve_web()` with Tavily API
    - **Academic papers**: Uses `RetrievalService.aretrieve_papers()` with arXiv API (and optionally Semantic Scholar)
  - Supports deep scraping of top web results (`scrape_top_result=True`)
  - Filters out already-visited URLs (both web and paper)
  - Implements timeout protection (`TIMEOUT_WEB_SEARCH`)
  - **LLM-Driven Source Selection**: Both results are passed to analysis_node, where the LLM decides which sources to use based on task requirements

#### 3. **analysis_node** (`nodes/subagent/analysis.py`)
- **Purpose**: Analyzes gathered context and produces structured findings
- **Input**: `internal_result`, `web_result`, and/or `paper_result`
- **Output**: `subagent_findings`, `extracted_citations`
- **Key Features**:
  - **Structured Output**: Uses LLM with structured output (`AnalysisOutput` schema)
  - **Context Formatting**: Formats retrieval results with citations from all source types
  - **Intelligent Source Selection**: LLM decides which sources to use:
    - Academic papers for research/technical queries
    - Web sources for current events/general information
    - Combination of both when complementary information is needed
  - **Reference Enhancement**: Automatically retrieves "References" sections from internal papers
  - **Tool Execution**: Supports Python code execution for data analysis
  - **Refinement Loop**: If code is generated, executes it and refines analysis with results
  - **Retry Logic**: Implements retry mechanism for LLM calls with error handling

### Routing Logic

#### `route_source_necessity` Function
A **gating node** that determines whether web search is needed:

```python
def route_source_necessity(state: SubagentState) -> Literal["analysis_node", "web_search_node"]:
    internal_result = state.get("internal_result")
    
    if internal_result and has_content:
        return "analysis_node"  # Skip web search
    else:
        return "web_search_node"  # Need external search
```

**Decision Criteria**:
- Checks if `internal_result` exists and has meaningful content
- Handles both `RetrievalResult` objects and dict representations (from state serialization)
- Prevents unnecessary web searches when internal knowledge is sufficient

## State Management

### SubagentState Schema

```python
class SubagentState(DictCompatibleModel):
    subagent_tasks: List[ResearchTask]  # Required: at least one task
    task_description: str  # Cached task description
    internal_result: Optional[Any]  # Internal knowledge retrieval result
    web_result: Optional[Any]  # Web search retrieval result
    paper_result: Optional[Any]  # Academic paper search result (arXiv/Semantic Scholar)
    subagent_findings: List[Finding]  # Findings output
    visited_sources: List[VisitedSource]  # Already visited sources
    extracted_citations: List[dict]  # Citations extracted in this run
```

### Context Isolation

The subagent implements **strict context isolation** to prevent state leakage between parallel subagents:

#### 1. **Input State Isolation** (`isolate_subgraph_state`)
- Sanitizes input state before passing to subgraph
- Only includes fields the subagent needs:
  - `query`
  - `subagent_tasks` (filtered to current task)
  - `visited_sources`
- Uses `SubgraphContextSandbox` to enforce field-level access control

#### 2. **Output State Filtering** (`subagent_node` wrapper)
- Filters output to only return allowed fields:
  - `subagent_findings`
  - `visited_sources`
  - `all_extracted_citations`
- Prevents unauthorized state modifications
- Validates output using sandbox before returning

#### 3. **Namespace Isolation** (via `task_id`)
- Each subagent task gets a unique namespace
- Prevents write conflicts in parallel execution
- Uses `NamespaceIsolator` for memory write isolation

### State Flow

```
Parent State (ResearchState)
    ↓ [isolate_subgraph_state]
Isolated SubagentState
    ↓ [subagent_app.ainvoke]
Subgraph Execution
    ↓ [validate_output_state]
Filtered Output
    ↓ [merge to parent]
Parent State Updated
```

## Error Handling & Resilience

### Timeout Protection

All async operations have timeout protection:

| Operation | Timeout Setting | Behavior on Timeout |
|-----------|----------------|---------------------|
| Retrieval | `TIMEOUT_RETRIEVAL` | Returns empty `RetrievalResult` |
| Web Search | `TIMEOUT_WEB_SEARCH` | Returns empty `RetrievalResult` |
| Subgraph Execution | `TIMEOUT_SUBAGENT` | Returns empty findings |
| Python Code Execution | `TIMEOUT_PYTHON_REPL` | Returns timeout message |
| LLM Calls | `TIMEOUT_LLM_CALL` | Retries up to `max_retries` |

### Retry Logic

#### LLM Call Retries (`invoke_llm_with_retry`)
- **Max Retries**: 2 attempts
- **Retry Triggers**:
  - Timeout errors
  - Validation/parsing errors
  - General exceptions
- **Retry Strategy**: Appends error message to conversation history (preserves KV cache)
- **Fallback**: Returns minimal `AnalysisOutput` if all retries fail

### Graceful Degradation

- Empty retrieval results don't crash the subagent
- Timeout errors are logged but don't propagate
- Failed code execution continues with initial analysis
- Validation errors trigger retries before giving up

## Advanced Features

### 1. Reference Section Enhancement

The `analysis_node` automatically enhances context by retrieving "References" sections from internal papers:

```python
def enhance_context_with_references(all_sources: List[Source]) -> str:
    # For each internal source, retrieve its References section
    # Appends to formatted_context for better citation accuracy
```

**Benefits**:
- Improves citation extraction accuracy
- Provides official reference lists from papers
- Helps LLM identify proper citation formats

### 2. Tool Execution & Refinement

When the LLM generates Python code in `AnalysisOutput`:

1. **Code Execution**: Runs code in secure Python REPL
2. **Result Integration**: Appends tool output to conversation
3. **Refinement**: LLM analyzes results and refines findings
4. **KV Cache Optimization**: Uses message appending to preserve cache

**Use Cases**:
- Data analysis and visualization
- Statistical computations
- Text processing and extraction

### 3. Academic Paper Search

The `web_search_node` automatically searches for academic papers alongside web sources:

- **arXiv Integration**: Free, no API key required. Searches arXiv for research papers.
- **Semantic Scholar Integration**: Optional, requires `SEMANTIC_SCHOLAR_API_KEY`. Provides additional academic paper sources.
- **Parallel Execution**: Paper search runs in parallel with web search for efficiency.
- **Structured Paper Data**: Returns paper title, authors, abstract, publication date, and URL.
- **LLM-Driven Selection**: The LLM in `analysis_node` intelligently selects which sources (papers vs web) to use based on task requirements.

**Benefits**:
- Automatically provides academic sources for research queries
- No hard-coded routing logic - LLM decides what's relevant
- Comprehensive coverage: both academic depth and web currency

### 4. Unified Source Tracking

The subagent uses a unified `VisitedSource` format:

```python
class VisitedSource:
    identifier: str  # URL or internal document ID
    source_type: Literal["internal", "web", "paper"]
```

**Benefits**:
- Prevents duplicate retrieval from same source
- Tracks internal, web, and paper sources uniformly
- Enables efficient deduplication across parallel subagents

### 5. Structured Output Schema

The `AnalysisOutput` schema ensures consistent, parseable results:

```python
class AnalysisOutput(BaseModel):
    summary: str  # Main finding summary
    citations: List[Citation]  # Structured citations
    reasoning: str  # LLM's reasoning process
    python_code: Optional[str]  # Optional code for analysis
```

**Benefits**:
- Type-safe output parsing
- Consistent structure across all subagents
- Enables programmatic processing of findings

## Integration with Parent Graph

### Entry Point: `subagent_node` Wrapper

The `subagent_node` function serves as the bridge between parent graph and subgraph:

```python
async def subagent_node(state: SubagentState):
    # 1. Isolate input state
    isolated_state = isolate_subgraph_state(state, "subagent", task_id)
    
    # 2. Invoke subgraph with timeout
    result_state = await asyncio.wait_for(
        subagent_app.ainvoke(isolated_state),
        timeout=settings.TIMEOUT_SUBAGENT
    )
    
    # 3. Extract and format visited sources
    visited_sources = extract_visited_sources(result_state)
    
    # 4. Filter and validate output
    output = {
        "subagent_findings": result_state.get("subagent_findings", []),
        "visited_sources": visited_sources,
        "all_extracted_citations": result_state.get("extracted_citations", [])
    }
    
    return validated_output
```

### Parallel Execution

The parent graph uses LangGraph's `Send()` API to spawn multiple subagents:

```python
def assign_subagents(state: ResearchState):
    return [
        Send("subagent", {
            "subagent_tasks": [task],
            "query": state["query"],
            "visited_sources": state.get("visited_sources", []),
        })
        for task in tasks
    ]
```

**Key Points**:
- Each subagent receives only its assigned task
- All subagents share the same `visited_sources` list (for deduplication)
- Results are automatically merged via reducer functions in parent state

## Performance Optimizations

### 1. KV Cache Preservation
- Uses message appending instead of rewriting prompts
- Preserves conversation history for refinement steps
- Maximizes cache hit rate in LLM calls

### 2. Async Operations
- All I/O operations are async (`aretrieve_internal`, `aretrieve_web`)
- Parallel execution of independent subagents
- Non-blocking timeout handling

### 3. Context Chunking
- Uses `ContextFormatter` to efficiently format large contexts
- Implements smart truncation and summarization
- Prevents context window overflow

### 4. Early Termination
- Gating node prevents unnecessary web searches
- Skips analysis if no content is retrieved
- Returns empty findings on critical failures

## Configuration

### Timeout Settings

All timeouts are configurable via `config.py`:

```python
TIMEOUT_RETRIEVAL = 30  # seconds
TIMEOUT_WEB_SEARCH = 60  # seconds
TIMEOUT_SUBAGENT = 300  # seconds (5 minutes)
TIMEOUT_PYTHON_REPL = 30  # seconds
TIMEOUT_LLM_CALL = 120  # seconds
```

### Model Selection

The subagent uses `get_llm_by_model_choice("turbo")` for:
- Fast response times
- Cost efficiency
- Structured output support

### Paper Search Configuration

Paper search is automatically enabled and runs in parallel with web search:

```python
USE_SEMANTIC_SCHOLAR: bool = False  # Enable Semantic Scholar (requires API key)
```

**Configuration Options**:
- **arXiv**: Always enabled (free, no API key required)
- **Semantic Scholar**: Optional, set `USE_SEMANTIC_SCHOLAR=true` and provide `SEMANTIC_SCHOLAR_API_KEY` environment variable

**Note**: The LLM in `analysis_node` automatically decides which sources to use based on task requirements. No hard-coded routing logic is needed.

## Testing Considerations

### Unit Test Coverage
- Each node should be testable in isolation
- Mock retrieval services for predictable results
- Test routing logic with various state configurations

### Integration Test Scenarios
- Full subgraph execution with real retrieval
- Parallel subagent execution
- Timeout and error handling
- State isolation verification

### Edge Cases
- Empty retrieval results
- All sources already visited
- LLM parsing failures
- Code execution errors
- Concurrent state updates

## Future Enhancements

### Potential Improvements
1. **Caching Layer**: Cache retrieval results for repeated queries
2. **Adaptive Timeouts**: Adjust timeouts based on query complexity
3. **Multi-modal Support**: Handle images, PDFs, and other media
4. **Streaming Output**: Stream findings as they're generated
5. **Quality Scoring**: Add confidence scores to findings
6. **Source Ranking**: Prioritize high-quality sources

### Known Limitations
- Reference section retrieval relies on query-based search (could be more precise)
- Python code execution is limited to REPL (no file system access)
- No built-in rate limiting for external APIs
- State serialization can be complex with nested objects

## Summary

The Subagent design is a sophisticated, production-ready system that:

✅ **Isolates context** to prevent state leakage  
✅ **Handles errors gracefully** with timeouts and retries  
✅ **Optimizes performance** with async operations and caching  
✅ **Provides structured output** for reliable parsing  
✅ **Supports tool execution** for advanced analysis  
✅ **Searches multiple source types** - web, academic papers (arXiv/Semantic Scholar), and internal knowledge  
✅ **Intelligent source selection** - LLM decides which sources to use based on task requirements  
✅ **Tracks sources** to prevent duplicates across all source types  
✅ **Integrates seamlessly** with parent graph  

**Key Innovation**: The system automatically searches both web and academic papers in parallel, then lets the LLM intelligently select which sources are most relevant. This eliminates the need for hard-coded routing logic and ensures comprehensive coverage for both research-oriented and general queries.

This design enables the Research Agent to scale horizontally while maintaining quality and reliability.

