# LangChain Agent Patterns

This project contains multiple LangChain agent patterns implemented using LangGraph.

## Patterns

### Orchestrator-Worker Pattern
Located in `orchestrator-worker/`, this pattern demonstrates a task orchestration system where:
- An orchestrator breaks down tasks into sub-tasks
- Multiple workers execute sub-tasks in parallel
- Results are aggregated

### Evaluator-Optimizer Pattern
Located in `evaluator-optimizer/`, this pattern demonstrates a code generation and optimization system where:
- A generator creates code based on tasks
- An evaluator reviews the code for correctness
- The system iteratively refines code based on feedback
- Maximum retry mechanism prevents infinite loops

### ReAct Agent Pattern
Located in `react-agent/`, this pattern demonstrates a Reasoning + Acting agent system where:
- Agent receives user queries and reasons about what actions to take
- Agent can call tools (calculator, time, text processor) to gather information
- Agent observes tool results and continues reasoning
- Implements a ReAct loop: Think → Act → Observe → Think
- Conditional routing based on tool calls and maximum iteration limits
- Shows how to bind tools to LLMs and use ToolNode for execution

### Router Pattern
Located in `router-agent/`, this pattern demonstrates intelligent task routing:
- A **router node** analyzes tasks and determines the best expert to handle them
- Multiple **specialized expert nodes**: code_expert, analysis_expert, writing_expert, calculation_expert, general_expert
- All experts converge to a **formatter node** for consistent output formatting
- Dynamic routing based on task classification using LLM
- Shows how to implement the Router Pattern from Building Effective Agents
- Demonstrates chaining: Router → Expert → Formatter

### Parallelization Patterns
Located in `parallel-agent/`, this demonstrates two key parallelization workflows:

**1. Sectioning Pattern** (`sectioning.py`):
- Breaks a complex task into independent subtasks
- Processes each section in parallel using `Send()`
- Aggregates results into a comprehensive final answer
- Use case: Complex tasks with multiple independent aspects (e.g., security system design with technical, operational, and strategic sections)

**2. Voting Pattern** (`voting.py`):
- Runs the same task multiple times in parallel
- Gets diverse perspectives using different LLM configurations (strict, creative, balanced)
- Reaches consensus by synthesizing all votes
- Use case: High-confidence evaluations (e.g., code security review, content appropriateness check)
- Demonstrates true parallelization with `Send()` and automatic result aggregation

Both patterns use LangGraph's `Send()` for true parallel execution and `Annotated[List, operator.add]` for automatic result aggregation.

### Multi-Stage Decision Agent Pattern
Located in `decision-agent/`, this pattern demonstrates a complex multi-stage processing system where:
- **Stage 1 (Input Parser)**: Extracts structured data from requests using JSON extraction
- **Stage 2 (Validator)**: Validates extracted data against business rules and complexity metrics
- **Stage 3 (Processor/Complex Processor)**: Routes to standard or complex processing based on complexity
- **Stage 4 (Quality Checker)**: Validates output quality and relevance
- **Stage 5 (Refiner)**: Refines output if quality issues are detected
- **Stage 6 (Formatter)**: Formats final output based on intent type
- **Error Handler**: Handles errors with retry logic and error counting
- Multiple conditional routing points based on validation results and error counts
- Complex state management with validation results, processing logs, and decision history

## Setup

1. Install dependencies using uv:
```bash
uv sync
```

2. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## Running

### Orchestrator-Worker Pattern
```bash
uv run python orchestrator-worker/main.py
```

### Evaluator-Optimizer Pattern
```bash
uv run python evaluator-optimizer/main.py
```

### ReAct Agent Pattern
```bash
uv run python react-agent/main.py
```

### Router Pattern
```bash
uv run python router-agent/main.py
```

### Parallelization Patterns
```bash
uv run python parallel-agent/main.py
```

### Multi-Stage Decision Agent Pattern
```bash
uv run python decision-agent/main.py
```

To visualize the graph:
```bash
uv run python decision-agent/visualize.py
```

## Project Structure

```
.
├── orchestrator-worker/    # Orchestrator-Worker pattern implementation
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflow definition
│   ├── schemas.py         # Data schemas
│   └── main.py            # Entry point
├── evaluator-optimizer/   # Evaluator-Optimizer pattern implementation
│   ├── graph.py           # LangGraph workflow definition
│   ├── schemas.py         # Data schemas
│   └── main.py            # Entry point
├── react-agent/            # ReAct Agent pattern implementation
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflow definition
│   ├── schemas.py         # Data schemas
│   ├── tools.py           # Tool definitions
│   └── main.py            # Entry point
├── router-agent/           # Router Pattern implementation
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflow definition
│   ├── schemas.py         # Data schemas
│   └── main.py            # Entry point
├── parallel-agent/        # Parallelization Patterns (Sectioning & Voting)
│   ├── __init__.py
│   ├── sectioning.py      # Sectioning pattern implementation
│   ├── voting.py          # Voting pattern implementation
│   ├── schemas.py         # Data schemas
│   └── main.py            # Entry point
├── decision-agent/         # Multi-Stage Decision Agent pattern
│   ├── __init__.py
│   ├── graph.py           # LangGraph workflow definition
│   ├── schemas.py         # Data schemas
│   ├── main.py            # Entry point
│   └── visualize.py       # Graph visualization script
├── pyproject.toml         # Project configuration
├── uv.lock                # Dependency lock file
└── .env                   # Environment variables (create this)
```

