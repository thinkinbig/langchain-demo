# Multi-Agent Research System MVP

This directory contains the implementation of a multi-agent research system inspired by [Anthropic's Research feature](https://www.anthropic.com/engineering/multi-agent-research-system).

## Overview

The system implements an orchestrator-worker pattern where:
- **Lead Researcher**: Decomposes queries into subtasks (uses RAG for internal context).
- **Subagent**: Can execute Python code and search the web (uses RAG for internal context).
- **Synthesizer**: Aggregates findings into a dense summary.
- **Verifier**: Checker for hallucinations.
- **Decision Maker**: Determines if more research loop is needed.

### 2. Memory Architecture
- **Vector Store (RAG)**: Uses ChromaDB + HuggingFace Embeddings for efficient retrieval of internal documents (PDFs, TXT).
- **Short-Term Memory**: LangGraph State (persists across graph nodes).
- **Long-Term Persistence**: MemorySaver (persists across sessions).

### 3. Toolset
 
## Project Status
 
✅ **MVP Complete** - Core functionality, parallel execution, and cost control implemented.
 
## Usage
 
### 1. Setup Environment

**Prerequisites:**
- Docker installed and running (required for secure Python code execution)
- Python 3.11+

**Install Docker:**
```bash
# Verify Docker is installed
docker --version

# Ensure Docker daemon is running
docker ps
```

**Install Dependencies:**
```bash
pip install -e .
# or
uv sync
```

**Configure API Keys:**
Ensure your `.env` file is configured with necessary API keys:

```bash
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
LANGCHAIN_TRACING_V2=true  # Optional: For LangSmith monitoring
LANGCHAIN_API_KEY="lsv2-..."
```

**Optional: Configure Docker Sandbox:**
The Python code execution sandbox can be configured via environment variables:

```bash
PYTHON_SANDBOX_IMAGE=python:3.11-slim  # Docker image to use
PYTHON_SANDBOX_TIMEOUT=30              # Execution timeout (seconds)
PYTHON_SANDBOX_MEMORY=256m             # Memory limit
PYTHON_SANDBOX_CPU_QUOTA=50000         # CPU quota (50% = 50000/100000)
ENABLE_DOCKER_SANDBOX=true             # Enable Docker sandboxing
```

**Note:** If Docker is not available, the system will automatically fall back to restricted execution mode (less secure but functional).
 
### 2. Run Research Agent
 
Run the main script to execute a sample research query:
 
```bash
python research-agent/main.py
```
 
This will:
1.  Check your daily budget (default $50/day).
2.  Plan research tasks.
3.  Execute web searches in parallel pairs.
4.  Synthesize findings and iterate if necessary.
5.  Generate a final report with citations.
 
### 3. Run Evaluation
 
To test performance against a baseline dataset:
 
```bash
python research-agent/tests/evaluate_agent.py
```
 
## Key Metrics & Targets
 
### Critical Metrics (Must Achieve)
- **End-to-End Latency**: < 120s (acceptable), < 60s (target)
- **Query Success Rate**: > 85% (acceptable), > 90% (target)
- **Answer Completeness**: > 70% (acceptable), > 80% (target)
- **Token Usage**: < 100k per query (**HARD LIMIT**), 30k-50k (target)
- **Automation Rate**: > 60% (acceptable), > 70% (target)
 
### Cost Control (Mandatory)
- **Per Query**: Max 100k tokens / $0.50 (**Immediately stop** if exceeded)
- **Daily Budget**: Max 5M tokens / $50 (**Reject new queries** if exceeded)
- **Monthly Budget**: Max 120M tokens / $1200 (**Complete stop** if exceeded)
 
See [QUANTIFIED_REQUIREMENTS.md](./QUANTIFIED_REQUIREMENTS.md) for complete metrics.
 
## Documentation
 
1.  **[MVP_ANALYSIS.md](./MVP_ANALYSIS.md)** - Comprehensive requirements and architecture analysis
2.  **[ARCHITECTURE_COMPARISON.md](./ARCHITECTURE_COMPARISON.md)** - Comparison of existing patterns vs target architecture
3.  **[REQUIREMENTS_CHECKLIST.md](./REQUIREMENTS_CHECKLIST.md)** - Detailed implementation checklist
4.  **[METRICS_AND_KPIs.md](./METRICS_AND_KPIs.md)** - Comprehensive metrics and KPIs definition
5.  **[QUANTIFIED_REQUIREMENTS.md](./QUANTIFIED_REQUIREMENTS.md)** - Quick reference for all quantified targets
6.  **[MEASUREMENT_PLAN.md](./MEASUREMENT_PLAN.md)** - Implementation plan for metrics tracking
 
## Key Design Decisions
 
### Architecture
- **Pattern**: Orchestrator-Worker with iterative refinement
- **Parallelization**: LangGraph's `Send()` API (proven in `parallel-agent/`)
- **Search**: Tavily API for web search
- **Memory**: In-memory state initially, external storage post-MVP
 
### Core Components
 
1.  **LeadResearcher Node**
    - Query analysis and strategy formulation
    - Research plan creation and storage
    - Subagent task generation with detailed descriptions
    - Result synthesis
    - Iterative decision-making
 
2.  **Subagent Nodes**
    - Parallel web search execution
    - Result evaluation and filtering
    - Source tracking
    - Structured findings output
 
3.  **Synthesis Node**
    - Aggregation of subagent findings
    - Comprehensive result synthesis
    - Citation integration
 
4.  **CitationAgent Node**
    - Citation extraction from sources
    - Attribution formatting
    - Final report generation
 
5.  **FACT Verifier Node (V2)**
    - Cross-references final report claims against deep-scraped source text.
    - Reduces hallucinations by ensuring evidence entailment.

## Implementation Status
 
### Phase 1: Core Orchestration (Completed)
- [x] Project structure setup
- [x] State schema definition
- [x] LeadResearcher node
- [x] Subagent nodes with web search
- [x] Basic synthesis
 
### Phase 2: Iterative Loop (Completed)
- [x] Decision logic
- [x] Conditional routing
- [x] Strategy refinement
 
### Phase 3: Memory & Citations (Completed)
- [x] Plan persistence
- [x] Citation extraction
- [x] Report formatting
 
### Phase 4: Optimization (Completed)
- [x] Cost Control & Budgeting
- [x] Search Retry Logic
- [x] Evaluation Script
- [x] Comprehensive Unit Tests

### Phase 5: Deep Research V2 (Completed)
- [x] **Context Engineering**: Integrated Anthropic-style high-context agent prompts.
- [x] **Causal Chains**: Implemented `ResearchTask` objects with dependencies.
- [x] **Deep Navigation**: Subagents scrape and read full page content.
- [x] **FACT Verification**: Added `verifier_node` to fact-check reports against source text.

## Dependencies

All required dependencies are in the project:
- `langgraph` - Workflow orchestration
- `langchain-openai` - LLM integration
- `tavily-python` - Web search API
- `docker` - Docker Python SDK (for secure code execution sandbox)

## Security Features

### Python Code Execution Sandboxing

The system implements production-grade security for Python code execution:

- **Docker Containerization**: Code runs in isolated containers with no network access
- **Resource Limits**: CPU (50%), Memory (256MB), Timeout (30s)
- **AST Validation**: Pre-execution validation blocks dangerous operations
- **Execution Logging**: All code execution attempts are logged
- **Graceful Fallback**: Falls back to restricted execution if Docker unavailable

See [SECURITY_ANALYSIS.md](./SECURITY_ANALYSIS.md) for detailed security documentation.

### Troubleshooting

**Docker not available:**
- The system will automatically use restricted execution mode
- A warning will be logged about reduced security
- For production use, ensure Docker is installed and running

**Docker permission errors:**
- Ensure your user is in the `docker` group: `sudo usermod -aG docker $USER`
- Restart your session after adding to docker group

**Container timeout:**
- Increase `PYTHON_SANDBOX_TIMEOUT` if legitimate code needs more time
- Check logs for execution time to optimize timeout settings

## Next Steps

1. Review the analysis documents
2. Approve architecture and requirements
3. Begin Phase 1 implementation

## References

- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- Existing patterns in this codebase:
  - `parallel-agent/` - Parallelization patterns
  - `orchestrator-worker/` - Basic orchestrator pattern

