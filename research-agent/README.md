# Multi-Agent Research System MVP

This directory contains the implementation of a multi-agent research system inspired by [Anthropic's Research feature](https://www.anthropic.com/engineering/multi-agent-research-system).

## Overview

The system implements an orchestrator-worker pattern where:
- **LeadResearcher** (orchestrator) analyzes queries, creates research strategies, and coordinates subagents
- **Subagents** (workers) perform parallel web searches and return findings
- **Synthesizer** aggregates and synthesizes results
- **CitationAgent** extracts and formats citations

## Project Status
 
✅ **MVP Complete** - Core functionality, parallel execution, and cost control implemented.
 
## Usage
 
### 1. Setup Environment
 
Ensure your `.env` file is configured with necessary API keys:
 
```bash
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
LANGCHAIN_TRACING_V2=true  # Optional: For LangSmith monitoring
LANGCHAIN_API_KEY="lsv2-..."
```
 
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
python research-agent/tests/evaluate_langsmith.py
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
 
### Phase 4: Optimization (In Progress)
- [x] Cost Control & Budgeting
- [x] Search Retry Logic
- [x] Evaluation Script
- [ ] Comprehensive Unit Tests

## Dependencies

All required dependencies are already in the project:
- `langgraph` - Workflow orchestration
- `langchain-openai` - LLM integration
- `tavily-python` - Web search API

## Next Steps

1. Review the analysis documents
2. Approve architecture and requirements
3. Begin Phase 1 implementation

## References

- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- Existing patterns in this codebase:
  - `parallel-agent/` - Parallelization patterns
  - `orchestrator-worker/` - Basic orchestrator pattern

