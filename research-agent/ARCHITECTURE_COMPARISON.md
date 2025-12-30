# Architecture Comparison: Current State vs Target MVP

## Current Codebase Capabilities

### ✅ What We Have

1. **Parallel Execution Patterns**
   - `parallel-agent/sectioning.py`: Sectioning pattern with `Send()`
   - `parallel-agent/voting.py`: Voting pattern with parallel workers
   - Proven parallelization using LangGraph's `Send()` API

2. **Orchestrator-Worker Pattern**
   - `orchestrator-worker/graph.py`: Basic orchestrator that delegates to workers
   - Task decomposition
   - Result aggregation

3. **State Management**
   - TypedDict schemas
   - Annotated lists for aggregation
   - LangGraph state management

4. **LLM Integration**
   - ChatOpenAI setup with qwen-plus
   - Structured output support
   - Multiple LLM instances for different purposes

5. **Dependencies**
   - Tavily Python SDK (available but not yet used)
   - LangGraph for workflow orchestration
   - All necessary libraries

### ❌ What We Need to Build

1. **Web Search Integration**
   - Tavily tool wrapper
   - Search result processing
   - Source extraction

2. **Research-Specific Components**
   - LeadResearcher orchestrator with research strategy
   - Subagents with search capabilities
   - Synthesis node for research results
   - Citation extraction

3. **Iterative Research Loop**
   - Decision logic for continuing research
   - Strategy refinement
   - Iteration control

4. **Memory System**
   - Plan persistence
   - Context management
   - State retrieval

## Mapping: Existing Patterns → Research System

### Pattern Reuse

| Research Component | Existing Pattern | Adaptation Needed |
|-------------------|------------------|-------------------|
| LeadResearcher | `orchestrator-worker/orchestrator` | Add research strategy, plan creation, iterative decision-making |
| Subagents | `parallel-agent/section_worker` | Add web search, result evaluation, source tracking |
| Parallel Execution | `parallel-agent/assign_sections` | Adapt for research task distribution |
| Synthesis | `parallel-agent/aggregator` | Enhance for research synthesis with citations |
| Memory | None | New: Plan storage, context management |

### Code Structure Comparison

#### Current: Orchestrator-Worker
```python
# orchestrator-worker/graph.py
def orchestrator_node(state):
    # Simple task breakdown
    plan = planner.invoke([...])
    return {"plan": plan["steps"]}

def worker_node(state):
    # Execute single step
    response = llm.invoke([...])
    return {"results": [...]}
```

#### Target: LeadResearcher
```python
# research-agent/graph.py
def lead_researcher_node(state):
    # Research strategy formulation
    # Plan creation and storage
    # Subagent task generation with detailed descriptions
    # Decision: continue or exit
    return {
        "research_plan": plan,
        "subagent_tasks": tasks,
        "needs_more_research": decision
    }
```

#### Current: Section Worker
```python
# parallel-agent/sectioning.py
def section_worker_node(state):
    section = state["sections"][0]
    response = llm.invoke([...])
    return {"section_results": [result]}
```

#### Target: Research Subagent
```python
# research-agent/graph.py
def subagent_node(state):
    task = state["subagent_task"]
    # Perform web search
    search_results = tavily_search(task)
    # Evaluate results
    findings = evaluate_results(search_results)
    return {
        "subagent_findings": [{
            "task": task,
            "findings": findings,
            "sources": sources
        }]
    }
```

## Implementation Strategy

### Phase 1: Foundation (Reuse + Extend)
1. **Start with orchestrator-worker pattern**
   - Copy structure from `orchestrator-worker/`
   - Extend orchestrator to LeadResearcher

2. **Add parallel execution**
   - Use `Send()` pattern from `parallel-agent/sectioning.py`
   - Adapt for research subagents

3. **Integrate web search**
   - Create Tavily tool wrapper
   - Add to subagent nodes

### Phase 2: Research-Specific Features
1. **Enhance synthesis**
   - Extend aggregator from `parallel-agent/sectioning.py`
   - Add citation extraction

2. **Add iterative loop**
   - Conditional routing for research continuation
   - Strategy refinement logic

3. **Implement memory**
   - Plan storage in state
   - Context management

### Phase 3: Optimization
1. **Prompt engineering**
   - Research-specific prompts
   - Delegation instructions
   - Effort scaling

2. **Testing and validation**
   - Various query types
   - Performance metrics

## Key Differences from Existing Patterns

### 1. Tool Integration
- **Existing**: No external tools (pure LLM)
- **Research**: Requires Tavily web search tool

### 2. Iterative Nature
- **Existing**: Single-pass execution
- **Research**: Multi-iteration loop with refinement

### 3. Source Attribution
- **Existing**: No source tracking
- **Research**: Citation extraction and attribution

### 4. Strategy Formulation
- **Existing**: Simple task breakdown
- **Research**: Research strategy, plan creation, effort scaling

### 5. Result Evaluation
- **Existing**: Direct aggregation
- **Research**: Evaluation, filtering, synthesis

## Reusable Components

### Direct Reuse
- ✅ LangGraph `Send()` for parallelization
- ✅ State schema patterns (TypedDict + Annotated)
- ✅ LLM initialization patterns
- ✅ Graph structure patterns

### Adaptable Patterns
- 🔄 Orchestrator → LeadResearcher (extend)
- 🔄 Worker → Subagent (add search)
- 🔄 Aggregator → Synthesizer (add citations)

### New Components
- ⚡ Tavily search tool
- ⚡ Citation extraction
- ⚡ Research decision logic
- ⚡ Memory system

## Migration Path

```
orchestrator-worker/
  └─→ research-agent/ (extend orchestrator)
       ├─ Add research strategy
       ├─ Add plan storage
       └─ Add iterative decision

parallel-agent/sectioning.py
  └─→ research-agent/ (adapt workers)
       ├─ Add web search
       ├─ Add source tracking
       └─ Add result evaluation

parallel-agent/aggregator
  └─→ research-agent/ (enhance synthesis)
       ├─ Add citation extraction
       └─ Add research synthesis
```

---

**Conclusion**: We have strong foundations in parallelization and orchestration. The main work is:
1. Adding web search capability
2. Enhancing orchestration for research strategy
3. Adding iterative refinement
4. Implementing citations

