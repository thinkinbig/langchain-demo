# Multi-Agent Research System MVP - Requirements & Architecture Analysis

## Executive Summary

This document analyzes the requirements for building an MVP that replicates Anthropic's multi-agent research system. Based on the [Anthropic engineering blog post](https://www.anthropic.com/engineering/multi-agent-research-system), we'll identify core features, architectural decisions, and implementation priorities.

## 1. Core Requirements Analysis

### 1.1 Key Features from Anthropic's System

From the article, the essential components are:

1. **LeadResearcher (Orchestrator)**
   - Analyzes user queries
   - Develops research strategy
   - Creates and coordinates subagents
   - Synthesizes results
   - Decides when more research is needed

2. **Subagents (Workers)**
   - Operate in parallel
   - Perform web searches independently
   - Use interleaved thinking to evaluate results
   - Return findings to LeadResearcher

3. **Memory System**
   - Persists research plan (context > 200k tokens)
   - Stores essential information across phases
   - Enables context retrieval when limits approach

4. **CitationAgent**
   - Processes documents and research report
   - Identifies specific locations for citations
   - Ensures proper attribution

5. **Dynamic Search**
   - Multi-step search (not static RAG)
   - Adapts to new findings
   - Analyzes results iteratively

6. **Iterative Research Loop**
   - Can create additional subagents
   - Refines strategy based on findings
   - Exits when sufficient information gathered

### 1.2 MVP Scope Definition

**Must Have (MVP Core):**
- ✅ LeadResearcher orchestrator
- ✅ Parallel subagents with web search
- ✅ Basic synthesis of results
- ✅ Simple memory/plan persistence
- ✅ Iterative research loop (at least 1 iteration)

**Nice to Have (Post-MVP):**
- ⚠️ CitationAgent (can be simplified initially)
- ⚠️ Advanced memory management (context compression)
- ⚠️ Dynamic subagent creation based on findings
- ⚠️ Interleaved thinking in subagents

**Out of Scope (Future):**
- ❌ Rainbow deployments
- ❌ Full production observability
- ❌ Complex error recovery
- ❌ Asynchronous execution

## 2. Architecture Design

### 2.1 High-Level Architecture

```
User Query
    ↓
LeadResearcher (Orchestrator)
    ├─ Analyze query
    ├─ Create research plan → Memory
    ├─ Spawn Subagents (parallel)
    │   ├─ Subagent 1: Search & Analyze
    │   ├─ Subagent 2: Search & Analyze
    │   └─ Subagent N: Search & Analyze
    ├─ Synthesize results
    ├─ Decide: More research needed?
    │   ├─ Yes → Spawn more subagents
    │   └─ No → Continue
    └─ CitationAgent (simplified)
        └─ Final report with citations
```

### 2.2 Component Breakdown

#### 2.2.1 LeadResearcher Node
**Responsibilities:**
- Query analysis and strategy formulation
- Plan creation and persistence
- Subagent creation with clear task descriptions
- Result synthesis
- Decision: continue or exit research loop

**Key Prompt Engineering Principles (from article):**
- Scale effort to query complexity
- Teach orchestrator how to delegate
- Provide detailed task descriptions to subagents

#### 2.2.2 Subagent Nodes
**Responsibilities:**
- Receive specific research task
- Perform web searches (using Tavily)
- Evaluate and filter results
- Return structured findings

**Key Features:**
- Parallel execution using `Send()`
- Independent context windows
- Focused, specialized prompts

#### 2.2.3 Memory System
**MVP Implementation:**
- Simple in-memory storage for research plan
- State persistence in LangGraph state
- Basic context retrieval

**Future Enhancement:**
- External memory (database/file)
- Context compression
- Checkpointing

#### 2.2.4 CitationAgent Node
**MVP Implementation:**
- Simple citation extraction from sources
- Basic attribution formatting

**Future Enhancement:**
- Precise location identification
- Source verification

### 2.3 State Schema Design

```python
class ResearchState(TypedDict):
    # Input
    query: str
    
    # Planning
    research_plan: str  # Stored plan
    strategy: str  # Research strategy
    
    # Subagent coordination
    subagent_tasks: List[str]  # Tasks for subagents
    subagent_findings: Annotated[List[SubagentFinding], operator.add]
    
    # Iteration control
    iteration_count: int
    max_iterations: int
    needs_more_research: bool
    
    # Output
    synthesized_results: str
    citations: List[Citation]
    final_report: str
```

### 2.4 Graph Structure

```
START
  ↓
LeadResearcher (plan & delegate)
  ↓
[Conditional: Create Subagents]
  ├─→ Subagent 1 (parallel)
  ├─→ Subagent 2 (parallel)
  └─→ Subagent N (parallel)
  ↓
Synthesizer
  ↓
[Conditional: More research needed?]
  ├─ Yes → LeadResearcher (refine & delegate)
  └─ No → CitationAgent
      ↓
      END
```

## 3. Implementation Plan

### 3.1 Phase 1: Core Orchestration (Week 1)
**Goal:** Basic orchestrator-worker pattern with search

**Tasks:**
1. Create `research-agent/` directory structure
2. Define `ResearchState` schema
3. Implement `LeadResearcher` node:
   - Query analysis
   - Plan creation
   - Subagent task generation
4. Implement `Subagent` node:
   - Web search integration (Tavily)
   - Result processing
5. Basic graph with parallel subagents
6. Simple synthesis node

**Deliverable:** Working research system with parallel search

### 3.2 Phase 2: Iterative Research Loop (Week 1-2)
**Goal:** Add iterative refinement capability

**Tasks:**
1. Add iteration control to state
2. Implement decision logic in LeadResearcher
3. Add conditional routing for research loop
4. Test with queries requiring multiple iterations

**Deliverable:** System that can refine research based on findings

### 3.3 Phase 3: Memory & Citations (Week 2)
**Goal:** Add persistence and attribution

**Tasks:**
1. Implement basic memory system for plan storage
2. Create CitationAgent node
3. Add citation extraction from sources
4. Format final report with citations

**Deliverable:** Complete MVP with citations

### 3.4 Phase 4: Prompt Engineering & Testing (Week 2-3)
**Goal:** Optimize prompts and validate system

**Tasks:**
1. Refine LeadResearcher prompts (delegation, scaling)
2. Optimize Subagent prompts (task clarity)
3. Create test suite with various query types
4. Performance evaluation

**Deliverable:** Polished MVP ready for demonstration

## 4. Technical Decisions

### 4.1 Search Tool
**Decision:** Use Tavily API (already in dependencies)
**Rationale:** 
- Already available in project
- Good web search capabilities
- Simple integration

### 4.2 Parallelization
**Decision:** Use LangGraph's `Send()` API
**Rationale:**
- Already proven in existing parallel-agent patterns
- True parallel execution
- Automatic result aggregation

### 4.3 Memory System
**Decision:** Start with in-memory state, add persistence later
**Rationale:**
- MVP focus on core functionality
- LangGraph state is sufficient for initial version
- Can upgrade to external storage post-MVP

### 4.4 Model Selection
**Decision:** Use same model (qwen-plus) for all agents initially
**Rationale:**
- Consistency
- Cost efficiency
- Can differentiate later (e.g., stronger model for LeadResearcher)

## 5. Key Challenges & Solutions

### 5.1 Challenge: Agent Coordination
**Problem:** Subagents might duplicate work or miss information
**Solution:** 
- Detailed task descriptions from LeadResearcher
- Clear division of labor in prompts
- Explicit guidance on tools and sources

### 5.2 Challenge: Appropriate Effort Scaling
**Problem:** Agents might over-invest in simple queries
**Solution:**
- Embed scaling rules in prompts
- Simple queries: 1 agent, 3-10 tool calls
- Complex queries: 2-4 subagents, 10-15 calls each

### 5.3 Challenge: Context Management
**Problem:** Long conversations exceed context limits
**Solution (MVP):**
- Store research plan in state
- Basic summarization between iterations
- Future: External memory, compression

### 5.4 Challenge: Non-deterministic Behavior
**Problem:** Agents behave differently between runs
**Solution:**
- Comprehensive logging
- State inspection
- Test with multiple runs

## 6. Success Metrics

### 6.1 Functional Metrics
- ✅ System can handle breadth-first queries
- ✅ Parallel subagents execute successfully
- ✅ Results are synthesized coherently
- ✅ Citations are properly attributed

### 6.2 Quality Metrics
- Research quality (subjective evaluation)
- Citation accuracy
- Response completeness

### 6.3 Performance Metrics
- Execution time
- Token usage
- Number of tool calls

## 7. Dependencies & Prerequisites

### 7.1 Existing Infrastructure
- ✅ LangGraph for workflow orchestration
- ✅ Tavily for web search
- ✅ Parallel-agent patterns as reference
- ✅ Orchestrator-worker pattern as reference

### 7.2 New Requirements
- None (all dependencies already available)

## 8. Next Steps

1. **Review this analysis** - Validate requirements and architecture
2. **Create project structure** - Set up `research-agent/` directory
3. **Implement Phase 1** - Core orchestration with search
4. **Iterate and refine** - Build incrementally, test frequently

## 9. References

- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- Existing patterns in this codebase:
  - `parallel-agent/` - Sectioning and Voting patterns
  - `orchestrator-worker/` - Basic orchestrator pattern
- LangGraph documentation for `Send()` and parallelization

---

**Document Status:** Draft for Review
**Last Updated:** Initial Analysis
**Next Review:** After architecture approval

