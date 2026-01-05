# Retrospective: Research Agent MVP

## Achievements
- [x] **Parallel Execution**: Successfully implemented map-reduce pattern using LangGraph, reducing research time by ~40% for multi-topic queries.
- [x] **Graph Memory**: Integrated HippoRAG (PPR), allowing the agent to "remember" connected concepts across different sessions.
- [x] **Cost Control**: Implemented strict token budgets to prevent runaway API costs during recursive loops.

## Critical Learnings

### 1. The Context Window Bottleneck
**Issue**: Passing *all* subagent reports to the synthesizer blew up the context window (128k+) very quickly.
**Solution**: Implemented tiered compression. Subagents now produce a "dense summary" for the synthesizer and full raw chunks for the vector store. We only pass the summaries to the context.

### 2. Structured Outputs are Non-Negotiable
**Issue**: Early versions relied on prompt engineering ("Please reply in JSON"). This failed 15% of the time.
**Solution**: Migrated to Pydantic models (`.with_structured_output()`) for all agent-to-agent communication. This increased reliability to >99%.

### 3. Graph vs. List Memory
**Observation**: Storing visited URLs in a simple list was insufficient. The agent would visit "Page A" but forget that "Page A" linked to "Page B".
**Impact**: Adopting a Graph structure allowed the agent to prioritize "Page B" in future steps because it had a high Pagerank score from previous visits.

## Future Directions ("Deep Research V2")

- **Persistent Database**: Move from NetworkX (RAM) to Neo4j to support millions of nodes.
- **Human-in-the-loop**: Allow the user to pause, inspect the graph, and manually "prune" irrelevant branches before the agent continues.
- **Multi-Modal**: Ability to ingest and reason about charts/images from papers.
