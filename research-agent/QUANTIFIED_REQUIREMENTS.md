# Quantified Requirements Summary

## Quick Reference: All Metrics and Targets

This document provides a concise summary of all quantified requirements for the Research Agent MVP.

## Critical Metrics (Must Track)

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **End-to-End Latency** | < 60s | < 120s | Time from query to final report |
| **Query Success Rate** | 90%+ | 85%+ | % of queries that complete successfully |
| **Answer Completeness** | 80%+ | 70%+ | Manual evaluation: covers all aspects |
| **Token Usage** | 30k-50k | < 100k | Total tokens per query |
| **End-to-End Automation** | 70%+ | 60%+ | % completed without human intervention |

## Performance Metrics

### Latency Breakdown

| Component | Target | Acceptable |
|-----------|--------|------------|
| LeadResearcher (planning) | < 10s | < 20s |
| Subagent (search + analysis) | < 10s each | < 20s each |
| Parallel execution | Max(subagent times) | Max + 5s |
| Synthesis | < 15s | < 30s |
| Citation extraction | < 10s | < 20s |
| Decision (continue/exit) | < 5s | < 10s |
| **Total (End-to-End)** | **< 60s** | **< 120s** |

### Throughput

| Metric | Target | Acceptable |
|--------|--------|------------|
| Queries per minute | 2-3 | 1 |
| Concurrent subagents | 3-5 | 2-8 |
| Search results per subagent | 5-10 | 3-15 |

### Resource Usage

| Metric | Target | Acceptable | Hard Limit |
|--------|--------|------------|------------|
| Tokens per query | 30k-50k | 20k-80k | 200k |
| API calls per query | 5-15 | 3-25 | 50 |
| Memory usage | < 100MB | < 200MB | 500MB |
| Max subagents | 10 | 15 | 20 |
| Max iterations | 3 | 5 | 10 |

## Quality Metrics

### Research Quality

| Metric | Target | Acceptable | Evaluation |
|--------|--------|------------|------------|
| Answer completeness | 80%+ | 70%+ | Manual: covers all aspects? |
| Source relevance | 85%+ | 75%+ | Manual: sources relevant? |
| Citation accuracy | 90%+ | 80%+ | Manual: citations match claims? |
| Factual accuracy | 85%+ | 75%+ | Manual: facts correct? |
| Synthesis quality | 80%+ | 70%+ | Manual: coherent integration? |

### Search Quality

| Metric | Target | Acceptable |
|--------|--------|------------|
| Search result relevance | 80%+ | 70%+ |
| Source diversity | 3-5 unique domains | 2-7 |
| Information coverage | 80%+ | 70%+ |

### Coordination Quality

| Metric | Target | Acceptable |
|--------|--------|------------|
| Task overlap (duplication) | < 20% | < 30% |
| Task coverage | 100% | 95%+ |
| Delegation clarity | 85%+ | 75%+ |

## Automation Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| End-to-end automation | 70%+ | 60%+ |
| Search automation | 95%+ | 90%+ |
| Synthesis automation | 85%+ | 75%+ |
| Citation automation | 80%+ | 70%+ |
| Iteration decision accuracy | 80%+ | 70%+ |
| Subagent count accuracy | 75%+ | 65%+ |
| Effort scaling accuracy | 70%+ | 60%+ |

## Reliability Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| Query success rate | 90%+ | 85%+ |
| Search success rate | 95%+ | 90%+ |
| Tool call success rate | 98%+ | 95%+ |
| Synthesis success rate | 95%+ | 90%+ |
| Error recovery rate | 80%+ | 70%+ |
| Graceful degradation | 90%+ | 85%+ |

## Cost Metrics

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| Tokens per query | 30k-50k | 20k-80k | Varies by complexity |
| Cost per query | $0.10-$0.30 | $0.05-$0.50 | Based on qwen-plus pricing |
| Cost efficiency ratio | 15× chat baseline | 10-20× | Multi-agent overhead |
| Subagents per query | 2-4 | 1-6 | Optimal balance |
| Iterations per query | 1-2 | 1-3 | Most queries need 1 |

## Scalability Metrics

### Query Complexity Handling

| Complexity | Target | Acceptable | Definition |
|------------|--------|------------|------------|
| Simple queries | 100% | 95%+ | 1 subagent, 3-10 tool calls |
| Medium queries | 90%+ | 85%+ | 2-4 subagents, 10-15 calls each |
| Complex queries | 80%+ | 70%+ | 5+ subagents, multiple iterations |

### System Limits

| Limit | Target | Acceptable | Hard Limit |
|-------|--------|------------|------------|
| Max subagents | 10 | 15 | 20 |
| Max iterations | 3 | 5 | 10 |
| Max context length | 200k tokens | 300k tokens | 500k tokens |
| Max search results | 50 | 100 | 200 |

## User Experience Metrics

| Metric | Target | Acceptable |
|--------|--------|------------|
| Report readability | 85%+ | 75%+ |
| Report length | 500-2000 words | 300-3000 words |
| Citation format validity | 100% | 95%+ |
| Source accessibility | 90%+ | 80%+ |
| Progress update frequency | Every 10-15s | Every 20s |

## Testing Requirements

### Test Coverage

| Metric | Target | Acceptable |
|--------|--------|------------|
| Unit test coverage | 70%+ | 60%+ |
| Integration test coverage | 80%+ | 70%+ |
| Query type coverage | 10+ types | 5+ types |
| Edge case coverage | 15+ cases | 10+ cases |

### Test Suite Size

- **Total queries**: 20-30
- **Simple queries**: 5
- **Medium queries**: 5
- **Complex queries**: 5
- **Edge cases**: 5-10

## MVP Success Criteria

The MVP is considered successful if it achieves **ALL** of the following:

### ✅ Performance
- [ ] 80%+ of queries complete in < 120s
- [ ] Parallel execution works (subagents run concurrently)
- [ ] Average latency < 120s

### ✅ Quality
- [ ] 70%+ answer completeness
- [ ] 75%+ source relevance
- [ ] 80%+ citation accuracy

### ✅ Reliability
- [ ] 85%+ query success rate
- [ ] 90%+ search success rate
- [ ] 95%+ tool call success rate

### ✅ Automation
- [ ] 60%+ end-to-end automation
- [ ] System handles simple and medium queries
- [ ] Basic iterative loop works

### ✅ Functionality
- [ ] Parallel subagents execute searches
- [ ] Results are synthesized
- [ ] Citations are extracted
- [ ] Iterative loop works (at least 1 iteration)

## Measurement Implementation

### Required Components

1. **MetricsCollector** - Track all metrics
2. **Test Suite** - 20-30 diverse queries
3. **Performance Benchmarks** - Latency, parallelism verification
4. **Quality Evaluation** - Manual evaluation framework
5. **Monitoring Dashboard** - Real-time metrics
6. **Reporting** - Daily/weekly reports

### Measurement Frequency

- **Real-time**: Latency, success rate, token usage
- **Per Query**: All performance and resource metrics
- **Daily**: Aggregate statistics, cost analysis
- **Weekly**: Quality evaluation, error analysis
- **MVP Completion**: Full test suite, comprehensive report

## Cost Estimates & Budget Limits

### Per Query (Average)

| Component | Tokens | Cost Estimate | **Hard Limit** |
|-----------|--------|---------------|----------------|
| LeadResearcher | 5k-10k | $0.02-$0.05 | 20k tokens |
| Subagents (3-4) | 15k-30k | $0.06-$0.15 | 60k tokens |
| Synthesis | 5k-10k | $0.02-$0.05 | 15k tokens |
| Citations | 2k-5k | $0.01-$0.02 | 5k tokens |
| Search API | - | $0.01-$0.05 | 20 calls |
| **Total** | **30k-50k** | **$0.10-$0.30** | **100k tokens / $0.50** |

### Budget Limits (Hard Limits - Cannot Violate)

| Timeframe | Token Limit | Query Limit | Cost Limit | Action if Exceeded |
|-----------|-------------|-------------|------------|-------------------|
| **Per Query** | 100k | 1 | $0.50 | **Immediately stop** |
| **Daily** | 5M | 100 | $50 | **Reject new queries** |
| **Weekly** | 30M | 600 | $300 | **Degrade service** |
| **Monthly** | 120M | 2400 | $1200 | **Complete stop** |

### Monthly Estimates (100 queries/day)

- **Tokens**: ~90M-150M tokens/month (Budget: 120M)
- **Cost**: ~$300-$900/month (Budget: $1200)
- **Search API**: ~$30-$150/month

**⚠️ Important: All cost control measures are mandatory. See [COST_CONTROL.md](./COST_CONTROL.md) for details.**

## Key Performance Indicators (KPIs)

### Primary KPIs (Track Daily)

1. **End-to-End Latency** - User experience
2. **Query Success Rate** - System reliability
3. **Token Usage** - Cost control
4. **Automation Rate** - System effectiveness

### Secondary KPIs (Track Weekly)

1. **Quality Scores** - Research quality
2. **Error Rates** - System stability
3. **Cost per Query** - Economic viability
4. **User Satisfaction** - Overall value

## Decision Thresholds

### When to Continue Research

- Information coverage < 70%
- Key aspects missing
- Sources insufficient (< 3 relevant sources)
- User query explicitly asks for more depth

### When to Stop Research

- Information coverage ≥ 80%
- All key aspects covered
- Sufficient sources (≥ 3 relevant)
- Max iterations reached (3)
- Max subagents reached (10)

### Resource Allocation

| Query Complexity | Subagents | Tool Calls | Iterations |
|------------------|-----------|------------|------------|
| Simple | 1 | 3-10 | 1 |
| Medium | 2-4 | 10-15 each | 1-2 |
| Complex | 5-10 | 15-20 each | 2-3 |

## Risk Mitigation

### Performance Risks

| Risk | Mitigation | Target |
|------|------------|--------|
| Latency too high | Optimize prompts, limit iterations | < 120s |
| Token usage too high | Limit subagents, compress results | < 100k |
| Parallel execution fails | Verify Send() implementation | Concurrent execution |

### Quality Risks

| Risk | Mitigation | Target |
|------|------------|--------|
| Low completeness | Better task decomposition | 70%+ |
| Poor citations | Improved extraction | 80%+ |
| Inaccurate facts | Source verification | 75%+ |

### Reliability Risks

| Risk | Mitigation | Target |
|------|------------|--------|
| Tool failures | Retry logic, error handling | 95%+ success |
| Context overflow | Memory management | < 200k tokens |
| Infinite loops | Max iteration limits | Max 3 iterations |

---

## Summary Table: All Targets

| Category | Key Metric | Target | Acceptable | Critical? |
|----------|-----------|--------|------------|-----------|
| **Performance** | End-to-end latency | < 60s | < 120s | ✅ Yes |
| **Quality** | Answer completeness | 80%+ | 70%+ | ✅ Yes |
| **Reliability** | Query success rate | 90%+ | 85%+ | ✅ Yes |
| **Automation** | End-to-end automation | 70%+ | 60%+ | ✅ Yes |
| **Cost** | Tokens per query | 30k-50k | < 100k | ⚠️ Important |
| **Scalability** | Complex query handling | 80%+ | 70%+ | ⚠️ Important |
| **UX** | Report readability | 85%+ | 75%+ | ⚠️ Important |

---

**Document Status**: Complete
**Last Updated**: Initial Quantification
**Next Review**: After MVP implementation begins

