# Metrics and KPIs: Research Agent MVP

## Executive Summary

This document defines quantifiable requirements, success metrics, and performance targets for the multi-agent research system MVP. All metrics are designed to be measurable and actionable.

## 1. Performance Metrics

### 1.1 Latency Targets

| Metric | Target | Acceptable | Measurement Method |
|--------|--------|------------|-------------------|
| **End-to-End Latency** | < 60s | < 120s | Time from query to final report |
| **Subagent Search Latency** | < 10s per subagent | < 20s | Time for single subagent to complete search |
| **Parallel Execution Time** | Max(subagent times) | Max(subagent times) + 5s | Time for all subagents to complete |
| **Synthesis Latency** | < 15s | < 30s | Time to synthesize all findings |
| **Citation Extraction** | < 10s | < 20s | Time to extract and format citations |
| **Decision Latency** | < 5s | < 10s | Time for LeadResearcher to decide next action |

**Rationale:**
- Anthropic mentions agents use ~4× more tokens than chat, multi-agent ~15×
- For MVP, we target reasonable user experience (< 2 minutes for complex queries)
- Parallel execution should minimize total time

### 1.2 Throughput Metrics

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| **Queries per Minute** | 2-3 | 1 | Limited by LLM rate limits |
| **Concurrent Subagents** | 3-5 | 2-8 | Balance between speed and cost |
| **Search Results per Subagent** | 5-10 | 3-15 | Quality over quantity |

### 1.3 Resource Utilization

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Token Usage per Query** | < 50k tokens | < 100k tokens | Track total tokens used |
| **Token Usage Ratio** | 10-15× chat baseline | 5-20× | Multi-agent overhead |
| **API Calls per Query** | 5-15 | 3-25 | Count of LLM + search API calls |
| **Memory Usage** | < 100MB | < 200MB | In-memory state size |

## 2. Quality Metrics

### 2.1 Research Quality

| Metric | Target | Acceptable | Evaluation Method |
|--------|--------|------------|-------------------|
| **Answer Completeness** | 80%+ | 70%+ | Manual evaluation: covers all aspects? |
| **Source Relevance** | 85%+ | 75%+ | Manual: are sources relevant to query? |
| **Citation Accuracy** | 90%+ | 80%+ | Manual: do citations match claims? |
| **Factual Accuracy** | 85%+ | 75%+ | Manual: are facts correct? |
| **Synthesis Quality** | 80%+ | 70%+ | Manual: coherent integration of findings? |

**Evaluation Method:**
- Create test suite of 20-30 diverse queries
- Manual evaluation by 2+ reviewers
- Score each dimension 0-100%
- Average across test suite

### 2.2 Search Quality

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Search Result Relevance** | 80%+ | 70%+ | % of results that are relevant |
| **Source Diversity** | 3-5 unique domains | 2-7 | Number of different sources |
| **Information Coverage** | 80%+ | 70%+ | % of query aspects covered |

### 2.3 Agent Coordination

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Task Overlap** | < 20% | < 30% | % of duplicate work between subagents |
| **Task Coverage** | 100% | 95%+ | % of query aspects assigned to subagents |
| **Delegation Clarity** | 85%+ | 75%+ | Manual: are subagent tasks clear? |

## 3. Automation Metrics

### 3.1 Automation Rate

| Metric | Target | Acceptable | Definition |
|--------|--------|------------|------------|
| **End-to-End Automation** | 70%+ | 60%+ | % of queries completed without human intervention |
| **Search Automation** | 95%+ | 90%+ | % of searches that don't need manual refinement |
| **Synthesis Automation** | 85%+ | 75%+ | % of syntheses that don't need manual editing |
| **Citation Automation** | 80%+ | 70%+ | % of citations correctly extracted automatically |

**Test Suite:**
- 50 diverse research queries
- Measure % that complete successfully without manual intervention
- Track intervention types (search refinement, synthesis editing, etc.)

### 3.2 Decision Accuracy

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Iteration Decision Accuracy** | 80%+ | 70%+ | % of correct "continue/exit" decisions |
| **Subagent Count Accuracy** | 75%+ | 65%+ | Appropriate number of subagents for query complexity |
| **Effort Scaling Accuracy** | 70%+ | 60%+ | Appropriate resource allocation |

**Evaluation:**
- For each query, determine optimal number of iterations
- Compare system decision vs optimal
- Measure accuracy

## 4. Reliability Metrics

### 4.1 Success Rate

| Metric | Target | Acceptable | Definition |
|--------|--------|------------|------------|
| **Query Success Rate** | 90%+ | 85%+ | % of queries that complete successfully |
| **Search Success Rate** | 95%+ | 90%+ | % of searches that return results |
| **Tool Call Success Rate** | 98%+ | 95%+ | % of tool calls that succeed |
| **Synthesis Success Rate** | 95%+ | 90%+ | % of syntheses that complete |

### 4.2 Error Handling

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Error Recovery Rate** | 80%+ | 70%+ | % of errors that are automatically recovered |
| **Graceful Degradation** | 90%+ | 85%+ | % of partial failures that still produce useful output |
| **Error Types Tracked** | 100% | 95%+ | All error types are logged and categorized |

### 4.3 Consistency

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Result Consistency** | 70%+ | 60%+ | Similar queries produce similar quality results |
| **Deterministic Behavior** | N/A | N/A | Non-deterministic by design, but track variance |

## 5. Cost Metrics

### 5.1 Token Economics

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| **Tokens per Query** | 30k-50k | 20k-80k | Varies by query complexity |
| **Cost per Query** | $0.10-$0.30 | $0.05-$0.50 | Based on qwen-plus pricing |
| **Cost Efficiency** | 15× chat baseline | 10-20× | Acceptable overhead for multi-agent |

**Cost Calculation:**
- Input tokens: ~10k-20k per query
- Output tokens: ~5k-15k per query
- Search API: ~$0.01-0.05 per query
- Total: Estimate based on provider pricing

### 5.2 Resource Efficiency

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Subagents per Query** | 2-4 | 1-6 | Optimal balance |
| **Iterations per Query** | 1-2 | 1-3 | Most queries need 1 iteration |
| **Search Calls per Subagent** | 2-5 | 1-8 | Efficient search strategy |

## 6. Scalability Metrics

### 6.1 Query Complexity Handling

| Metric | Target | Acceptable | Definition |
|--------|--------|------------|------------|
| **Simple Query Handling** | 100% | 95%+ | Queries needing 1 subagent, 3-10 tool calls |
| **Medium Query Handling** | 90%+ | 85%+ | Queries needing 2-4 subagents, 10-15 calls each |
| **Complex Query Handling** | 80%+ | 70%+ | Queries needing 5+ subagents, multiple iterations |

**Query Complexity Classification:**
- **Simple**: Fact-finding, single topic (e.g., "What is X?")
- **Medium**: Comparison, multi-aspect (e.g., "Compare X and Y")
- **Complex**: Broad research, multiple independent directions (e.g., "Research all board members of IT S&P 500 companies")

### 6.2 System Limits

| Metric | Target | Acceptable | Hard Limit |
|--------|--------|------------|------------|
| **Max Subagents** | 10 | 15 | 20 |
| **Max Iterations** | 3 | 5 | 10 |
| **Max Context Length** | 200k tokens | 300k tokens | 500k tokens |
| **Max Search Results** | 50 | 100 | 200 |

## 7. User Experience Metrics

### 7.1 Response Quality

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Report Readability** | 85%+ | 75%+ | Manual evaluation: clear, well-structured? |
| **Report Length** | 500-2000 words | 300-3000 words | Appropriate for query complexity |
| **Citation Format** | 100% valid | 95%+ | All citations follow format |
| **Source Accessibility** | 90%+ | 80%+ | % of sources that are accessible/valid URLs |

### 7.2 Progress Visibility

| Metric | Target | Acceptable | Notes |
|--------|--------|------------|-------|
| **Progress Updates** | Every 10-15s | Every 20s | User should see progress |
| **Status Messages** | Clear & informative | Basic | What is system doing? |
| **Error Messages** | Actionable | Informative | Help user understand issues |

## 8. Testing and Validation Metrics

### 8.1 Test Coverage

| Metric | Target | Acceptable | Measurement |
|--------|--------|------------|-------------|
| **Unit Test Coverage** | 70%+ | 60%+ | Code coverage percentage |
| **Integration Test Coverage** | 80%+ | 70%+ | All major workflows tested |
| **Query Type Coverage** | 10+ types | 5+ types | Different query categories |
| **Edge Case Coverage** | 15+ cases | 10+ cases | Error scenarios, edge cases |

### 8.2 Test Suite

**Query Categories (20-30 queries total):**
1. Simple fact-finding (5 queries)
2. Comparison (5 queries)
3. Multi-aspect research (5 queries)
4. Complex breadth-first (5 queries)
5. Edge cases (5-10 queries)

**Example Queries:**
- Simple: "What is LangGraph?"
- Comparison: "Compare Python and Rust for web development"
- Multi-aspect: "Research the pros and cons of microservices"
- Complex: "Find all board members of top 10 AI companies"
- Edge: Empty results, tool failures, etc.

## 9. Monitoring and Observability

### 9.1 Metrics to Track

**Real-time Metrics:**
- Query latency (p50, p95, p99)
- Token usage per query
- Success/failure rates
- Active subagents count
- Iteration counts

**Aggregated Metrics (Daily/Weekly):**
- Average query complexity
- Cost per query
- Quality scores (from evaluations)
- Error rates by type
- User satisfaction (if available)

### 9.2 Logging Requirements

**Must Log:**
- All query inputs
- Subagent tasks and findings
- Tool calls and results
- Decision points (continue/exit)
- Errors and exceptions
- Performance metrics (latency, tokens)

**Privacy Considerations:**
- Don't log full conversation content in production
- Log high-level patterns and metrics
- Anonymize user data

## 10. MVP Success Criteria

### 10.1 Minimum Viable Metrics

The MVP is considered successful if it achieves:

✅ **Performance:**
- 80%+ of queries complete in < 120s
- Parallel execution works (subagents run concurrently)

✅ **Quality:**
- 70%+ answer completeness
- 75%+ source relevance
- 80%+ citation accuracy

✅ **Reliability:**
- 85%+ query success rate
- 90%+ search success rate

✅ **Automation:**
- 60%+ end-to-end automation
- System handles simple and medium queries

✅ **Functionality:**
- Parallel subagents execute searches
- Results are synthesized
- Citations are extracted
- Iterative loop works (at least 1 iteration)

### 10.2 Stretch Goals

**If MVP exceeds targets, consider:**
- Handling complex queries (5+ subagents)
- Advanced memory management
- Better error recovery
- More sophisticated citation extraction

## 11. Measurement Plan

### 11.1 Baseline

**Before MVP:**
- Measure baseline chat performance (single agent)
- Establish token usage baseline
- Define test query suite

### 11.2 During Development

**Continuous Measurement:**
- Track metrics during development
- Run test suite after each major change
- Compare against targets

### 11.3 Post-MVP

**Validation:**
- Run full test suite
- Manual evaluation of quality metrics
- Performance benchmarking
- Cost analysis

## 12. Metric Definitions

### 12.1 Latency Definitions

- **End-to-End Latency**: Time from `app.invoke(query)` to final report
- **Subagent Latency**: Time from subagent start to findings return
- **Parallel Execution Time**: Time for all subagents to complete (should be max, not sum)

### 12.2 Quality Definitions

- **Completeness**: Does answer cover all aspects of query?
- **Relevance**: Are sources and information relevant?
- **Accuracy**: Are facts correct?
- **Synthesis**: Are findings integrated coherently?

### 12.3 Automation Definitions

- **End-to-End Automation**: Query completes without manual intervention
- **Partial Automation**: Query completes but needs manual refinement
- **Manual Required**: Query needs significant manual work

## 13. Reporting

### 13.1 Daily Metrics Dashboard

Track daily:
- Queries processed
- Average latency
- Success rate
- Token usage
- Cost

### 13.2 Weekly Quality Report

Weekly evaluation:
- Quality scores (completeness, relevance, accuracy)
- Error analysis
- User feedback (if available)
- Improvement opportunities

### 13.3 MVP Completion Report

Final report should include:
- All metrics vs targets
- Test suite results
- Known limitations
- Recommendations for improvements

---

## Appendix: Quick Reference

### Critical Metrics (Must Track)
1. End-to-end latency (< 120s acceptable)
2. Query success rate (85%+ acceptable)
3. Answer completeness (70%+ acceptable)
4. Token usage (< 100k acceptable)
5. Automation rate (60%+ acceptable)

### Measurement Tools
- LangGraph state inspection
- Custom logging/monitoring
- Manual evaluation sheets
- Performance profiling

### Target Summary Table

| Category | Key Metric | Target | Acceptable |
|----------|-----------|--------|------------|
| **Performance** | End-to-end latency | < 60s | < 120s |
| **Quality** | Answer completeness | 80%+ | 70%+ |
| **Reliability** | Query success rate | 90%+ | 85%+ |
| **Automation** | End-to-end automation | 70%+ | 60%+ |
| **Cost** | Tokens per query | 30k-50k | 20k-80k |

---

**Document Status**: Ready for Review
**Next Step**: Define measurement implementation plan

