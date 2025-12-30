# Measurement Implementation Plan

## Overview

This document outlines how to implement measurement and tracking for all defined metrics in the Research Agent MVP.

## 1. Measurement Infrastructure

### 1.1 Logging Framework

**Implementation:**
```python
# research-agent/metrics.py
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class QueryMetrics:
    """Track metrics for a single query"""
    query_id: str
    query: str
    start_time: float
    end_time: Optional[float] = None
    
    # Latency metrics
    lead_researcher_latency: float = 0.0
    subagent_latencies: List[float] = field(default_factory=list)
    synthesis_latency: float = 0.0
    citation_latency: float = 0.0
    
    # Resource metrics
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    search_calls: int = 0
    
    # Quality metrics (set post-execution)
    completeness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    
    # Coordination metrics
    num_subagents: int = 0
    num_iterations: int = 0
    task_overlap: Optional[float] = None
    
    # Success metrics
    success: bool = False
    error_type: Optional[str] = None
    
    def calculate_total_latency(self) -> float:
        """Calculate end-to-end latency"""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "query_id": self.query_id,
            "query": self.query[:100],  # Truncate for privacy
            "latency": self.calculate_total_latency(),
            "tokens": self.total_tokens,
            "subagents": self.num_subagents,
            "iterations": self.num_iterations,
            "success": self.success,
            "timestamp": datetime.now().isoformat(),
        }
```

### 1.2 Metrics Collector

**Implementation:**
```python
# research-agent/metrics.py (continued)

class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics: List[QueryMetrics] = []
        self.logger = logging.getLogger("research_agent.metrics")
    
    def start_query(self, query: str) -> QueryMetrics:
        """Start tracking a new query"""
        query_id = f"query_{int(time.time() * 1000)}"
        metrics = QueryMetrics(
            query_id=query_id,
            query=query,
            start_time=time.time()
        )
        self.metrics.append(metrics)
        return metrics
    
    def record_subagent_latency(self, metrics: QueryMetrics, latency: float):
        """Record subagent execution time"""
        metrics.subagent_latencies.append(latency)
        metrics.num_subagents += 1
    
    def record_tokens(self, metrics: QueryMetrics, input_tokens: int, output_tokens: int):
        """Record token usage"""
        metrics.input_tokens += input_tokens
        metrics.output_tokens += output_tokens
        metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
    
    def finish_query(self, metrics: QueryMetrics, success: bool = True, error: Optional[str] = None):
        """Mark query as complete"""
        metrics.end_time = time.time()
        metrics.success = success
        metrics.error_type = error
        self.logger.info(f"Query completed: {metrics.to_dict()}")
    
    def get_statistics(self) -> Dict:
        """Calculate aggregate statistics"""
        if not self.metrics:
            return {}
        
        successful = [m for m in self.metrics if m.success]
        
        return {
            "total_queries": len(self.metrics),
            "successful_queries": len(successful),
            "success_rate": len(successful) / len(self.metrics) if self.metrics else 0,
            "avg_latency": sum(m.calculate_total_latency() for m in successful) / len(successful) if successful else 0,
            "p95_latency": self._percentile([m.calculate_total_latency() for m in successful], 0.95),
            "avg_tokens": sum(m.total_tokens for m in successful) / len(successful) if successful else 0,
            "avg_subagents": sum(m.num_subagents for m in successful) / len(successful) if successful else 0,
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
```

### 1.3 Integration with LangGraph

**Usage in graph nodes:**
```python
# research-agent/graph.py
from metrics import MetricsCollector

collector = MetricsCollector()

def lead_researcher_node(state: ResearchState):
    metrics = state.get("_metrics")
    start_time = time.time()
    
    # ... node logic ...
    
    latency = time.time() - start_time
    metrics.lead_researcher_latency = latency
    collector.record_tokens(metrics, input_tokens, output_tokens)
    
    return {"_metrics": metrics, ...}
```

## 2. Automated Testing Framework

### 2.1 Test Query Suite

**Implementation:**
```python
# research-agent/test_suite.py

TEST_QUERIES = {
    "simple": [
        "What is LangGraph?",
        "Who created Python?",
        "What is the capital of France?",
    ],
    "medium": [
        "Compare Python and Rust for web development",
        "Research the pros and cons of microservices architecture",
        "Analyze the differences between REST and GraphQL APIs",
    ],
    "complex": [
        "Research all board members of the top 10 AI companies",
        "Find information about the history, current state, and future of quantum computing",
        "Compare the top 5 cloud providers across pricing, features, and reliability",
    ],
    "edge_cases": [
        "",  # Empty query
        "xyz123nonexistenttopic456",  # No results expected
        "A" * 1000,  # Very long query
    ]
}

def run_test_suite(app, collector: MetricsCollector):
    """Run full test suite"""
    results = {}
    
    for category, queries in TEST_QUERIES.items():
        category_results = []
        for query in queries:
            metrics = collector.start_query(query)
            try:
                result = app.invoke({"query": query, "_metrics": metrics})
                collector.finish_query(metrics, success=True)
                category_results.append({
                    "query": query,
                    "success": True,
                    "latency": metrics.calculate_total_latency(),
                    "tokens": metrics.total_tokens,
                })
            except Exception as e:
                collector.finish_query(metrics, success=False, error=str(e))
                category_results.append({
                    "query": query,
                    "success": False,
                    "error": str(e),
                })
        results[category] = category_results
    
    return results
```

### 2.2 Performance Benchmarks

**Implementation:**
```python
# research-agent/benchmarks.py

def benchmark_latency(app, num_runs: int = 10):
    """Benchmark average latency"""
    latencies = []
    
    test_query = "What is LangGraph?"
    
    for _ in range(num_runs):
        start = time.time()
        app.invoke({"query": test_query})
        latencies.append(time.time() - start)
    
    return {
        "avg": sum(latencies) / len(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "p95": percentile(latencies, 0.95),
    }

def benchmark_parallel_execution(app):
    """Verify parallel execution"""
    # Test with query that should spawn multiple subagents
    query = "Compare Python, Rust, and Go"
    
    start = time.time()
    result = app.invoke({"query": query})
    total_time = time.time() - start
    
    # If truly parallel, total time should be ~max(subagent times)
    # If sequential, total time would be ~sum(subagent times)
    
    metrics = result.get("_metrics")
    if metrics:
        max_subagent_time = max(metrics.subagent_latencies) if metrics.subagent_latencies else 0
        sequential_estimate = sum(metrics.subagent_latencies) if metrics.subagent_latencies else 0
        
        return {
            "total_time": total_time,
            "max_subagent_time": max_subagent_time,
            "sequential_estimate": sequential_estimate,
            "parallelism_ratio": sequential_estimate / total_time if total_time > 0 else 0,
        }
```

## 3. Quality Evaluation Framework

### 3.1 Manual Evaluation Sheet

**Template:**
```markdown
# Quality Evaluation Sheet

## Query: [Query text]

### Completeness (0-100)
- [ ] Covers all aspects: __/100
- Notes: [Comments]

### Source Relevance (0-100)
- [ ] Sources are relevant: __/100
- Notes: [Comments]

### Citation Accuracy (0-100)
- [ ] Citations match claims: __/100
- Notes: [Comments]

### Factual Accuracy (0-100)
- [ ] Facts are correct: __/100
- Notes: [Comments]

### Synthesis Quality (0-100)
- [ ] Coherent integration: __/100
- Notes: [Comments]

### Overall Assessment
- Strengths: [List]
- Weaknesses: [List]
- Recommendations: [List]
```

### 3.2 Automated Quality Checks

**Implementation:**
```python
# research-agent/quality_checks.py

def check_citation_format(citations: List[Dict]) -> Dict:
    """Check if citations follow expected format"""
    issues = []
    valid_count = 0
    
    for citation in citations:
        required_fields = ["source", "url", "claim"]
        missing = [f for f in required_fields if f not in citation]
        if missing:
            issues.append(f"Missing fields: {missing}")
        else:
            valid_count += 1
    
    return {
        "valid_count": valid_count,
        "total_count": len(citations),
        "validity_rate": valid_count / len(citations) if citations else 0,
        "issues": issues,
    }

def check_source_accessibility(sources: List[str]) -> Dict:
    """Check if source URLs are accessible"""
    import requests
    
    accessible = 0
    inaccessible = []
    
    for url in sources:
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                accessible += 1
            else:
                inaccessible.append(url)
        except:
            inaccessible.append(url)
    
    return {
        "accessible": accessible,
        "inaccessible": len(inaccessible),
        "accessibility_rate": accessible / len(sources) if sources else 0,
        "inaccessible_urls": inaccessible,
    }
```

## 4. Real-time Monitoring

### 4.1 Metrics Dashboard (Simple)

**Implementation:**
```python
# research-agent/dashboard.py

def print_metrics_dashboard(collector: MetricsCollector):
    """Print simple metrics dashboard"""
    stats = collector.get_statistics()
    
    print("\n" + "=" * 80)
    print("RESEARCH AGENT METRICS DASHBOARD")
    print("=" * 80)
    print(f"Total Queries: {stats.get('total_queries', 0)}")
    print(f"Success Rate: {stats.get('success_rate', 0) * 100:.1f}%")
    print(f"Avg Latency: {stats.get('avg_latency', 0):.2f}s")
    print(f"P95 Latency: {stats.get('p95_latency', 0):.2f}s")
    print(f"Avg Tokens: {stats.get('avg_tokens', 0):,.0f}")
    print(f"Avg Subagents: {stats.get('avg_subagents', 0):.1f}")
    print("=" * 80)
```

### 4.2 Progress Tracking

**Implementation:**
```python
# research-agent/progress.py

class ProgressTracker:
    """Track and display progress"""
    
    def __init__(self):
        self.stages = []
    
    def update(self, stage: str, details: str = ""):
        """Update progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.stages.append((timestamp, stage, details))
        print(f"[{timestamp}] {stage}: {details}")
    
    def get_timeline(self) -> List[Dict]:
        """Get progress timeline"""
        return [
            {"time": t, "stage": s, "details": d}
            for t, s, d in self.stages
        ]
```

## 5. Reporting

### 5.1 Daily Report Generator

**Implementation:**
```python
# research-agent/reporting.py

def generate_daily_report(collector: MetricsCollector) -> str:
    """Generate daily metrics report"""
    stats = collector.get_statistics()
    
    report = f"""
# Daily Metrics Report - {datetime.now().strftime('%Y-%m-%d')}

## Summary
- Total Queries: {stats.get('total_queries', 0)}
- Success Rate: {stats.get('success_rate', 0) * 100:.1f}%
- Average Latency: {stats.get('avg_latency', 0):.2f}s
- P95 Latency: {stats.get('p95_latency', 0):.2f}s

## Resource Usage
- Average Tokens: {stats.get('avg_tokens', 0):,.0f}
- Average Subagents: {stats.get('avg_subagents', 0):.1f}

## Targets vs Actual
- Latency Target: < 120s | Actual: {stats.get('avg_latency', 0):.2f}s {'✅' if stats.get('avg_latency', 0) < 120 else '❌'}
- Success Target: > 85% | Actual: {stats.get('success_rate', 0) * 100:.1f}% {'✅' if stats.get('success_rate', 0) > 0.85 else '❌'}
"""
    return report
```

### 5.2 MVP Completion Report Template

**Template:**
```markdown
# MVP Completion Report

## Date: [Date]

## Metrics Summary

### Performance
- End-to-End Latency: [Actual] / [Target: < 120s]
- Success Rate: [Actual]% / [Target: 85%+]

### Quality (Manual Evaluation)
- Completeness: [Actual]% / [Target: 70%+]
- Relevance: [Actual]% / [Target: 75%+]
- Accuracy: [Actual]% / [Target: 80%+]

### Automation
- End-to-End Automation: [Actual]% / [Target: 60%+]

### Resources
- Avg Tokens: [Actual] / [Target: < 100k]

## Test Results
- Test Suite: [X]/[Total] passed
- Edge Cases: [X]/[Total] handled

## Known Limitations
1. [Limitation 1]
2. [Limitation 2]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
```

## 6. Implementation Checklist

### Phase 1: Basic Metrics
- [ ] Create `MetricsCollector` class
- [ ] Integrate with graph nodes
- [ ] Track latency metrics
- [ ] Track token usage
- [ ] Basic logging

### Phase 2: Testing Framework
- [ ] Create test query suite
- [ ] Implement test runner
- [ ] Add performance benchmarks
- [ ] Verify parallel execution

### Phase 3: Quality Evaluation
- [ ] Create evaluation sheet template
- [ ] Implement automated quality checks
- [ ] Citation format validation
- [ ] Source accessibility checks

### Phase 4: Monitoring & Reporting
- [ ] Real-time metrics dashboard
- [ ] Progress tracking
- [ ] Daily report generator
- [ ] MVP completion report

## 7. Usage Examples

### 7.1 Basic Usage

```python
from research_agent.graph import app
from research_agent.metrics import MetricsCollector

collector = MetricsCollector()

# Run query with metrics
metrics = collector.start_query("What is LangGraph?")
result = app.invoke({"query": "What is LangGraph?", "_metrics": metrics})
collector.finish_query(metrics, success=True)

# View statistics
stats = collector.get_statistics()
print(stats)
```

### 7.2 Running Test Suite

```python
from research_agent.test_suite import run_test_suite
from research_agent.metrics import MetricsCollector

collector = MetricsCollector()
results = run_test_suite(app, collector)

# Analyze results
for category, category_results in results.items():
    print(f"\n{category}:")
    for result in category_results:
        print(f"  {result['query'][:50]}... - {result.get('latency', 0):.2f}s")
```

### 7.3 Quality Evaluation

```python
from research_agent.quality_checks import check_citation_format, check_source_accessibility

# Check citations
citations = result.get("citations", [])
citation_check = check_citation_format(citations)
print(f"Citation validity: {citation_check['validity_rate'] * 100:.1f}%")

# Check sources
sources = [c.get("url") for c in citations if c.get("url")]
accessibility = check_source_accessibility(sources)
print(f"Source accessibility: {accessibility['accessibility_rate'] * 100:.1f}%")
```

---

**Next Steps:**
1. Implement `MetricsCollector` class
2. Integrate with graph nodes
3. Create test suite
4. Set up monitoring dashboard

