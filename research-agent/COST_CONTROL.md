# Cost Control & Budget Management

## Cost Control Strategy

This document defines strict cost control measures to prevent token consumption from getting out of control.

## 1. Hard Budget Limits

### 1.1 Per-Query Budget

| Limit Type | Hard Limit | Warning Threshold | Auto Stop |
|------------|------------|-------------------|-----------|
| **Total Tokens** | 100k | 80k | 100k |
| **Input Tokens** | 60k | 50k | 60k |
| **Output Tokens** | 40k | 30k | 40k |
| **API Calls** | 30 | 25 | 30 |
| **Subagents** | 8 | 6 | 8 |
| **Iterations** | 3 | 2 | 3 |
| **Search Calls** | 20 | 15 | 20 |

**Enforcement Rules:**
- Reaching hard limit → **Immediately stop**, return current results
- Reaching warning threshold → **Log warning**, continue execution but limit subsequent resources
- Each component has independent budget, cannot exceed

### 1.2 Daily/Monthly Budget

| Timeframe | Token Budget | Query Limit | Cost Budget |
|-----------|--------------|-------------|-------------|
| **Daily** | 5M tokens | 100 queries | $50 |
| **Weekly** | 30M tokens | 600 queries | $300 |
| **Monthly** | 120M tokens | 2400 queries | $1200 |

**Enforcement Rules:**
- Reaching daily limit → **Reject new queries**, return "Budget exhausted"
- Reaching weekly limit → **Degrade service** (reduce subagent count)
- Reaching monthly limit → **Complete stop**, requires manual reset

## 2. Cost Monitoring & Alerts

### 2.1 Real-Time Monitoring Metrics

**Must track in real-time:**
```python
class CostMonitor:
    """Real-time cost monitoring"""
    
    # Per-query metrics
    current_query_tokens: int = 0
    current_query_cost: float = 0.0
    current_query_api_calls: int = 0
    
    # Cumulative metrics
    daily_tokens: int = 0
    daily_cost: float = 0.0
    daily_queries: int = 0
    
    # Alert thresholds
    WARNING_THRESHOLD = 0.8  # 80% budget usage
    CRITICAL_THRESHOLD = 0.95  # 95% budget usage
```

### 2.2 Alert Mechanism

| Alert Level | Trigger Condition | Action |
|------------|-------------------|--------|
| **Warning** | Per-query > 80k tokens | Log warning, limit subsequent resources |
| **Critical** | Per-query > 95k tokens | Log warning, prepare to stop |
| **Stop** | Per-query > 100k tokens | **Immediately stop**, return results |
| **Daily Warning** | Daily budget > 80% | Send alert, limit new queries |
| **Daily Stop** | Daily budget > 100% | **Reject all new queries** |

### 2.3 Cost Tracking Implementation

```python
# research-agent/cost_control.py

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional
import json
import os

@dataclass
class QueryBudget:
    """Per-query budget"""
    max_tokens: int = 100_000
    max_input_tokens: int = 60_000
    max_output_tokens: int = 40_000
    max_api_calls: int = 30
    max_subagents: int = 8
    max_iterations: int = 3
    max_search_calls: int = 20
    
    current_tokens: int = 0
    current_input_tokens: int = 0
    current_output_tokens: int = 0
    current_api_calls: int = 0
    current_subagents: int = 0
    current_iterations: int = 0
    current_search_calls: int = 0
    
    def check_limit(self, resource: str, amount: int) -> tuple[bool, str]:
        """Check if limit would be exceeded"""
        limits = {
            "tokens": (self.max_tokens, self.current_tokens),
            "input_tokens": (self.max_input_tokens, self.current_input_tokens),
            "output_tokens": (self.max_output_tokens, self.current_output_tokens),
            "api_calls": (self.max_api_calls, self.current_api_calls),
            "subagents": (self.max_subagents, self.current_subagents),
            "iterations": (self.max_iterations, self.current_iterations),
            "search_calls": (self.max_search_calls, self.current_search_calls),
        }
        
        if resource not in limits:
            return True, "Unknown resource"
        
        max_val, current_val = limits[resource]
        new_total = current_val + amount
        
        if new_total > max_val:
            return False, f"{resource} limit exceeded: {new_total} > {max_val}"
        
        return True, "OK"
    
    def consume(self, resource: str, amount: int) -> tuple[bool, str]:
        """Consume resources"""
        can_consume, message = self.check_limit(resource, amount)
        if not can_consume:
            return False, message
        
        # Update current values
        if resource == "tokens":
            self.current_tokens += amount
        elif resource == "input_tokens":
            self.current_input_tokens += amount
        elif resource == "output_tokens":
            self.current_output_tokens += amount
        elif resource == "api_calls":
            self.current_api_calls += amount
        elif resource == "subagents":
            self.current_subagents += amount
        elif resource == "iterations":
            self.current_iterations += amount
        elif resource == "search_calls":
            self.current_search_calls += amount
        
        return True, "OK"
    
    def get_usage_percentage(self) -> Dict[str, float]:
        """Get usage percentage"""
        return {
            "tokens": (self.current_tokens / self.max_tokens) * 100,
            "input_tokens": (self.current_input_tokens / self.max_input_tokens) * 100,
            "output_tokens": (self.current_output_tokens / self.max_output_tokens) * 100,
            "api_calls": (self.current_api_calls / self.max_api_calls) * 100,
            "subagents": (self.current_subagents / self.max_subagents) * 100,
            "iterations": (self.current_iterations / self.max_iterations) * 100,
            "search_calls": (self.current_search_calls / self.max_search_calls) * 100,
        }
    
    def should_stop(self) -> tuple[bool, str]:
        """Determine if should stop"""
        usage = self.get_usage_percentage()
        
        # Check if any resource exceeds 100%
        for resource, percentage in usage.items():
            if percentage >= 100:
                return True, f"{resource} exceeded 100% limit"
        
        # Check if any resource exceeds 95% (critical warning)
        for resource, percentage in usage.items():
            if percentage >= 95:
                return True, f"{resource} exceeded 95% critical threshold"
        
        return False, "OK"


@dataclass
class DailyBudget:
    """Daily budget"""
    date: date
    max_tokens: int = 5_000_000
    max_queries: int = 100
    max_cost: float = 50.0  # USD
    
    current_tokens: int = 0
    current_queries: int = 0
    current_cost: float = 0.0
    
    def can_accept_query(self, estimated_tokens: int, estimated_cost: float) -> tuple[bool, str]:
        """Check if can accept new query"""
        if self.current_queries >= self.max_queries:
            return False, "Daily query limit reached"
        
        if self.current_tokens + estimated_tokens > self.max_tokens:
            return False, "Daily token limit would be exceeded"
        
        if self.current_cost + estimated_cost > self.max_cost:
            return False, "Daily cost limit would be exceeded"
        
        return True, "OK"
    
    def record_query(self, tokens: int, cost: float):
        """Record query consumption"""
        self.current_tokens += tokens
        self.current_queries += 1
        self.current_cost += cost
    
    def get_usage_percentage(self) -> Dict[str, float]:
        """Get usage percentage"""
        return {
            "tokens": (self.current_tokens / self.max_tokens) * 100,
            "queries": (self.current_queries / self.max_queries) * 100,
            "cost": (self.current_cost / self.max_cost) * 100,
        }


class CostController:
    """Cost controller"""
    
    def __init__(self, budget_file: str = "budget.json"):
        self.budget_file = budget_file
        self.daily_budgets: Dict[date, DailyBudget] = {}
        self.load_budgets()
    
    def load_budgets(self):
        """Load budget data"""
        if os.path.exists(self.budget_file):
            with open(self.budget_file, 'r') as f:
                data = json.load(f)
                for date_str, budget_data in data.items():
                    budget_date = datetime.fromisoformat(date_str).date()
                    self.daily_budgets[budget_date] = DailyBudget(
                        date=budget_date,
                        **budget_data
                    )
    
    def save_budgets(self):
        """Save budget data"""
        data = {
            str(budget.date): {
                "current_tokens": budget.current_tokens,
                "current_queries": budget.current_queries,
                "current_cost": budget.current_cost,
            }
            for budget in self.daily_budgets.values()
        }
        with open(self.budget_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_today_budget(self) -> DailyBudget:
        """Get today's budget"""
        today = date.today()
        if today not in self.daily_budgets:
            self.daily_budgets[today] = DailyBudget(date=today)
        return self.daily_budgets[today]
    
    def check_daily_limit(self, estimated_tokens: int, estimated_cost: float) -> tuple[bool, str]:
        """Check daily limits"""
        today_budget = self.get_today_budget()
        return today_budget.can_accept_query(estimated_tokens, estimated_cost)
    
    def record_daily_usage(self, tokens: int, cost: float):
        """Record daily usage"""
        today_budget = self.get_today_budget()
        today_budget.record_query(tokens, cost)
        self.save_budgets()
    
    def get_daily_status(self) -> Dict:
        """Get daily status"""
        today_budget = self.get_today_budget()
        usage = today_budget.get_usage_percentage()
        
        status = {
            "date": str(today_budget.date),
            "usage": usage,
            "remaining": {
                "tokens": today_budget.max_tokens - today_budget.current_tokens,
                "queries": today_budget.max_queries - today_budget.current_queries,
                "cost": today_budget.max_cost - today_budget.current_cost,
            },
            "warnings": [],
        }
        
        # Check warnings
        for resource, percentage in usage.items():
            if percentage >= 95:
                status["warnings"].append(f"CRITICAL: {resource} at {percentage:.1f}%")
            elif percentage >= 80:
                status["warnings"].append(f"WARNING: {resource} at {percentage:.1f}%")
        
        return status
```

## 3. Cost Optimization Strategies

### 3.1 Intelligent Resource Allocation

**Dynamically adjust based on query complexity:**

| Query Type | Subagents | Token Budget | Iterations |
|------------|-----------|--------------|------------|
| **Simple** | 1 | 10k-20k | 1 |
| **Medium** | 2-3 | 30k-50k | 1-2 |
| **Complex** | 4-6 | 60k-80k | 2-3 |
| **Very Complex** | 6-8 | 80k-100k | 3 |

**Rules:**
- Simple queries cannot use multiple subagents
- Only complex queries allow high token consumption
- Automatically detect query complexity and allocate appropriate budget

### 3.2 Token Usage Optimization

**Optimization measures:**
1. **Result Compression**: Subagents return compressed summaries, not full content
2. **Context Management**: Regularly clean unnecessary historical information
3. **Batch Processing**: Merge similar search requests
4. **Caching Mechanism**: Cache results for common queries

### 3.3 Early Stopping Mechanism

**Stop conditions (stop if any condition is met):**
- Token usage > 95% → Immediately stop current iteration
- Information coverage > 80% → Stop research, start synthesis
- 3 consecutive searches with no new information → Stop searching
- Reached maximum iterations → Force stop

## 4. Cost Reporting

### 4.1 Real-Time Cost Tracking

```python
def track_query_cost(query_budget: QueryBudget, component: str, 
                     input_tokens: int, output_tokens: int, 
                     cost_per_1k_input: float, cost_per_1k_output: float):
    """Track per-query cost"""
    input_cost = (input_tokens / 1000) * cost_per_1k_input
    output_cost = (output_tokens / 1000) * cost_per_1k_output
    total_cost = input_cost + output_cost
    
    # Check and consume resources
    can_consume, message = query_budget.consume("input_tokens", input_tokens)
    if not can_consume:
        raise CostLimitExceeded(f"Input tokens: {message}")
    
    can_consume, message = query_budget.consume("output_tokens", output_tokens)
    if not can_consume:
        raise CostLimitExceeded(f"Output tokens: {message}")
    
    can_consume, message = query_budget.consume("tokens", input_tokens + output_tokens)
    if not can_consume:
        raise CostLimitExceeded(f"Total tokens: {message}")
    
    return total_cost
```

### 4.2 Daily Cost Report

**Report contents:**
- Total token consumption
- Total cost
- Query count
- Average cost per query
- Budget usage percentage
- Alert information

### 4.3 Cost Anomaly Detection

**Detection rules:**
- Per-query cost > 3× average cost → Mark as anomaly
- 5 consecutive queries exceed budget → Trigger alert
- Daily cost growth rate > 50% → Trigger alert

## 5. Implementation Checklist

### 5.1 Required Features

- [ ] `QueryBudget` class - Per-query budget management
- [ ] `DailyBudget` class - Daily budget management
- [ ] `CostController` class - Cost controller
- [ ] Real-time token tracking
- [ ] Auto-stop mechanism
- [ ] Budget alert system
- [ ] Cost report generation
- [ ] Budget persistence storage

### 5.2 Integration Points

- [ ] Check budget before each LLM call
- [ ] Check budget before creating each subagent
- [ ] Check budget before each iteration
- [ ] Check budget before each search
- [ ] Check daily budget before query starts

### 5.3 Testing Requirements

- [ ] Test stop mechanism when budget exceeded
- [ ] Test daily budget limits
- [ ] Test alert triggering
- [ ] Test cost tracking accuracy
- [ ] Test exception handling

## 6. Usage Examples

### 6.1 Basic Usage

```python
from research_agent.cost_control import QueryBudget, CostController

# Initialize
cost_controller = CostController()
query_budget = QueryBudget()

# Check if can accept query
estimated_tokens = 50_000
estimated_cost = 0.15
can_accept, message = cost_controller.check_daily_limit(estimated_tokens, estimated_cost)
if not can_accept:
    raise Exception(f"Cannot accept query: {message}")

# Track cost during query execution
try:
    # LLM call
    input_tokens = 5000
    output_tokens = 2000
    cost = track_query_cost(
        query_budget, "lead_researcher",
        input_tokens, output_tokens,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.002
    )
    
    # Check if should stop
    should_stop, reason = query_budget.should_stop()
    if should_stop:
        print(f"Stopping query: {reason}")
        return current_results
    
except CostLimitExceeded as e:
    print(f"Cost limit exceeded: {e}")
    return partial_results

# Record daily usage
cost_controller.record_daily_usage(query_budget.current_tokens, total_cost)
```

### 6.2 Budget Status Check

```python
# Check daily budget status
status = cost_controller.get_daily_status()
print(f"Daily usage: {status['usage']}")
if status['warnings']:
    for warning in status['warnings']:
        print(f"⚠️ {warning}")
```

## 7. Cost Control Rules Summary

### 7.1 Hard Limits (Cannot Violate)

1. **Per-query**: Maximum 100k tokens, stop immediately if exceeded
2. **Daily budget**: Maximum 5M tokens or $50, reject new queries if exceeded
3. **Subagents**: Maximum 8, reject creation if exceeded
4. **Iterations**: Maximum 3, force stop if exceeded

### 7.2 Warning Thresholds (Log but Continue)

1. **Per-query**: > 80k tokens → Warning, limit subsequent resources
2. **Daily budget**: > 80% → Warning, prepare to degrade service

### 7.3 Automatic Optimization

1. **Intelligent allocation**: Allocate budget based on query complexity
2. **Early stopping**: Stop early when information is sufficient
3. **Result compression**: Reduce token transmission
4. **Cache utilization**: Avoid duplicate computation

---

**Important Reminders:**
- All cost control measures are **mandatory**
- **Must stop** when budget exceeded, cannot continue
- **Reject new queries** when daily budget exhausted
- All cost data **must be persisted**

**Targets:**
- Per-query cost: < $0.50 (hard limit)
- Daily total cost: < $50 (hard limit)
- Monthly total cost: < $1200 (hard limit)
