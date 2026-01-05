import json
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class QueryBudget:
    """Per-query budget"""
    max_tokens: int = 120_000
    max_input_tokens: int = 120_000
    max_output_tokens: int = 80_000
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
    current_cost: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock,
    init=False, repr=False)

    def check_limit(self, resource: str, amount: int) -> tuple[bool, str]:
        """Check if limit would be exceeded"""
        with self._lock:
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
        with self._lock:
            # We must re-check inside the lock to avoid race conditions
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
            if current_val + amount > max_val:
                return False, (
                    f"{resource} limit exceeded: {current_val + amount} > {max_val}"
                )

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
        with self._lock:
            return {
                "tokens": (self.current_tokens / self.max_tokens) * 100,
                "input_tokens": (
                    self.current_input_tokens / self.max_input_tokens
                ) * 100,
                "output_tokens": (
                    self.current_output_tokens / self.max_output_tokens
                ) * 100,
                "api_calls": (self.current_api_calls / self.max_api_calls) * 100,
                "subagents": (self.current_subagents / self.max_subagents) * 100,
                "iterations": (self.current_iterations / self.max_iterations) * 100,
                "search_calls": (
                    self.current_search_calls / self.max_search_calls
                ) * 100,
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

    def can_accept_query(
        self, estimated_tokens: int, estimated_cost: float
    ) -> tuple[bool, str]:
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
                    # Filter out keys that might not exist in DailyBudget fields
                    # if structure changed
                    valid_keys = DailyBudget.__annotations__.keys()
                    filtered_data = {
                        k: v for k, v in budget_data.items() if k in valid_keys
                    }
                    self.daily_budgets[budget_date] = DailyBudget(
                        date=budget_date,
                        **filtered_data
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

    def check_daily_limit(
        self, estimated_tokens: int, estimated_cost: float
    ) -> tuple[bool, str]:
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

class CostLimitExceeded(Exception):
    """Exception raised when cost limit is exceeded"""
    pass

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

    # Accumulate cost (Thread-safe assignment would be better but this is MVP)
    # We use the lock in consume, ideally we'd have a method for this.
    # For now, we accept slight race condition risk on reporting.
    query_budget.current_cost += total_cost

    return total_cost





class CostTrackingCallback(BaseCallbackHandler):
    """Callback that tracks detailed cost information per query"""

    def __init__(self, budget: QueryBudget):
        self.budget = budget
        # Pricing Map (per 1k tokens)
        # Based on rough public pricing (adjust as needed)
        self.DATA_COST_MAP = {
            "qwen-plus": {"input": 0.0004, "output": 0.0012},  # $0.4 / $1.2 per 1M
            "qwen-turbo": {"input": 0.0002, "output": 0.0006}, # $0.2 / $0.6 per 1M
        }

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Track usage when LLM finishes"""
        try:
            # Attempt to extract token usage (this varies by provider)
            # For OpenAI/LangChain standard:
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                model_name = response.llm_output.get("model_name", "default")

                # Normalize model name slightly
                if "qwen-plus" in model_name:
                    cost_config = self.DATA_COST_MAP["qwen-plus"]
                elif "qwen-turbo" in model_name:
                    cost_config = self.DATA_COST_MAP["qwen-turbo"]
                else:
                    # Fallback to plus model pricing if model not found
                    cost_config = self.DATA_COST_MAP.get(
                        "qwen-plus", {"input": 0.0004, "output": 0.0012}
                    )

                if token_usage:
                    input_tokens = token_usage.get("prompt_tokens", 0)
                    output_tokens = token_usage.get("completion_tokens", 0)

                    track_query_cost(
                        self.budget,
                        "llm_call",
                        input_tokens,
                        output_tokens,
                        cost_config["input"],
                        cost_config["output"]
                    )
        except CostLimitExceeded as e:
            # Raise to stop execution
            print(f"⛔ COST LIMIT EXCEEDED: {e}")
            raise e
        except Exception as e:
            print(f"Error tracking cost: {e}")

