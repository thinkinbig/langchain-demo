import pytest
from unittest.mock import MagicMock, patch
from cost_control import QueryBudget, track_query_cost, CostLimitExceeded
from config import settings

class TestCostLimits:
    def test_default_limits_increased(self):
        """Verify that default limits are increased as expected"""
        budget = QueryBudget()
        assert budget.max_tokens == 200_000
        assert budget.max_input_tokens == 160_000
        assert budget.max_output_tokens == 80_000

    def test_config_limit_sync(self):
        """Verify config limits match budget limits"""
        assert settings.MAX_TOKENS_PER_QUERY == 200_000

    def test_high_token_usage_simulation(self):
        """Verify that usage between 80k and 160k is now allowed"""
        budget = QueryBudget()
        
        # Simulate 100k input tokens (previously would fail > 80k)
        # Using specific cost values for testing
        cost_input = 0.001
        cost_output = 0.002
        
        track_query_cost(
            budget,
            "test_large_input",
            input_tokens=100_000,
            output_tokens=100,
            cost_per_1k_input=cost_input,
            cost_per_1k_output=cost_output
        )
        
        assert budget.current_input_tokens == 100_000
        assert budget.current_tokens == 100_100
        
        # Now pushing it over the new limit should still fail
        with pytest.raises(CostLimitExceeded) as exc:
            track_query_cost(
                budget,
                "test_overflow",
                input_tokens=61_000, # 100k + 61k = 161k > 160k
                output_tokens=0,
                cost_per_1k_input=cost_input,
                cost_per_1k_output=cost_output
            )
        assert "input_tokens limit exceeded" in str(exc.value)

if __name__ == "__main__":
    pytest.main([__file__])
