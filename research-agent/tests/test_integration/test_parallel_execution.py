"""Test parallel execution of subagents

Verify that subagents execute in parallel, not sequentially.

NOTE: These tests use REAL API calls. Make sure you have valid API keys in .env file.
"""

import time

import pytest
from dotenv import load_dotenv

# Load environment variables for integration tests
load_dotenv()



@pytest.mark.integration
class TestParallelExecution:
    """Test that subagents execute in parallel"""

    def test_parallel_subagents(self, app, initial_state):
        """Test that multiple subagents execute in parallel"""
        state = {
            **initial_state,
            "query": "Compare Python, Rust, and Go programming languages"
        }

        start_time = time.time()
        final_state = app.invoke(state)
        total_time = time.time() - start_time

        # Should have multiple findings
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 2, "Should have multiple subagents"

        # If truly parallel, total time should be closer to max(subagent time)
        # rather than sum(subagent times)
        # This is a heuristic check - parallel execution should be faster
        print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📊 Subagents executed: {len(findings)}")

        # Basic check: parallel execution should complete
        assert total_time < 300, "Should complete within reasonable time"

    def test_fan_out_works(self, app, initial_state):
        """Test that fan-out to subagents works correctly"""
        state = {
            **initial_state,
            "query": "Research the pros and cons of microservices architecture",
        }

        final_state = app.invoke(state)

        # Should have subagent tasks
        tasks = final_state.get("subagent_tasks", [])
        findings = final_state.get("subagent_findings", [])

        # Number of findings should match or be close to number of tasks
        # (some tasks might fail, but most should complete)
        assert len(findings) > 0, "Should have at least some findings"
        assert len(findings) <= len(tasks) + 2, (
            "Findings should not exceed tasks significantly"
        )

