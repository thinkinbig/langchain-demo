"""Integration tests for complex queries

Based on Anthropic's principle: "Complex research might use more than 10 subagents"

NOTE: These tests use REAL API calls. Make sure you have valid API keys in .env file.
"""

import pytest
from dotenv import load_dotenv

# Load environment variables for integration tests
load_dotenv()



@pytest.mark.integration
@pytest.mark.slow
class TestComplexQueries:
    """Test complex breadth-first queries"""

    def test_quantum_computing_research(self, app, initial_state):
        """Test: Research history, current state, and future of quantum computing"""
        state = {
            **initial_state,
            "query": "Research the history, current state, and future of quantum computing",
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        assert len(final_report) > 0

        # Complex query should have multiple findings
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 3, "Complex query should use multiple subagents"

        # Should have comprehensive synthesis
        synthesis = final_state.get("synthesized_results", "")
        assert len(synthesis) > 500, "Should have substantial synthesis"

        # Should have multiple citations
        citations = final_state.get("citations", [])
        assert len(citations) >= 3, "Should have multiple citations"

        # May iterate multiple times
        assert final_state.get("iteration_count", 0) <= 3

    def test_cloud_providers_comparison(self, app, initial_state):
        """Test: Compare top cloud providers"""
        state = {
            **initial_state,
            "query": "Compare the top 5 cloud providers across pricing, features, and reliability",
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        assert len(final_report) > 0

        # Should have multiple findings
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 3

        # Should have comprehensive synthesis
        synthesis = final_state.get("synthesized_results", "")
        assert len(synthesis) > 500

