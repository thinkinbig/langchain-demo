"""Integration tests for medium complexity queries

Based on Anthropic's principle: "Direct comparisons might need 2-4 "
"subagents with 10-15 calls each"

NOTE: These tests use REAL API calls. Make sure you have valid API keys in .env file.
"""

import pytest
from dotenv import load_dotenv

# Load environment variables for integration tests
load_dotenv()



@pytest.mark.integration
class TestMediumQueries:
    """Test medium complexity queries (comparisons, multi-aspect)"""

    def test_compare_python_rust(self, app, initial_state):
        """Test: Compare Python and Rust for web development"""
        state = {
            **initial_state,
            "query": "Compare Python and Rust for web development"
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        assert len(final_report) > 0

        # Should have multiple findings (multiple subagents)
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 2, "Medium query should use multiple subagents"

        # Should have comprehensive synthesis
        synthesis = final_state.get("synthesized_results", "")
        assert len(synthesis) > 200, "Should have substantial synthesis"

        # Should have citations from diverse sources
        citations = final_state.get("citations", [])
        assert len(citations) >= 2, "Should have multiple citations"

        # May iterate once
        assert final_state.get("iteration_count", 0) <= 2

    def test_pros_cons_microservices(self, app, initial_state):
        """Test: Research pros and cons of microservices"""
        state = {
            **initial_state,
            "query": "Research the pros and cons of microservices architecture",
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        assert len(final_report) > 0

        # Should have multiple findings
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 2

        # Should have citations
        assert len(final_state.get("citations", [])) > 0

    def test_rest_vs_graphql(self, app, initial_state):
        """Test: Analyze differences between REST and GraphQL"""
        state = {
            **initial_state,
            "query": "Analyze the differences between REST and GraphQL APIs"
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        assert len(final_report) > 0

        # Should have multiple findings
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 2

        # Should have comprehensive synthesis
        synthesis = final_state.get("synthesized_results", "")
        assert len(synthesis) > 200

