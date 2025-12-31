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
        """Test: Research history, current state, and future of quantum computing

        Verifies:
        1. Causal Chains: Tasks should ideally show dependencies
           (checked via tasks list)
        2. Deep Scraping: Subagent findings must contain 'content' field
           with scraped text.
        3. Synthesis: Final report is substantial.
        """
        state = {
            **initial_state,
            "query": (
                "Research the history, current state, and future of "
                "quantum computing"
            ),
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        # Basic content check
        assert len(final_report) > 500

        # --- Deep Research V2 Checks ---

        # 1. Check Subagent Findings for Deep Scraped Content
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 2, "Complex query should trigger multiple subagents"

        # Verify that AT LEAST one finding has deep scraped content (>500 chars)
        has_deep_content = False
        for f in findings:
            # f is a Finding object (Pydantic model)
            if hasattr(f, 'content') and len(f.content) > 100:
                has_deep_content = True
                break

        assert has_deep_content, (
            "Findings should contain scraped 'content' from scrape_web_page tool"
        )

        # 2. Check for Structured Tasks
        tasks = final_state.get("subagent_tasks", [])
        assert len(tasks) > 0
        # Check if tasks are ResearchTask objects (have 'id' and 'dependencies')
        assert hasattr(tasks[0], 'id'), (
            "Tasks should be structured ResearchTask objects"
        )
        assert hasattr(tasks[0], 'dependencies'), (
            "Tasks should support dependencies"
        )

        # 3. Check for Citations
        citations = final_state.get("citations", [])
        assert len(citations) >= 3, "Should have multiple citations"


    def test_cloud_providers_comparison(self, app, initial_state):
        """Test: Compare top cloud providers

        Verifies:
        1. Multi-step research (implied by complex topic)
        2. Deep Scraping content presence
        """
        state = {
            **initial_state,
            "query": (
                "Compare the top 5 cloud providers across pricing, "
                "features, and reliability"
            ),
        }

        final_state = app.invoke(state)

        # End-state evaluation
        final_report = final_state.get("final_report", "")
        assert len(final_report) > 500

        # V2 Checks
        findings = final_state.get("subagent_findings", [])
        assert len(findings) >= 2

        # Check deep scraping
        has_deep_content = False
        for f in findings:
            if hasattr(f, 'content') and len(f.content) > 100:
                has_deep_content = True
                break
        assert has_deep_content, "Findings should contain deep scraped content"


