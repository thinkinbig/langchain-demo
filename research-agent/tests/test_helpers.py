"""Helper functions for more robust test assertions"""


def assert_has_required_fields(state: dict, required_fields: list):
    """Assert that state has all required fields"""
    for field in required_fields:
        assert field in state, f"State should have field: {field}"


def assert_content_relevance(content: str, query: str, min_keyword_matches: int = 1):
    """Assert that content is relevant to the query by checking for keyword matches"""
    query_lower = query.lower()
    content_lower = content.lower()

    # Extract key terms from query (simple approach: split and filter)
    key_terms = [term for term in query_lower.split() if len(term) > 3]

    # Count how many key terms appear in content
    matches = sum(1 for term in key_terms if term in content_lower)

    assert matches >= min_keyword_matches, (
        f"Content should be relevant to query. "
        f"Found {matches} keyword matches, expected at least {min_keyword_matches}"
    )


def assert_complete_workflow(state: dict):
    """Assert that the workflow completed successfully"""
    # Check that we have a final report
    assert "final_report" in state, "Should have final_report"
    assert len(state.get("final_report", "")) > 0, "final_report should not be empty"

    # Check that we have citations
    citations = state.get("citations", [])
    assert len(citations) > 0, "Should have at least one citation"

    # Check that citations have required structure
    for citation in citations:
        assert "title" in citation or "url" in citation, (
            "Citations should have at least title or url"
        )


def assert_findings_structure(findings: list):
    """Assert that findings have proper structure"""
    if not findings:
        return  # Empty findings might be valid in some cases

    for finding in findings:
        # Findings should have some structure
        assert isinstance(finding, dict), "Findings should be dictionaries"
        # At minimum, should have task or summary
        assert "task" in finding or "summary" in finding, (
            "Findings should have task or summary"
        )


def assert_synthesis_quality(synthesis: str, query: str):
    """Assert that synthesis is meaningful and relevant"""
    assert len(synthesis) > 0, "Synthesis should not be empty"

    # Check relevance
    assert_content_relevance(synthesis, query, min_keyword_matches=1)

    # Synthesis should be more than just a few words
    words = synthesis.split()
    assert len(words) > 5, "Synthesis should have meaningful content (more than 5 words)"


def assert_iteration_reasonable(iteration_count: int, max_iterations: int = 10):
    """Assert that iteration count is within reasonable bounds"""
    assert iteration_count >= 0, "Iteration count should be non-negative"
    assert iteration_count <= max_iterations, (
        f"Iteration count {iteration_count} exceeds maximum {max_iterations}"
    )


def assert_tasks_related_to_query(tasks: list, query: str):
    """Assert that generated tasks are related to the query"""
    if not tasks:
        return  # No tasks might be valid in some cases

    query_lower = query.lower()
    query_terms = [term for term in query_lower.split() if len(term) > 3]

    # At least some tasks should mention query terms
    relevant_tasks = 0
    for task in tasks:
        task_lower = task.lower()
        if any(term in task_lower for term in query_terms):
            relevant_tasks += 1

    # At least 50% of tasks should be relevant
    relevance_ratio = relevant_tasks / len(tasks) if tasks else 0
    assert relevance_ratio >= 0.5, (
        f"At least 50% of tasks should be relevant to query. "
        f"Found {relevance_ratio:.1%} relevance"
    )

