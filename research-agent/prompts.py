"""Centralized prompt definitions for the research agent."""

from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# Lead Researcher Prompts
# =============================================================================

LEAD_RESEARCHER_SYSTEM = "You are a research planner. Respond in JSON."

LEAD_RESEARCHER_INITIAL = ChatPromptTemplate.from_template(
    """Break this query into 2-3 parallel research tasks:

Query: "{query}"
"""
)

LEAD_RESEARCHER_REFINE = ChatPromptTemplate.from_template(
    """Refine research strategy:

Query: {query}

Findings:
{findings_summary}

Generate 1-2 tasks to fill gaps.
"""
)

LEAD_RESEARCHER_RETRY = ChatPromptTemplate.from_template(
    """{previous_prompt}

Previous attempt failed validation:
{error}

Please ensure you return a valid JSON object matching the schema.
"""
)


# =============================================================================
# Subagent Prompts
# =============================================================================

SUBAGENT_SYSTEM = "Research assistant. Summarize clearly. You must respond in JSON."

SUBAGENT_ANALYSIS = ChatPromptTemplate.from_template(
    """Task: {task}

Results:
{results}

Summarize key findings in 2-3 sentences.
"""
)

SUBAGENT_RETRY = ChatPromptTemplate.from_template(
    """{previous_prompt}

Previous attempt failed: {error}
Ensure you return a valid JSON object.
"""
)


# =============================================================================
# Synthesizer Prompts
# =============================================================================

SYNTHESIZER_SYSTEM = "Synthesis expert. Combine insights effectively. You must respond in JSON."

SYNTHESIZER_MAIN = ChatPromptTemplate.from_template(
    """Query: {query}

Findings:
{findings}

Synthesize into comprehensive answer.
"""
)

SYNTHESIZER_RETRY = ChatPromptTemplate.from_template(
    """{previous_prompt}

Previous attempt failed validation:
{error}

Please ensure you return a valid JSON object matching the schema.
"""
)
