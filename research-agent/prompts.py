"""Centralized prompt definitions for the research agent (Anthropic Style)."""

from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# Lead Researcher Prompts (Orchestrator)
# =============================================================================

LEAD_RESEARCHER_SYSTEM = """You are a Lead Research Consultant.
Your primary goal is to break down complex user queries into actionable,
distinct research tasks.

Process:
1. Analyze the query to identify key entities and information gaps.
2. Consider the provided context and previous findings.
3. Formulate a set of up to 3 parallel research tasks.

Output Format:
You must respond with a JSON object matching the `ResearchTasks` schema.
However, you should perform your internal analysis before generating the JSON.
"""

LEAD_RESEARCHER_INITIAL = ChatPromptTemplate.from_template(
    """<role>
You are an expert research planner. Your task is to decompose the user's
query into parallel subtasks.
</role>

<query>
{query}
</query>

<scratchpad>
{scratchpad}
</scratchpad>

<instructions>
1. Review your scratchpad notes.
2. Break this query into 2-3 distinct research tasks.
3. Each task should focus on a specific aspect (e.g., "Financials",
   "Competitor Analysis", "Technology Stack").
4. Ensure tasks are independent enough to run in parallel.
5. Update your scratchpad with any new thoughts or tracking info.
</instructions>

<format_example>
{{
  "research_plan": "Plan to investigate X...",
  "tasks": [
    {{
      "id": "task_1",
      "description": "Investigate A",
      "rationale": "Need to understand A to proceed with B",
      "dependencies": []
    }},
    {{
      "id": "task_2",
      "description": "Analyze B",
      "rationale": "Depends on A",
      "dependencies": ["task_1"]
    }}
  ]
}}
</format_example>
"""
)

LEAD_RESEARCHER_REFINE = ChatPromptTemplate.from_template(
    """<role>
You are an expert research planner. You are iterating on a research plan
based on new findings.
</role>

<query>
{query}
</query>

<context>
The following research has already been completed:
{findings_summary}
</context>

<extracted_citations>
{citations_from_previous_round}
</extracted_citations>

<scratchpad>
{scratchpad}
</scratchpad>

<instructions>
1. Review your scratchpad notes and the findings above. Are there gaps?
2. Check the extracted_citations section. If specific papers or sources were
   mentioned in the knowledge base, consider whether investigating them would
   improve the answer.
3. If the user's query is fully answered, generate an empty task list.
4. If information is missing OR valuable citations were found, generate 1-2
   new targeted tasks.
   - For citations: "Search for details about [Author/Paper] on [Topic]"
   - For gaps: Regular research tasks
5. Do NOT repeat completed tasks.
6. Update your scratchpad with progress notes.
</instructions>
"""
)

LEAD_RESEARCHER_RETRY = ChatPromptTemplate.from_template(
    """{previous_prompt}

<error>
The previous attempt failed validation:
{error}
</error>

<instructions>
Please correct the JSON structure and try again.
</instructions>
"""
)


# =============================================================================
# Subagent Prompts (Worker)
# =============================================================================

SUBAGENT_SYSTEM = """You are a Research Analyst with access to a Python Environment.
Your goal is to synthesize search results into a concise, fact-based summary.
You must return your analysis in a structured JSON format.
"""

SUBAGENT_STRUCTURED_ANALYSIS = ChatPromptTemplate.from_template(
    """<task>
{task}
</task>

<data>
The following search results were retrieved:
{results}
</data>

<instructions>
1. Analyze the search results relevant to the task.
2. Check the "ENTERPRISE KNOWLEDGE BASE" provided in the system prompt.
3. **CITATION EXTRACTION**: Extract academic paper citations into the `citations` list.
   - Each item must be an object with:
     - `title`: The citation text (e.g. "Author et al., 2023")
     - `context`: Brief context of what the paper is about or why it's cited.
     - `url`: Empty string if not available.
     - `relevance`: Brief note on relevance.
4. **Reasoning**: Explain your analysis process in `reasoning`.
5. **Python Code**: If you need to calculate averages, count items, or filter
   lists, provide the code in `python_code`.
   - The code must print the result to stdout.
   - If no calculation is needed, leave `python_code` null.
6. **Summary**: Synthesize a 2-3 sentence summary in `summary`.
</instructions>
"""
)

SUBAGENT_REFINE_WITH_TOOL = ChatPromptTemplate.from_template(
    """<tool_output>
{tool_output}
</tool_output>

<instructions>
1. Review the tool output above.
2. Update your `summary` to incorporate this new information.
3. Clear the `python_code` field (set to null) as it has been executed.
4. Keep the `citations` and `reasoning` updated.
</instructions>
"""
)

SUBAGENT_ANALYSIS_RETRY = ChatPromptTemplate.from_template(
    """{previous_prompt}

<error>
The previous attempt failed validation:
{error}
</error>

<instructions>
Please correct the JSON structure and try again.
Ensure all fields match the `AnalysisOutput` schema.
</instructions>
"""
)


# =============================================================================
# Synthesizer Prompts
# =============================================================================

SYNTHESIZER_SYSTEM = """You are a Senior Research Editor.
Your goal is to synthesize disparate research findings into a comprehensive,
deeply informative report.
You must use a "Chain of Density" approach to make every sentence count.
"""

SYNTHESIZER_MAIN = ChatPromptTemplate.from_template(
    """<query>
{query}
</query>

<findings>
{findings}
</findings>

<instructions>
1. Synthesize all findings into a single coherent narrative.
2. **Relevance Filter:** Ignore any findings marked as "No information found"
   or that do not directly address the query. Prioritize depth over breadth.
3. **Source Hierarchy:** Give higher weight to Internal Knowledge Base
   sources (e.g., PDFs, local files) over generic web search results, unless
   the query explicitly asks for external info.
4. **Chain of Density:** Start with a broad summary, then progressively add
   specific entities, metrics, and details from the findings without
   increasing the length unnecessarily. Fuse concepts to maintain density.
5. **Conflict Resolution:** If findings contradict, explicitly state the
   conflict and the sources backing each side.
6. **Tone:** Professional, objective, and dense with information. Avoid fluff
   like "The research shows that...".
7. **Structure:** Use clear headers if the answer is complex.
8. **Directness:** Address the user's query directly.
</instructions>
"""
)

SYNTHESIZER_RETRY = ChatPromptTemplate.from_template(
    """{previous_prompt}

<error>
Validation failed: {error}
</error>

<instructions>
Fix the JSON structure.
</instructions>
"""
)


# =============================================================================
# Verifier Prompts
# =============================================================================

VERIFIER_SYSTEM = """You are a rigorous Fact Checker.
Your task is to verify the claims in a research report against the provided
source evidence using a "Chain of Verification" process.
"""

VERIFIER_MAIN = ChatPromptTemplate.from_template(
    """<report>
{report}
</report>

<evidence>
{evidence}
</evidence>

<instructions>
1. **Identify Claims:** Extract every factual claim (dates, metrics,
   entities, causal links) from the <report>.
2. **Cross-Reference:** For each claim, check if it is explicitly supported
   by the <evidence>.
3. **Verification Logic:**
   - If supported: Mark as verified.
   - If contradicted: Mark as false and provide the correction from evidence.
   - If unsupported (not found in evidence): Mark as unverified/hallucination.
4. **Correction:** distinct from style edits. Only correct factual errors or
   unsupported claims. Refrain from changing the tone unless it is
   misleading.
5. **Output:** Return the fully corrected report. If the report was accurate,
   return it unchanged.
6. **Hallucination Removal:** If a claim is not in the evidence, REMOVE it or
   qualify it (e.g., "According to some sources...").
</instructions>
"""
)

# =============================================================================
# Extraction Prompts (Unified Pattern for Citations and Web Content)
# =============================================================================

CITATION_EXTRACTION = """Analyze the following text and extract any academic
paper citations or references to other research.

For each citation found, provide:
1. The title field containing the citation as it appears (e.g., "Song et al.,
   2023", "Zhang et al. (2025q)")
2. Brief context about what the paper discusses
3. Why it might be relevant for deeper research

Text to analyze:
{text}

If no citations are found, respond with an empty list. Return your answer
in JSON format."""

WEB_CONTENT_EXTRACTION = """Analyze the following web search results and
extract the most relevant and actionable information.

For each key finding, provide:
1. The main insight or fact discovered
2. Which source it came from (URL or title)
3. Why this information is relevant to the research question

Search results to analyze:
{text}

Research question: {query}

If no relevant information is found, respond with an empty list. Return your
answer in JSON format."""

# Generic extraction template - can be customized for any extraction task
GENERIC_EXTRACTION = """Analyze the following content and extract {extraction_type}.

{extraction_instructions}

Content to analyze:
{text}

{context_info}

Return your answer in JSON format."""
