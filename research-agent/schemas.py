"""State schemas for research agent"""

import operator
from typing import Annotated, Dict, List, TypedDict


class ResearchState(TypedDict):
    """State for multi-agent research system"""
    query: str  # Original user query
    research_plan: str  # Research plan created by LeadResearcher
    subagent_tasks: List[str]  # Tasks for subagents
    subagent_findings: Annotated[List[Dict], operator.add]  # Findings from each subagent
    iteration_count: int  # Number of research iterations
    needs_more_research: bool  # Whether more research is needed
    synthesized_results: str  # Synthesized results from all findings
    citations: List[Dict]  # Extracted citations
    final_report: str  # Final report with citations

