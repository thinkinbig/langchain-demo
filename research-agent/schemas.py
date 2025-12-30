"""State schemas for research agent with Pydantic validation"""

import operator
from typing import Annotated, Any, List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DictCompatibleModel(BaseModel):
    """BaseModel adapter that supports both attribute and dict-style access"""

    def __getitem__(self, key: str) -> Any:
        """Support dict-style access: state['key']"""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict-style get: state.get('key', default)"""
        return getattr(self, key, default)


class Finding(DictCompatibleModel):
    """Individual finding from a subagent"""
    task: str = Field(..., description="The research task")
    summary: str = Field(..., min_length=1, description="Summary of findings")
    sources: List[dict] = Field(
        default_factory=list, description="Source URLs and titles"
    )


class SubagentOutput(DictCompatibleModel):
    """Output from subagent LLM validaton"""
    summary: str = Field(..., min_length=1, description="Summary of findings from search results")



class ResearchTasks(DictCompatibleModel):
    """List of research tasks"""
    tasks: List[str] = Field(..., min_length=1, description="List of research tasks")


class SynthesisResult(DictCompatibleModel):
    """Result of synthesis step"""
    summary: str = Field(..., description="Comprehensive summary of findings")


class Citation(DictCompatibleModel):
    """Citation entry"""
    title: str
    url: str = Field(..., description="Source URL")


class ResearchState(DictCompatibleModel):
    """State for multi-agent research system with Pydantic validation"""

    query: str = Field(..., min_length=1, description="Original user query")
    research_plan: str = Field(default="", description="Research plan")
    subagent_tasks: List[str] = Field(
        default_factory=list, description="Tasks for subagents"
    )

    # For LangGraph reducers, we use Annotated with operator.add
    # Pydantic will validate each item in the list
    subagent_findings: Annotated[
        List[Finding],
        operator.add
    ] = Field(default_factory=list, description="Findings from subagents")

    iteration_count: int = Field(
        default=0, ge=0, le=10, description="Iteration count"
    )
    needs_more_research: bool = Field(
        default=False, description="Whether more research needed"
    )
    synthesized_results: str = Field(
        default="", description="Synthesized results"
    )
    citations: List[Citation] = Field(
        default_factory=list, description="Citations"
    )
    final_report: str = Field(default="", description="Final report")
    
    # Retry state
    error: str | None = Field(default=None, description="Error message from validation")
    retry_count: int = Field(default=0, description="Retry attempt count")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    model_config = ConfigDict(
        # Allow extra fields (for LangGraph compatibility)
        extra="allow"
    )


# Auxiliary Pydantic models for type safety
class SubagentState(DictCompatibleModel):
    """Minimal state for subagent workers"""
    subagent_tasks: List[str] = Field(..., min_length=1)

    model_config = ConfigDict(extra="allow")


class LeadResearcherState(DictCompatibleModel):
    """State needed by lead researcher"""
    query: str = Field(..., min_length=1)
    iteration_count: int = Field(default=0, ge=0)
    subagent_findings: List[Finding] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class SynthesizerState(DictCompatibleModel):
    """State needed by synthesizer"""
    query: str = Field(..., min_length=1)
    subagent_findings: List[Finding] = Field(..., min_length=1)

    model_config = ConfigDict(extra="allow")


class DecisionState(DictCompatibleModel):
    """State needed by decision node"""
    iteration_count: int = Field(..., ge=0)
    subagent_findings: List[Finding] = Field(default_factory=list)
    synthesized_results: str = Field(default="")

    model_config = ConfigDict(extra="allow")


class CitationAgentState(DictCompatibleModel):
    """State needed by citation agent"""
    query: str = Field(..., min_length=1)
    subagent_findings: List[Finding] = Field(..., min_length=1)
    synthesized_results: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="allow")

