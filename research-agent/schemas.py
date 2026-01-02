"""State schemas for research agent with Pydantic validation"""

import operator
from datetime import datetime

# Forward reference for RetrievalResult (defined in retrieval.py)
from typing import TYPE_CHECKING, Annotated, Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    pass


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
    content: str = Field(
        default="",
        description=(
            "Content reference or metadata. "
            "Can be: (1) ContentReference dict with content_hash and metadata, "
            "(2) Legacy string format with full/truncated content, "
            "(3) Empty string if no content stored"
        )
    )
    sources: List[dict] = Field(
        default_factory=list, description="Source URLs and titles"
    )

    def get_content_metadata(self) -> dict:
        """
        Extract content metadata for token-efficient storage.
        Returns dict with content_hash, content_length, and summary.
        """
        content = self.content

        # If it's a ContentReference dict
        if isinstance(content, dict) and "content_hash" in content:
            return content

        # If it's a legacy string
        if isinstance(content, str):
            if len(content) <= 16 and content.isalnum():
                # Likely a hash reference
                return {
                    "content_hash": content,
                    "content_length": 0,
                    "content_type": "reference"
                }
            else:
                # Full content - return metadata
                from memory_helpers import compute_content_hash
                return {
                    "content_hash": compute_content_hash(content),
                    "content_length": len(content),
                    "content_type": "legacy_full",
                    "content_preview": content[:200]  # First 200 chars for context
                }

        return {
            "content_hash": "",
            "content_length": 0,
            "content_type": "empty"
        }


class VerificationResult(DictCompatibleModel):
    """Result of the verification process"""
    is_valid: bool = Field(..., description="Whether the report is factual")
    corrected_report: str = Field(
        ..., description="The verified and corrected report"
    )
    corrections: List[str] = Field(
        default_factory=list, description="List of corrections made"
    )


class SubagentOutput(DictCompatibleModel):
    """Output from subagent LLM validaton"""
    summary: str = Field(
        ..., min_length=1, description="Summary of findings from search results"
    )




class ResearchTask(DictCompatibleModel):
    """Detailed research task"""
    id: str = Field(..., description="Unique task ID (e.g., T1)")
    description: str = Field(..., description="Detailed task description")
    rationale: str = Field(default="", description="Why this task is necessary")
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete first (optional)"
    )


class ResearchTasks(DictCompatibleModel):
    """List of research tasks"""
    tasks: List[ResearchTask] = Field(
        ..., min_length=1, description="List of research tasks"
    )
    scratchpad: str = Field(
        default="", description="Updated scratchpad/notes for the agent"
    )


class SynthesisResult(DictCompatibleModel):
    """Result of synthesis step"""
    summary: str = Field(..., description="Comprehensive summary of findings")


class Citation(DictCompatibleModel):
    """Citation entry"""
    title: str
    url: str = Field(..., description="Source URL")


class VisitedSource(DictCompatibleModel):
    """Unified representation of a visited source"""
    identifier: str = Field(..., description="URL or document name")
    source_type: str = Field(..., description="Type: 'internal' or 'web'")
    visited_at: Optional[datetime] = Field(default=None, description="When source was visited")

    def __hash__(self):
        """Make hashable for set operations"""
        return hash((self.identifier, self.source_type))

    def __eq__(self, other):
        """Equality based on identifier and source_type"""
        if not isinstance(other, VisitedSource):
            return False
        return self.identifier == other.identifier and self.source_type == other.source_type


class ResearchState(DictCompatibleModel):
    """State for multi-agent research system with Pydantic validation"""

    query: str = Field(..., min_length=1, description="Original user query")
    research_plan: str = Field(default="", description="Research plan")
    scratchpad: str = Field(
        default="", description="Internal working memory/notes for the agent"
    )

    subagent_tasks: List[ResearchTask] = Field(
        default_factory=list, description="Tasks for subagents"
    )

    # For LangGraph reducers, we use Annotated with operator.add
    # Pydantic will validate each item in the list
    subagent_findings: Annotated[
        List[Finding],
        operator.add
    ] = Field(default_factory=list, description="Findings from subagents")

    # Filtered findings (overwrite, not append)
    filtered_findings: List[Finding] = Field(default_factory=list, description="Quality-filtered findings")

    # Unified visited sources tracking
    visited_sources: Annotated[
        List[VisitedSource],
        operator.add
    ] = Field(default_factory=list, description="Unified list of visited sources")

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

    @staticmethod
    def get_visited_identifiers(state: dict, source_type: Optional[str] = None) -> List[str]:
        """
        Get visited identifiers from state.
        
        Args:
            state: State dictionary
            source_type: Optional filter ('internal' or 'web')
            
        Returns:
            List of visited identifiers
        """
        identifiers = []

        visited_sources = state.get("visited_sources", [])
        for vs in visited_sources:
            if isinstance(vs, VisitedSource):
                if source_type is None or vs.source_type == source_type:
                    identifiers.append(vs.identifier)
            elif isinstance(vs, dict):
                vs_type = vs.get("source_type", "")
                if source_type is None or vs_type == source_type:
                    identifiers.append(vs.get("identifier", ""))

        return identifiers


# Auxiliary Pydantic models for type safety
class SubagentState(DictCompatibleModel):
    """Minimal state for subagent workers"""
    subagent_tasks: List[ResearchTask] = Field(..., min_length=1)

    # Context channels for subgraph
    task_description: str = Field(default="", description="Cached task description")
    internal_result: Optional[Any] = Field(default=None, description="Internal knowledge retrieval result")
    web_result: Optional[Any] = Field(default=None, description="Web search retrieval result")
    subagent_findings: List[Finding] = Field(default_factory=list, description="Findings output")
    visited_sources: List[VisitedSource] = Field(default_factory=list, description="Already visited sources")

    model_config = ConfigDict(extra="allow")


class LeadResearcherState(DictCompatibleModel):
    """State needed by lead researcher"""
    query: str = Field(..., min_length=1)
    iteration_count: int = Field(default=0, ge=0)
    scratchpad: str = Field(default="")
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

