"""State schemas for research agent with Pydantic validation"""

import hashlib
import json
import operator
from datetime import datetime

# Forward reference for RetrievalResult (defined in retrieval.py)
from typing import TYPE_CHECKING, Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
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
            "Content metadata as JSON string. "
            "Contains content_hash, content_length, and content_preview. "
            "Empty string if no content stored."
        )
    )
    sources: List[dict] = Field(
        default_factory=list, description="Source URLs and titles"
    )
    extracted_citations: List[dict] = Field(
        default_factory=list, description="Citations extracted from retrieved content"
    )

    def get_content_metadata(self) -> dict:
        """
        Extract content metadata for token-efficient storage.
        Returns dict with content_hash, content_length, and preview.
        """
        return parse_content_metadata(self.content)


# =============================================================================
# Finding utility functions
# =============================================================================


def compute_content_hash(content: str) -> str:
    """Compute short hash for content reference"""
    if not content:
        return ""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_content_metadata(content: str, max_preview: int = 200) -> dict:
    """
    Create metadata dict for content instead of storing full content.
    This reduces token usage while preserving essential information.

    Args:
        content: Full content string
        max_preview: Maximum length of preview to include

    Returns:
        Dictionary with content_hash, content_length, and preview
    """
    if not content:
        return {
            "content_hash": "",
            "content_length": 0,
            "content_preview": ""
        }

    return {
        "content_hash": compute_content_hash(content),
        "content_length": len(content),
        "content_preview": content[:max_preview],
        "content_type": "metadata"
    }


def content_metadata_to_string(metadata: dict) -> str:
    """
    Convert content metadata dict to string for storage in Finding.content.
    Uses JSON format for structured data.
    """
    if isinstance(metadata, dict):
        return json.dumps(metadata)
    return str(metadata)


def parse_content_metadata(content_str: str) -> dict:
    """
    Parse content string to extract metadata dict.
    Expects JSON format with content_hash, content_length, and content_preview.
    """
    if not content_str:
        return {"content_hash": "", "content_length": 0, "content_preview": ""}

    # Parse as JSON format
    try:
        metadata = json.loads(content_str)
        if isinstance(metadata, dict) and "content_hash" in metadata:
            return metadata
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, treat as empty/invalid
        return {
            "content_hash": "",
            "content_length": 0,
            "content_preview": "",
            "content_type": "invalid"
        }

    # If JSON parsing succeeded but no content_hash, return empty
    return {"content_hash": "", "content_length": 0, "content_preview": ""}


def extract_evidence_summaries(
    findings: List[Finding],
    max_length: int = 500
) -> List[str]:
    """
    Extract concise evidence summaries from findings.
    Uses metadata and summaries instead of full content to reduce tokens.

    Args:
        findings: List of Finding objects
        max_length: Maximum length per summary

    Returns:
        List of evidence summary strings (optimized for token usage)
    """
    summaries = []

    for finding in findings:
        # Parse content metadata
        content_meta = parse_content_metadata(finding.content)

        # Use summary as primary evidence (already concise)
        # Include content preview if available and relevant
        evidence_parts = [finding.summary]

        # Add content preview if it provides additional context
        preview = content_meta.get("content_preview", "")
        if preview and len(preview) > 50:
            # Only add preview if it's different from summary
            if preview[:100] not in finding.summary:
                evidence_parts.append(f"Context: {preview[:max_length//2]}")

        # Combine evidence parts
        evidence = " | ".join(evidence_parts)[:max_length]

        # Format with task context (keep it concise)
        summary = f"Task: {finding.task[:40]}\n{evidence}"
        summaries.append(summary)

    return summaries


def extract_findings_metadata(findings: List[Finding]) -> dict:
    """
    Extract metadata from findings for use in prompts.

    Args:
        findings: List of Finding objects

    Returns:
        Dictionary with findings statistics and metadata
    """
    if not findings:
        return {
            "count": 0,
            "total_sources": 0,
            "avg_summary_length": 0,
            "tasks": []
        }

    total_sources = sum(len(f.sources) for f in findings)
    avg_summary_length = sum(len(f.summary) for f in findings) / len(findings)
    tasks = [f.task[:60] for f in findings]

    return {
        "count": len(findings),
        "total_sources": total_sources,
        "avg_summary_length": int(avg_summary_length),
        "tasks": tasks
    }


def extract_source_metadata(sources: List[Any]) -> Dict[str, str]:
    """
    Create identifier → title mapping for sources.

    Args:
        sources: List of Source objects or dicts

    Returns:
        Dictionary mapping source identifier to title
    """
    from retrieval import Source

    source_map = {}
    for source in sources:
        if isinstance(source, Source):
            source_map[source.identifier] = source.title or source.identifier
        elif isinstance(source, dict):
            identifier = source.get("identifier", "")
            title = source.get("title") or identifier
            source_map[identifier] = title
    return source_map


def create_findings_statistics(findings: List[Finding]) -> dict:
    """
    Create statistics about findings for use in prompts.

    Args:
        findings: List of Finding objects

    Returns:
        Dictionary with findings statistics
    """
    if not findings:
        return {
            "count": 0,
            "avg_length": 0,
            "coverage_areas": []
        }

    summaries = [f.summary for f in findings]
    avg_length = sum(len(s) for s in summaries) / len(summaries)

    # Extract coverage areas from task descriptions
    coverage_areas = list(set([
        f.task.split(":")[0].strip() for f in findings if ":" in f.task
    ]))

    return {
        "count": len(findings),
        "avg_length": int(avg_length),
        "coverage_areas": coverage_areas[:5]  # Top 5 areas
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
    """Unified citation entry for both extraction and reporting"""
    title: str = Field(
        ..., description="Title or reference text of the citation"
    )
    url: str = Field(
        default="", description="Source URL (optional for extracted papers)"
    )
    context: str = Field(
        default="",
        description="Brief context about what this paper discusses"
    )
    relevance: str = Field(
        default="", description="Why this citation might be relevant"
    )


class AnalysisOutput(DictCompatibleModel):
    """Output from subagent structured analysis"""
    summary: str = Field(
        ..., min_length=1, description="Summary of findings from search results"
    )
    citations: List[Citation] = Field(
        default_factory=list, description="Extracted citations"
    )
    python_code: Optional[str] = Field(
        default=None, description="Python code to run if calculations/filtering needed"
    )
    reasoning: str = Field(
        default="", description="Reasoning behind the summary and tool use"
    )


class VisitedSource(DictCompatibleModel):
    """Unified representation of a visited source"""
    identifier: str = Field(..., description="URL or document name")
    source_type: str = Field(..., description="Type: 'internal' or 'web'")
    visited_at: Optional[datetime] = Field(
        default=None, description="When source was visited"
    )

    def __hash__(self):
        """Make hashable for set operations"""
        return hash((self.identifier, self.source_type))

    def __eq__(self, other):
        """Equality based on identifier and source_type"""
        if not isinstance(other, VisitedSource):
            return False
        return (
            self.identifier == other.identifier
            and self.source_type == other.source_type
        )


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
    filtered_findings: List[Finding] = Field(
        default_factory=list, description="Quality-filtered findings"
    )

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
        default_factory=list, description="Final list of citations"
    )
    final_report: str = Field(default="", description="Final report")

    # Citation tracking - now using unified Citation object
    all_extracted_citations: Annotated[
        List[dict],
        operator.add
    ] = Field(
        default_factory=list,
        description="All citations extracted across all findings"
    )

    # Retry state
    error: str | None = Field(default=None, description="Error message from validation")
    retry_count: int = Field(default=0, description="Retry attempt count")

    # Conversation history for cost optimization
    lead_researcher_messages: Annotated[
        List[BaseMessage],
        operator.add
    ] = Field(
        default_factory=list,
        description="Conversation history for lead researcher (incremental updates)"
    )

    synthesizer_messages: Annotated[
        List[BaseMessage],
        operator.add
    ] = Field(
        default_factory=list,
        description="Conversation history for synthesizer (incremental updates)"
    )

    # Tracking for incremental processing
    processed_findings_ids: List[str] = Field(
        default_factory=list,
        description="Hash IDs of findings already processed by synthesizer"
    )

    rag_cache: Dict[str, str] = Field(
        default_factory=dict,
        description="Cached RAG retrieval results per query"
    )

    # Tracking for incremental lead researcher updates
    sent_finding_hashes: List[str] = Field(
        default_factory=list,
        description="Hash IDs of findings already sent to lead researcher"
    )

    previous_citation_count: int = Field(
        default=0,
        description="Count of citations from previous iteration (for delta tracking)"
    )

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
    def get_visited_identifiers(
        state: dict, source_type: Optional[str] = None
    ) -> List[str]:
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
    internal_result: Optional[Any] = Field(
        default=None, description="Internal knowledge retrieval result"
    )
    web_result: Optional[Any] = Field(
        default=None, description="Web search retrieval result"
    )
    subagent_findings: List[Finding] = Field(
        default_factory=list, description="Findings output"
    )
    visited_sources: List[VisitedSource] = Field(
        default_factory=list, description="Already visited sources"
    )
    extracted_citations: List[dict] = Field(
        default_factory=list, description="Citations extracted in this subagent run"
    )

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
    subagent_findings: List[Finding] = Field(..., min_length=1)
    synthesized_results: str = Field(..., min_length=1)
    all_extracted_citations: List[dict] = Field(
        default_factory=list, description="Extracted citations"
    )

    model_config = ConfigDict(extra="allow")


class VerifierState(DictCompatibleModel):
    """State needed by verifier"""
    query: str = Field(..., min_length=1)
    final_report: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="allow")


class CitationAgentState(DictCompatibleModel):
    """State needed by citation agent"""
    query: str = Field(..., min_length=1)
    subagent_findings: List[Finding] = Field(..., min_length=1)
    synthesized_results: str = Field(..., min_length=1)
    all_extracted_citations: List[Citation] = Field(
        default_factory=list, description="Extracted citations to include in sources"
    )

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Extraction Schemas (for unified extraction pattern)
# =============================================================================


class CitationExtractionResult(DictCompatibleModel):
    """Result of citation extraction"""
    citations: List[Citation] = Field(default_factory=list)
    has_citations: bool = Field(default=False)


# =============================================================================
# Extraction Schemas (for extraction_service)
# =============================================================================


class WebInsight(DictCompatibleModel):
    """Single insight from web content"""
    insight: str = Field(..., description="The main fact or finding")
    source: str = Field(..., description="URL or source title")
    relevance: str = Field(..., description="Why this is relevant")


class WebExtractionResult(DictCompatibleModel):
    """Result of web content extraction"""
    findings: List[WebInsight] = Field(default_factory=list)


class GenericItem(DictCompatibleModel):
    """Generic extracted item"""
    content: str = Field(..., description="The extracted content")
    context: str = Field(default="", description="Context or explanation")


class GenericExtractionResult(DictCompatibleModel):
    """Generic extraction result"""
    items: List[GenericItem] = Field(default_factory=list)
