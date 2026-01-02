"""Memory helpers for metadata extraction and content optimization"""

import hashlib
import json
from typing import Dict, List

from retrieval import Source
from schemas import Finding


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
    Handles both JSON format and legacy string format.
    """
    if not content_str:
        return {"content_hash": "", "content_length": 0, "content_preview": ""}

    # Try to parse as JSON (new format)
    try:
        metadata = json.loads(content_str)
        if isinstance(metadata, dict) and "content_hash" in metadata:
            return metadata
    except (json.JSONDecodeError, TypeError):
        pass

    # Legacy format: if it's a short hash-like string, treat as reference
    if len(content_str) <= 16 and content_str.isalnum():
        return {
            "content_hash": content_str,
            "content_length": 0,
            "content_preview": "",
            "content_type": "legacy_hash"
        }

    # Legacy format: full content string
    return {
        "content_hash": compute_content_hash(content_str),
        "content_length": len(content_str),
        "content_preview": content_str[:200],
        "content_type": "legacy_full"
    }


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


def extract_source_metadata(sources: List[Source]) -> Dict[str, str]:
    """
    Create identifier → title mapping for sources.

    Args:
        sources: List of Source objects

    Returns:
        Dictionary mapping source identifier to title
    """
    source_map = {}
    for source in sources:
        source_map[source.identifier] = source.title or source.identifier
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

