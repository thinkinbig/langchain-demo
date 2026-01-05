"""Text processing utilities for cleaning and formatting output"""

import re
from typing import List


def clean_xml_tags(text: str, tags: List[str] | None = None) -> str:
    """
    Remove XML-like tags from text while preserving the content inside.

    This function handles common cases where LLMs may include XML tags
    in their output (e.g., <report></report>, <query></query>).

    Args:
        text: The text to clean
        tags: Optional list of specific tag names to remove.
              If None, removes common tags: report, query, findings,
              synthesis, initial_synthesis, reflection_analysis,
              decision_context, decision_guidance, decision_improvements

    Returns:
        Cleaned text with tags removed but content preserved

    Examples:
        >>> clean_xml_tags("<report>Hello</report>")
        'Hello'
        >>> clean_xml_tags("Text <query>query</query> more text")
        'Text query more text'
        >>> clean_xml_tags("<report>\\nContent\\n</report>")
        '\\nContent\\n'
    """
    if not text:
        return text

    # Default tags to clean (common tags used in prompts)
    default_tags = [
        "report",
        "query",
        "findings",
        "synthesis",
        "initial_synthesis",
        "reflection_analysis",
        "decision_context",
        "decision_guidance",
        "decision_improvements",
        "decision_task_guidance",
        "task",
        "data",
        "instructions",
    ]

    tags_to_remove = tags if tags is not None else default_tags

    cleaned = text

    # Remove each tag (both opening and closing)
    for tag in tags_to_remove:
        # Pattern matches:
        # - <tag> or <tag > (with optional whitespace)
        # - </tag> or </tag > (with optional whitespace)
        # Handles multiline tags and whitespace variations
        pattern = rf"</?{re.escape(tag)}\s*>"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Clean up extra whitespace that might result from tag removal
    # Replace multiple consecutive newlines (3+) with double newline
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def clean_report_output(text: str) -> str:
    """
    Clean report output by removing XML tags and normalizing whitespace.

    This is a convenience wrapper around clean_xml_tags specifically
    for cleaning final report outputs.

    Args:
        text: The report text to clean

    Returns:
        Cleaned report text
    """
    return clean_xml_tags(text)

