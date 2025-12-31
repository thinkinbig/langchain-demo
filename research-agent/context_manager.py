"""
Context Manager for Long-Term Memory (Enterprise Knowledge)

This module handles the loading and formatting of the static "Knowledge Base"
for the agent. It implements the "Long Context" + "Prompt Caching" pattern
by constructing a cacheable system prompt block.
"""

import glob
import os
from typing import Optional

# Global cache for the loaded knowledge retrieval
_CACHED_KNOWLEDGE_BLOCK: Optional[str] = None


def load_long_term_memory(source_dir: str = "data") -> str:
    """
    Load all text documents from the source directory and format them
    into an XML-like structure for the LLM.

    Args:
        source_dir: Directory containing knowledge base files (relative to package root)

    Returns:
        str: Formatted XML string <knowledge_base>...</knowledge_base>
    """
    global _CACHED_KNOWLEDGE_BLOCK

    # Return valid cache if available
    if _CACHED_KNOWLEDGE_BLOCK:
        return _CACHED_KNOWLEDGE_BLOCK

    # Resolve absolute path
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, source_dir)

    if not os.path.exists(full_path):
        print(f"  ⚠️  Knowledge base directory not found: {full_path}")
        return "<knowledge_base>\n(No internal documents found)\n</knowledge_base>"

    # Find all text-based files
    files = []
    patterns = ["*.txt", "*.md", "*.csv"]
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(full_path, pattern)))

    if not files:
        print(f"  ⚠️  No documents found in: {full_path}")
        return "<knowledge_base>\n(No internal documents found)\n</knowledge_base>"

    print(f"  📚 Loading {len(files)} documents from knowledge base...")

    # Build XML structure
    xml_parts = ["<knowledge_base>"]

    for i, file_path in enumerate(sorted(files)):
        try:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            xml_parts.append(f'  <document id="{i+1}" source="{filename}">')
            xml_parts.append(f"{content}")
            xml_parts.append('  </document>')

        except Exception as e:
            print(f"  ❌ Failed to load {file_path}: {e}")

    xml_parts.append("</knowledge_base>")

    # Cache the result
    _CACHED_KNOWLEDGE_BLOCK = "\n".join(xml_parts)
    return _CACHED_KNOWLEDGE_BLOCK


def get_system_context(
    role_instructions: str,
    include_knowledge: bool = True
) -> str:
    """
    Construct the full system context for an agent, including the
    long-term memory block (if requested).

    This entire string is suitable for Prompt Caching (Prefix Caching).

    Args:
        role_instructions: The specific system instructions for this agent role.
        include_knowledge: Whether to append the enterprise knowledge base.

    Returns:
        str: The full system prompt content.
    """
    parts = [role_instructions]

    if include_knowledge:
        knowledge_block = load_long_term_memory()
        parts.append("\n\n--- ENTERPRISE KNOWLEDGE BASE ---\n")
        parts.append(
            "You have access to the following internal knowledge. "
            "Always search this knowledge base first before relying on "
            "general knowledge."
        )
        parts.append(knowledge_block)
        parts.append("\n--- END KNOWLEDGE BASE ---")

    return "\n".join(parts)


def clear_cache():
    """Clear the memory cache (useful for tests or reloading)"""
    global _CACHED_KNOWLEDGE_BLOCK
    _CACHED_KNOWLEDGE_BLOCK = None
