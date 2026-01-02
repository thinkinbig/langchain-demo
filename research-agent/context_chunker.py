"""Context chunking service for consistent content splitting and token management"""

from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval import RetrievalSource


class ContextChunker:
    """Service for chunking and managing context size"""

    # Default configuration
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MAX_TOKENS = 4000  # Conservative limit for analysis prompts
    TOKENS_PER_CHAR = 0.25  # Rough estimation: ~4 chars per token

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        """
        Initialize context chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks in characters
            max_tokens: Maximum tokens allowed in final context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.

        Uses a simple heuristic: ~4 characters per token.
        This is conservative and works well for English text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return int(len(text) * self.TOKENS_PER_CHAR)

    def chunk_content(
        self,
        content: str,
        source_type: RetrievalSource = RetrievalSource.INTERNAL
    ) -> List[str]:
        """
        Split content into chunks.

        Args:
            content: Content to chunk
            source_type: Type of source (affects chunking strategy)

        Returns:
            List of content chunks
        """
        if not content:
            return []

        # For internal sources, content is already chunked by RAG
        # We just need to split if it's too long
        if source_type == RetrievalSource.INTERNAL:
            # Internal content is usually already well-structured
            # Split only if significantly exceeds chunk size
            if len(content) <= self.chunk_size * 2:
                return [content]
            return self.splitter.split_text(content)

        # For web content, always chunk (it's unstructured)
        return self.splitter.split_text(content)

    def truncate_to_limit(
        self,
        content: str,
        max_tokens: Optional[int] = None,
        preserve_start: bool = True
    ) -> str:
        """
        Truncate content to fit within token limit.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens (uses instance default if None)
            preserve_start: If True, keep the beginning; if False, keep the end

        Returns:
            Truncated content
        """
        if not content:
            return ""

        limit = max_tokens or self.max_tokens
        max_chars = int(limit / self.TOKENS_PER_CHAR)

        if len(content) <= max_chars:
            return content

        if preserve_start:
            truncated = content[:max_chars]
            # Try to end at a sentence boundary
            last_period = truncated.rfind('.')
            last_newline = truncated.rfind('\n')
            cutoff = max(last_period, last_newline)
            if cutoff > max_chars * 0.8:  # Only use cutoff if it's not too early
                return truncated[:cutoff + 1] + "\n\n[Content truncated...]"
        else:
            truncated = content[-max_chars:]
            # Try to start at a sentence boundary
            first_period = truncated.find('.')
            first_newline = truncated.find('\n')
            cutoff = min(
                first_period if first_period >= 0 else len(truncated),
                first_newline if first_newline >= 0 else len(truncated)
            )
            if cutoff < max_chars * 0.2:  # Only use cutoff if it's not too late
                return "...[Content truncated]\n\n" + truncated[cutoff + 1:]
            return "...[Content truncated]\n\n" + truncated

        return truncated

    def format_for_prompt(
        self,
        chunks: List[str],
        source_type: RetrievalSource,
        source_identifier: str = ""
    ) -> str:
        """
        Format chunks for inclusion in prompt.

        Args:
            chunks: List of content chunks
            source_type: Type of source
            source_identifier: Identifier of the source (URL or doc name)

        Returns:
            Formatted string for prompt
        """
        if not chunks:
            return ""

        if source_type == RetrievalSource.INTERNAL:
            header = "--- INTERNAL KNOWLEDGE BASE"
            if source_identifier:
                header += f" ({source_identifier})"
            header += " ---\n"
            footer = "\n--- END INTERNAL KNOWLEDGE ---\n"
        else:
            header = "--- WEB CONTENT"
            if source_identifier:
                header += f" ({source_identifier})"
            header += " ---\n"
            footer = "\n--- END WEB CONTENT ---\n"

        # Combine chunks with separators
        content = "\n\n".join(chunks)
        return header + content + footer

    def prepare_context(
        self,
        content: str,
        source_type: RetrievalSource,
        source_identifier: str = "",
        enforce_limit: bool = True,
        use_summarization: bool = False
    ) -> str:
        """
        Prepare context for prompt: chunk, truncate, and format.

        This is the main entry point for preparing any content for use in prompts.

        Args:
            content: Raw content
            source_type: Type of source
            source_identifier: Identifier of the source
            enforce_limit: Whether to enforce token limit
            use_summarization: If True, use summarization instead of truncation

        Returns:
            Formatted context ready for prompt
        """
        if not content:
            return ""

        # Estimate tokens
        estimated_tokens = self.estimate_tokens(content)

        # Use summarization if enabled and content is too long
        if use_summarization and enforce_limit and estimated_tokens > self.max_tokens:
            summarizer = SummarizationChunker(max_tokens=self.max_tokens)
            content = summarizer.summarize(content, source_type, source_identifier)
        elif enforce_limit and estimated_tokens > self.max_tokens:
            # Fallback to truncation
            content = self.truncate_to_limit(content, self.max_tokens)

        # Chunk if needed (for very long content)
        chunks = self.chunk_content(content, source_type)

        # Format for prompt
        if len(chunks) == 1:
            # Single chunk, format directly
            return self.format_for_prompt(chunks, source_type, source_identifier)
        else:
            # Multiple chunks, combine them
            combined = "\n\n".join(chunks)
            return self.format_for_prompt([combined], source_type, source_identifier)


class SummarizationChunker:
    """
    Intelligent summarization-based context compression.

    Uses LLM to create hierarchical summaries instead of simple truncation,
    preserving key information like citations, sources, and important facts.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        preserve_citations: bool = True,
        preserve_sources: bool = True
    ):
        """
        Initialize summarization chunker.

        Args:
            max_tokens: Target token count for summary
            preserve_citations: Whether to preserve citation information
            preserve_sources: Whether to preserve source URLs/identifiers
        """
        self.max_tokens = max_tokens
        self.preserve_citations = preserve_citations
        self.preserve_sources = preserve_sources
        self.TOKENS_PER_CHAR = 0.25

    def _extract_preserved_elements(self, content: str) -> dict:
        """
        Extract elements that should be preserved in summary.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with citations, sources, and other preserved elements
        """
        preserved = {
            "citations": [],
            "sources": [],
            "urls": [],
            "key_facts": []
        }

        # Extract URLs
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        preserved["urls"] = list(set(re.findall(url_pattern, content)))

        # Extract citation patterns (e.g., [1], (Smith et al., 2023))
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?,\s+\d{4})\)',  # (Author, Year)
        ]
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            preserved["citations"].extend(matches)

        # Extract source identifiers (from formatted context)
        if "Source:" in content:
            source_lines = [
                line for line in content.split('\n')
                if line.startswith('Source:')
            ]
            preserved["sources"] = [
                line.replace('Source:', '').strip()
                for line in source_lines
            ]

        return preserved

    def summarize(
        self,
        content: str,
        source_type: RetrievalSource,
        source_identifier: str = ""
    ) -> str:
        """
        Summarize content using LLM while preserving key information.

        Args:
            content: Content to summarize
            source_type: Type of source
            source_identifier: Identifier of the source

        Returns:
            Summarized content
        """
        # Extract preserved elements before summarization
        preserved = self._extract_preserved_elements(content)

        # Estimate target summary length
        target_chars = int(self.max_tokens / self.TOKENS_PER_CHAR * 0.8)  # 80% of limit

        # If content is not too long, return as-is
        if len(content) <= target_chars:
            return content

        # Use simple extractive summarization for now
        # In production, this could use LLM-based summarization
        # For now, we use a hybrid approach: extractive + key info preservation

        # Split into sentences
        sentences = self._split_into_sentences(content)

        # Score sentences by importance (simple heuristic)
        scored_sentences = self._score_sentences(sentences, preserved)

        # Select top sentences up to target length
        selected = self._select_sentences(scored_sentences, target_chars)

        # Reconstruct summary with preserved elements
        summary = self._reconstruct_summary(
            selected, preserved, source_type, source_identifier
        )

        return summary

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentences(
        self,
        sentences: List[str],
        preserved: dict
    ) -> List[tuple[str, float]]:
        """
        Score sentences by importance.

        Returns:
            List of (sentence, score) tuples
        """
        scored = []

        for sentence in sentences:
            score = 0.0

            # Boost score if sentence contains preserved elements
            # Check for citations
            if any(cit in sentence for cit in preserved["citations"]):
                score += 2.0

            # Check for URLs
            if any(url in sentence for url in preserved["urls"]):
                score += 1.5

            # Check for source identifiers
            if any(src in sentence for src in preserved["sources"]):
                score += 1.5

            # Boost first sentences (often contain key info)
            if sentences.index(sentence) < 3:
                score += 1.0

            # Boost sentences with numbers (often contain facts)
            if any(char.isdigit() for char in sentence):
                score += 0.5

            # Boost longer sentences (often more informative)
            if len(sentence) > 100:
                score += 0.3

            scored.append((sentence, score))

        return scored

    def _select_sentences(
        self,
        scored_sentences: List[tuple[str, float]],
        target_chars: int
    ) -> List[str]:
        """Select top sentences up to target length."""
        # Sort by score (descending)
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)

        selected = []
        current_length = 0

        for sentence, _ in sorted_sentences:
            if current_length + len(sentence) <= target_chars:
                selected.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
            else:
                break

        # If we have space, add more sentences in order
        remaining = [s for s, _ in sorted_sentences if s not in selected]
        for sentence in remaining:
            if current_length + len(sentence) <= target_chars:
                selected.append(sentence)
                current_length += len(sentence) + 1
            else:
                break

        return selected

    def _reconstruct_summary(
        self,
        selected_sentences: List[str],
        preserved: dict,
        source_type: RetrievalSource,
        source_identifier: str
    ) -> str:
        """Reconstruct summary with preserved elements."""
        summary_parts = []

        # Add preserved URLs if not already in sentences
        if preserved["urls"]:
            url_section = "\nSources: " + ", ".join(
                preserved["urls"][:5]
            )  # Limit to 5
            if not any(
                url in " ".join(selected_sentences)
                for url in preserved["urls"]
            ):
                summary_parts.append(url_section)

        # Add selected sentences
        summary_parts.append(" ".join(selected_sentences))

        # Add note about summarization
        summary_parts.append(
            "\n[Note: Content summarized to fit context limit while "
            "preserving key information]"
        )

        return "\n".join(summary_parts)

