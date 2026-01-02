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
                return truncated[:cutoff + 1] + "\n\n[Content truncated..."
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
        enforce_limit: bool = True
    ) -> str:
        """
        Prepare context for prompt: chunk, truncate, and format.

        This is the main entry point for preparing any content for use in prompts.

        Args:
            content: Raw content
            source_type: Type of source
            source_identifier: Identifier of the source
            enforce_limit: Whether to enforce token limit

        Returns:
            Formatted context ready for prompt
        """
        if not content:
            return ""

        # Estimate tokens
        estimated_tokens = self.estimate_tokens(content)

        # Truncate if needed
        if enforce_limit and estimated_tokens > self.max_tokens:
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

