"""Unit tests for context chunker"""


from context_chunker import ContextChunker
from retrieval import RetrievalSource


class TestContextChunker:
    """Test ContextChunker"""

    def test_estimate_tokens(self):
        """Test token estimation"""
        chunker = ContextChunker()

        # Rough estimate: ~4 chars per token
        assert chunker.estimate_tokens("") == 0
        assert chunker.estimate_tokens("test") == 1  # 4 chars = 1 token
        assert chunker.estimate_tokens("test " * 10) == 10  # 50 chars = ~12 tokens

    def test_truncate_to_limit_preserve_start(self):
        """Test truncation preserving start"""
        chunker = ContextChunker(max_tokens=100)  # ~400 chars

        long_text = "A. " + "word " * 200  # Much longer than 400 chars
        truncated = chunker.truncate_to_limit(long_text, preserve_start=True)

        assert len(truncated) <= 400
        assert truncated.startswith("A.")
        assert "[Content truncated" in truncated or len(truncated) < len(long_text)

    def test_truncate_to_limit_preserve_end(self):
        """Test truncation preserving end"""
        chunker = ContextChunker(max_tokens=100)

        long_text = "word " * 200 + "Z."
        truncated = chunker.truncate_to_limit(long_text, preserve_start=False)

        assert len(truncated) <= 400
        assert truncated.endswith("Z.") or "[Content truncated" in truncated

    def test_truncate_no_truncation_needed(self):
        """Test truncation when content is within limit"""
        chunker = ContextChunker(max_tokens=1000)

        short_text = "This is a short text."
        truncated = chunker.truncate_to_limit(short_text)

        assert truncated == short_text

    def test_chunk_content_internal(self):
        """Test chunking internal content"""
        chunker = ContextChunker(chunk_size=100, chunk_overlap=20)

        # Short content should not be chunked
        short_content = "Short content"
        chunks = chunker.chunk_content(short_content, RetrievalSource.INTERNAL)
        assert len(chunks) == 1
        assert chunks[0] == short_content

        # Long content should be chunked
        long_content = "word " * 100  # Much longer than 100 chars
        chunks = chunker.chunk_content(long_content, RetrievalSource.INTERNAL)
        assert len(chunks) >= 1

    def test_chunk_content_web(self):
        """Test chunking web content"""
        chunker = ContextChunker(chunk_size=100, chunk_overlap=20)

        # Web content should always be chunked if long enough
        long_content = "word " * 100
        chunks = chunker.chunk_content(long_content, RetrievalSource.WEB)
        assert len(chunks) >= 1

    def test_format_for_prompt_internal(self):
        """Test formatting internal content for prompt"""
        chunker = ContextChunker()

        chunks = ["Chunk 1", "Chunk 2"]
        formatted = chunker.format_for_prompt(
            chunks,
            RetrievalSource.INTERNAL,
            "doc1.pdf"
        )

        assert "INTERNAL KNOWLEDGE BASE" in formatted
        assert "doc1.pdf" in formatted
        assert "Chunk 1" in formatted
        assert "Chunk 2" in formatted
        assert "END INTERNAL KNOWLEDGE" in formatted

    def test_format_for_prompt_web(self):
        """Test formatting web content for prompt"""
        chunker = ContextChunker()

        chunks = ["Web content chunk"]
        formatted = chunker.format_for_prompt(
            chunks,
            RetrievalSource.WEB,
            "http://example.com"
        )

        assert "WEB CONTENT" in formatted
        assert "http://example.com" in formatted
        assert "Web content chunk" in formatted
        assert "END WEB CONTENT" in formatted

    def test_prepare_context_short(self):
        """Test preparing short context"""
        chunker = ContextChunker(max_tokens=1000)

        content = "Short content"
        prepared = chunker.prepare_context(
            content,
            RetrievalSource.INTERNAL,
            "doc1.pdf",
            enforce_limit=True
        )

        assert "INTERNAL KNOWLEDGE BASE" in prepared
        assert "Short content" in prepared

    def test_prepare_context_long(self):
        """Test preparing long context"""
        chunker = ContextChunker(max_tokens=100)  # Small limit

        # Create content that exceeds limit
        long_content = "word " * 200  # Much longer than 400 chars
        prepared = chunker.prepare_context(
            long_content,
            RetrievalSource.INTERNAL,
            "doc1.pdf",
            enforce_limit=True
        )

        # Should be truncated
        assert len(prepared) < len(long_content) or "[Content truncated" in prepared

    def test_prepare_context_no_enforce_limit(self):
        """Test preparing context without enforcing limit"""
        chunker = ContextChunker(max_tokens=100)

        long_content = "word " * 200
        prepared = chunker.prepare_context(
            long_content,
            RetrievalSource.INTERNAL,
            "doc1.pdf",
            enforce_limit=False
        )

        # Should not be truncated (but may be chunked)
        assert "word" in prepared

    def test_prepare_context_empty(self):
        """Test preparing empty context"""
        chunker = ContextChunker()

        prepared = chunker.prepare_context(
            "",
            RetrievalSource.INTERNAL,
            "doc1.pdf"
        )

        assert prepared == ""

    def test_custom_configuration(self):
        """Test chunker with custom configuration"""
        chunker = ContextChunker(
            chunk_size=500,
            chunk_overlap=100,
            max_tokens=2000
        )

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.max_tokens == 2000

