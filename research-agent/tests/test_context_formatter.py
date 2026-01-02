"""Unit tests for context formatter"""


from context_formatter import ContextFormatter
from retrieval import RetrievalResult, RetrievalSource, Source


class TestContextFormatter:
    """Test ContextFormatter"""

    def test_format_retrieval_result_internal(self):
        """Test formatting internal retrieval result"""
        formatter = ContextFormatter()

        sources = [
            Source(
                identifier="doc1.pdf",
                title="Document 1",
                source_type=RetrievalSource.INTERNAL
            )
        ]
        result = RetrievalResult(
            content="Internal knowledge content",
            sources=sources,
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        formatted = formatter.format_retrieval_result(result, [])

        assert (
            "INTERNAL KNOWLEDGE BASE" in formatted
            or "Internal knowledge content" in formatted
        )

    def test_format_retrieval_result_empty(self):
        """Test formatting empty retrieval result"""
        formatter = ContextFormatter()

        result = RetrievalResult(
            content="",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=False
        )

        formatted = formatter.format_retrieval_result(result, [])
        assert formatted == ""

    def test_format_citation_instructions_internal(self):
        """Test citation instructions for internal sources"""
        formatter = ContextFormatter()

        sources = [
            Source(
                identifier="doc1.pdf",
                title="Document 1",
                source_type=RetrievalSource.INTERNAL
            ),
            Source(
                identifier="doc2.pdf",
                title="Document 2",
                source_type=RetrievalSource.INTERNAL
            )
        ]

        instructions = formatter.format_citation_instructions(sources)

        assert "IMPORTANT" in instructions
        assert "submit_findings" in instructions
        assert "internal/doc1.pdf" in instructions or "Document 1" in instructions

    def test_format_citation_instructions_web(self):
        """Test citation instructions for web sources"""
        formatter = ContextFormatter()

        sources = [
            Source(
                identifier="http://example.com",
                title="Example Site",
                source_type=RetrievalSource.WEB
            )
        ]

        instructions = formatter.format_citation_instructions(sources)

        assert "IMPORTANT" in instructions
        assert "http://example.com" in instructions or "Example Site" in instructions

    def test_format_citation_instructions_empty(self):
        """Test citation instructions with no sources"""
        formatter = ContextFormatter()

        instructions = formatter.format_citation_instructions([])
        assert instructions == ""

    def test_combine_contexts_both(self):
        """Test combining internal and web contexts"""
        formatter = ContextFormatter()

        internal_result = RetrievalResult(
            content="Internal content",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        web_result = RetrievalResult(
            content="Web content",
            sources=[],
            source_type=RetrievalSource.WEB,
            has_content=True
        )

        combined = formatter.combine_contexts(internal_result, web_result)

        assert "Internal content" in combined
        assert "Web content" in combined

    def test_combine_contexts_internal_only(self):
        """Test combining with only internal context"""
        formatter = ContextFormatter()

        internal_result = RetrievalResult(
            content="Internal content",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        combined = formatter.combine_contexts(internal_result, None)

        assert "Internal content" in combined

    def test_combine_contexts_web_only(self):
        """Test combining with only web context"""
        formatter = ContextFormatter()

        web_result = RetrievalResult(
            content="Web content",
            sources=[],
            source_type=RetrievalSource.WEB,
            has_content=True
        )

        combined = formatter.combine_contexts(None, web_result)

        assert "Web content" in combined

    def test_format_for_analysis_internal(self):
        """Test formatting for analysis with internal result"""
        formatter = ContextFormatter()

        sources = [
            Source(
                identifier="doc1.pdf",
                title="Doc 1",
                source_type=RetrievalSource.INTERNAL
            )
        ]
        internal_result = RetrievalResult(
            content="Internal knowledge",
            sources=sources,
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        context, instructions, all_sources = formatter.format_for_analysis(
            task="Test task",
            internal_result=internal_result,
            web_result=None,
            visited_identifiers=[]
        )

        assert "Internal knowledge" in context or len(context) > 0
        assert len(all_sources) == 1
        assert all_sources[0].identifier == "doc1.pdf"
        assert "submit_findings" in instructions

    def test_format_for_analysis_web(self):
        """Test formatting for analysis with web result"""
        formatter = ContextFormatter()

        sources = [
            Source(
                identifier="http://example.com",
                title="Example",
                source_type=RetrievalSource.WEB
            )
        ]
        web_result = RetrievalResult(
            content="Web content",
            sources=sources,
            source_type=RetrievalSource.WEB,
            has_content=True
        )

        context, instructions, all_sources = formatter.format_for_analysis(
            task="Test task",
            internal_result=None,
            web_result=web_result,
            visited_identifiers=[]
        )

        assert len(context) > 0
        assert len(all_sources) == 1
        assert all_sources[0].identifier == "http://example.com"

    def test_format_for_analysis_both(self):
        """Test formatting for analysis with both results"""
        formatter = ContextFormatter()

        internal_result = RetrievalResult(
            content="Internal",
            sources=[
                Source(identifier="doc1.pdf", source_type=RetrievalSource.INTERNAL)
            ],
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        web_result = RetrievalResult(
            content="Web",
            sources=[
                Source(identifier="http://example.com", source_type=RetrievalSource.WEB)
            ],
            source_type=RetrievalSource.WEB,
            has_content=True
        )

        context, instructions, all_sources = formatter.format_for_analysis(
            task="Test task",
            internal_result=internal_result,
            web_result=web_result,
            visited_identifiers=[]
        )

        assert len(all_sources) == 2
        internal_count = len([
            s for s in all_sources if s.source_type == RetrievalSource.INTERNAL
        ])
        web_count = len([
            s for s in all_sources if s.source_type == RetrievalSource.WEB
        ])
        assert internal_count == 1
        assert web_count == 1

    def test_format_for_analysis_empty(self):
        """Test formatting for analysis with no results"""
        formatter = ContextFormatter()

        context, instructions, all_sources = formatter.format_for_analysis(
            task="Test task",
            internal_result=None,
            web_result=None,
            visited_identifiers=[]
        )

        assert context == ""
        assert instructions == ""
        assert len(all_sources) == 0

    def test_format_search_results_summary(self):
        """Test formatting search results summary"""
        formatter = ContextFormatter()

        search_results = [
            {
                "title": "Result 1",
                "url": "http://example.com/1",
                "content": "Content 1"
            },
            {
                "title": "Result 2",
                "url": "http://example.com/2",
                "content": "Content 2"
            }
        ]

        summary = formatter.format_search_results_summary(search_results)

        assert "Result 1" in summary
        assert "Result 2" in summary
        assert "Content 1" in summary or "Content 2" in summary

    def test_format_search_results_summary_empty(self):
        """Test formatting empty search results"""
        formatter = ContextFormatter()

        summary = formatter.format_search_results_summary([])
        assert summary == ""

    def test_custom_chunker(self):
        """Test formatter with custom chunker"""
        from context_chunker import ContextChunker

        custom_chunker = ContextChunker(max_tokens=5000)
        formatter = ContextFormatter(chunker=custom_chunker)

        assert formatter.chunker.max_tokens == 5000

