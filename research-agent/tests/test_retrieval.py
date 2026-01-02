"""Unit tests for retrieval abstraction"""

from unittest.mock import patch

from retrieval import (
    RetrievalResult,
    RetrievalService,
    RetrievalSource,
    Source,
)


class TestSource:
    """Test Source model"""

    def test_source_creation(self):
        """Test creating a Source"""
        source = Source(
            identifier="test.pdf",
            title="Test Document",
            source_type=RetrievalSource.INTERNAL
        )
        assert source.identifier == "test.pdf"
        assert source.title == "Test Document"
        assert source.source_type == RetrievalSource.INTERNAL

    def test_source_hashable(self):
        """Test that Source is hashable"""
        source1 = Source(identifier="test.pdf", source_type=RetrievalSource.INTERNAL)
        source2 = Source(identifier="test.pdf", source_type=RetrievalSource.INTERNAL)
        source3 = Source(identifier="test.pdf", source_type=RetrievalSource.WEB)

        assert hash(source1) == hash(source2)
        assert hash(source1) != hash(source3)

    def test_source_equality(self):
        """Test Source equality"""
        source1 = Source(identifier="test.pdf", source_type=RetrievalSource.INTERNAL)
        source2 = Source(identifier="test.pdf", source_type=RetrievalSource.INTERNAL)
        source3 = Source(identifier="other.pdf", source_type=RetrievalSource.INTERNAL)

        assert source1 == source2
        assert source1 != source3


class TestRetrievalResult:
    """Test RetrievalResult model"""

    def test_retrieval_result_creation(self):
        """Test creating a RetrievalResult"""
        sources = [
            Source(
                identifier="doc1.pdf",
                title="Doc 1",
                source_type=RetrievalSource.INTERNAL
            )
        ]
        result = RetrievalResult(
            content="Test content",
            sources=sources,
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )
        assert result.content == "Test content"
        assert len(result.sources) == 1
        assert result.has_content is True

    def test_is_empty(self):
        """Test is_empty method"""
        empty_result = RetrievalResult(
            content="",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=False
        )
        assert empty_result.is_empty() is True

        no_relevant = RetrievalResult(
            content="(No relevant internal documents found)",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=False
        )
        assert no_relevant.is_empty() is True

        has_content = RetrievalResult(
            content="Some content",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )
        assert has_content.is_empty() is False


class TestRetrievalService:
    """Test RetrievalService"""

    @patch("retrieval.context_manager.retrieve_knowledge")
    def test_retrieve_internal_success(self, mock_retrieve):
        """Test successful internal retrieval"""
        mock_retrieve.return_value = (
            "Test context content",
            ["doc1.pdf", "doc2.pdf"]
        )

        result = RetrievalService.retrieve_internal(
            query="test query",
            visited_sources=[],
            k=4
        )

        assert result.has_content is True
        assert result.content == "Test context content"
        assert len(result.sources) == 2
        assert result.sources[0].identifier == "doc1.pdf"
        assert result.source_type == RetrievalSource.INTERNAL
        mock_retrieve.assert_called_once_with("test query", k=4)

    @patch("retrieval.context_manager.retrieve_knowledge")
    def test_retrieve_internal_no_relevant(self, mock_retrieve):
        """Test internal retrieval with no relevant results"""
        mock_retrieve.return_value = (
            "(No relevant internal documents found)",
            []
        )

        result = RetrievalService.retrieve_internal(
            query="test query",
            visited_sources=[],
            k=4
        )

        assert result.has_content is False
        assert result.is_empty() is True

    @patch("retrieval.context_manager.retrieve_knowledge")
    def test_retrieve_internal_with_visited(self, mock_retrieve):
        """Test internal retrieval with visited sources"""
        mock_retrieve.return_value = (
            "Test context",
            ["doc1.pdf", "doc2.pdf"]
        )

        result = RetrievalService.retrieve_internal(
            query="test query",
            visited_sources=["doc1.pdf"],
            k=4
        )

        # Should still return all sources, filtering happens elsewhere
        assert len(result.sources) == 2

    @patch("retrieval.tools.scrape_web_page")
    @patch("retrieval.tools.search_web")
    def test_retrieve_web_success(self, mock_search, mock_scrape):
        """Test successful web retrieval"""
        mock_search.return_value = [
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
        mock_scrape.return_value = "Full scraped content"

        result = RetrievalService.retrieve_web(
            query="test query",
            visited_urls=[],
            max_results=5,
            scrape_top_result=True
        )

        assert result.has_content is True
        assert len(result.sources) == 2
        assert result.sources[0].identifier == "http://example.com/1"
        assert result.source_type == RetrievalSource.WEB
        assert "Full scraped content" in result.content
        mock_search.assert_called_once_with("test query", max_results=5)

    @patch("retrieval.tools.search_web")
    def test_retrieve_web_empty_results(self, mock_search):
        """Test web retrieval with empty results"""
        mock_search.return_value = []

        result = RetrievalService.retrieve_web(
            query="test query",
            visited_urls=[],
            max_results=5
        )

        assert result.has_content is False
        assert result.is_empty() is True
        assert len(result.sources) == 0

    @patch("retrieval.tools.scrape_web_page")
    @patch("retrieval.tools.search_web")
    def test_retrieve_web_with_visited(self, mock_search, mock_scrape):
        """Test web retrieval with visited URLs"""
        mock_search.return_value = [
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
        mock_scrape.return_value = "Scraped content"

        result = RetrievalService.retrieve_web(
            query="test query",
            visited_urls=["http://example.com/1"],
            max_results=5,
            scrape_top_result=True
        )

        # Should scrape the first unvisited URL (result 2)
        assert result.has_content is True
        # Scrape should be called with the unvisited URL
        mock_scrape.assert_called_once_with("http://example.com/2")

    def test_get_new_sources(self):
        """Test get_new_sources helper"""
        sources = [
            Source(identifier="doc1.pdf", source_type=RetrievalSource.INTERNAL),
            Source(identifier="doc2.pdf", source_type=RetrievalSource.INTERNAL),
            Source(identifier="http://example.com", source_type=RetrievalSource.WEB)
        ]
        result = RetrievalResult(
            content="test",
            sources=sources,
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        visited = {"doc1.pdf"}
        new_sources = RetrievalService.get_new_sources(result, visited)

        assert "doc2.pdf" in new_sources
        assert "http://example.com" in new_sources
        assert "doc1.pdf" not in new_sources


class TestBackwardCompatibility:
    """Test backward compatibility with old state format"""

    def test_retrieval_result_dict_compatible(self):
        """Test that RetrievalResult works with dict-style access"""
        result = RetrievalResult(
            content="test",
            sources=[],
            source_type=RetrievalSource.INTERNAL,
            has_content=True
        )

        # Should support dict-style access
        assert result["content"] == "test"
        assert result.get("content") == "test"
        assert result.get("nonexistent", "default") == "default"

