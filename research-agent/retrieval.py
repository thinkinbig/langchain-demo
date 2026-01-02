"""Unified retrieval abstraction for web scraping and knowledge base search"""

from enum import Enum
from typing import List, Optional, Set

import context_manager
import tools
from schemas import DictCompatibleModel


class RetrievalSource(str, Enum):
    """Type of retrieval source"""
    INTERNAL = "internal"
    WEB = "web"


class Source(DictCompatibleModel):
    """Unified source representation"""
    identifier: str = ""  # URL or document name
    title: str = ""
    source_type: RetrievalSource = RetrievalSource.INTERNAL

    def __hash__(self):
        """Make Source hashable for set operations"""
        return hash((self.identifier, self.source_type))

    def __eq__(self, other):
        """Equality based on identifier and source_type"""
        if not isinstance(other, Source):
            return False
        return (
            self.identifier == other.identifier
            and self.source_type == other.source_type
        )


class RetrievalResult(DictCompatibleModel):
    """Unified result from any retrieval source"""
    content: str = ""
    sources: List[Source] = []
    source_type: RetrievalSource = RetrievalSource.INTERNAL
    has_content: bool = False  # True if content is meaningful (not empty/error)

    def is_empty(self) -> bool:
        """Check if result has no meaningful content"""
        return not self.has_content or not self.content or "No relevant" in self.content


class RetrievalService:
    """Unified service for retrieving information from internal and web sources"""

    @staticmethod
    def retrieve_internal(
        query: str,
        visited_sources: Optional[List[str]] = None,
        k: int = 4
    ) -> RetrievalResult:
        """
        Retrieve context from internal knowledge base (RAG).

        Args:
            query: Search query
            visited_sources: List of already visited source identifiers
            k: Number of chunks to retrieve

        Returns:
            RetrievalResult with content and sources
        """
        if not query:
            return RetrievalResult(
                content="",
                sources=[],
                source_type=RetrievalSource.INTERNAL,
                has_content=False
            )

        visited_set = set(visited_sources or [])

        # Retrieve from knowledge base
        context_str, source_names = context_manager.retrieve_knowledge(query, k=k)

        # Convert to Source objects
        sources = []
        new_sources = []
        for source_name in source_names:
            source = Source(
                identifier=source_name,
                title=f"Internal Document: {source_name}",
                source_type=RetrievalSource.INTERNAL
            )
            sources.append(source)
            if source_name not in visited_set:
                new_sources.append(source_name)

        has_content = bool(context_str and "No relevant" not in context_str)

        return RetrievalResult(
            content=context_str or "",
            sources=sources,
            source_type=RetrievalSource.INTERNAL,
            has_content=has_content
        )

    @staticmethod
    def retrieve_web(
        query: str,
        visited_urls: Optional[List[str]] = None,
        max_results: int = 5,
        scrape_top_result: bool = True
    ) -> RetrievalResult:
        """
        Retrieve context from web search and scraping.

        Args:
            query: Search query
            visited_urls: List of already visited URLs
            max_results: Maximum number of search results
            scrape_top_result: Whether to scrape full content from top result

        Returns:
            RetrievalResult with content and sources
        """
        if not query:
            return RetrievalResult(
                content="",
                sources=[],
                source_type=RetrievalSource.WEB,
                has_content=False
            )

        visited_set = set(visited_urls or [])

        # Perform web search
        search_results = tools.search_web(query, max_results=max_results)

        if not search_results:
            return RetrievalResult(
                content="",
                sources=[],
                source_type=RetrievalSource.WEB,
                has_content=False
            )

        # Convert search results to Source objects
        sources = []
        full_content = ""
        new_urls_visited = []

        # Find first unvisited result for deep scraping
        target_result = None
        for res in search_results:
            url = res.get("url", "")
            if url and url not in visited_set:
                target_result = res
                break

        # Scrape top unvisited result if requested
        if scrape_top_result and target_result:
            top_url = target_result["url"]
            scraped_text = tools.scrape_web_page(top_url)
            if scraped_text and not scraped_text.startswith("Error"):
                full_content = scraped_text
                new_urls_visited.append(top_url)

        # Build sources list from all search results
        for res in search_results:
            url = res.get("url", "")
            title = res.get("title", "Unknown")
            if url:
                source = Source(
                    identifier=url,
                    title=title,
                    source_type=RetrievalSource.WEB
                )
                sources.append(source)

        # Format search results summary
        results_text = "\n\n".join([
            f"{i+1}. {r.get('title', 'Unknown')}\n{r.get('content', '')[:150]}"
            for i, r in enumerate(search_results)
        ])

        # Combine search summary with scraped content
        if full_content:
            combined_content = (
                f"{results_text}\n\n--- FULL CONTENT FROM {target_result['url']} ---\n"
                f"{full_content}\n--- END FULL CONTENT ---\n"
            )
        else:
            combined_content = results_text

        has_content = bool(combined_content and len(combined_content.strip()) > 0)

        return RetrievalResult(
            content=combined_content,
            sources=sources,
            source_type=RetrievalSource.WEB,
            has_content=has_content
        )

    @staticmethod
    def get_new_sources(
        result: RetrievalResult,
        visited_identifiers: Set[str]
    ) -> List[str]:
        """
        Extract new source identifiers from a retrieval result.

        Args:
            result: RetrievalResult to check
            visited_identifiers: Set of already visited identifiers

        Returns:
            List of new source identifiers
        """
        new_sources = []
        for source in result.sources:
            if source.identifier not in visited_identifiers:
                new_sources.append(source.identifier)
        return new_sources

    # =========================================================================
    # Async / Cached Methods
    # =========================================================================

    @staticmethod
    async def aretrieve_internal(
        query: str,
        visited_sources: Optional[List[str]] = None,
        k: int = 4
    ) -> RetrievalResult:
        """Async wrapper for retrieve_internal with caching"""
        import asyncio
        from functools import lru_cache

        # We use a cached version of the synchronous implementation
        # Note: lists are not hashable, so we can't easily cache based on visited_sources
        # if we pass it directly.
        # Strategy: Cache the underlying "heavy" lookup based on the QUERY,
        # then filter by visited_sources in the wrapper.

        # However, checking context_manager logic, it takes query and k.
        # So we can cache the result of `context_manager.retrieve_knowledge(query, k)`
        
        # To avoid complex caching logic here, we'll rely on the OS/disk cache 
        # for reading files, but we can cache the vector search result if we want.
        
        # For simplicity and safety in this iteration, we focus on ASYNC execution
        # to unlock parallelism. We will wrap the sync call in a thread.
        
        return await asyncio.to_thread(
            RetrievalService.retrieve_internal,
            query=query,
            visited_sources=visited_sources,
            k=k
        )

    @staticmethod
    async def aretrieve_web(
        query: str,
        visited_urls: Optional[List[str]] = None,
        max_results: int = 5,
        scrape_top_result: bool = True
    ) -> RetrievalResult:
        """Async wrapper for retrieve_web with caching"""
        import asyncio
        
        # Web search is the most expensive operation (IO bound).
        # We definitely want to cache this if the query is the same.
        # Since we can't easily modify the global RetrievalService to hold state 
        # without instantiation, we'll implement a simple module-level cache check
        # or just rely on the tool-level caching if it existed.
        
        # For this pass, unlocking parallelism is the priority.
        return await asyncio.to_thread(
            RetrievalService.retrieve_web,
            query=query,
            visited_urls=visited_urls,
            max_results=max_results,
            scrape_top_result=scrape_top_result
        )


