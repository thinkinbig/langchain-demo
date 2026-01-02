"""Context formatting service for consistent prompt construction"""

from typing import List, Optional

from context_chunker import ContextChunker
from retrieval import RetrievalResult, RetrievalSource, Source


class ContextFormatter:
    """Service for formatting retrieval results into prompts"""

    def __init__(self, chunker: Optional[ContextChunker] = None):
        """
        Initialize context formatter.
        
        Args:
            chunker: ContextChunker instance (creates default if None)
        """
        self.chunker = chunker or ContextChunker()

    def format_retrieval_result(
        self,
        result: RetrievalResult,
        visited_identifiers: Optional[List[str]] = None
    ) -> str:
        """
        Format a single retrieval result for prompt inclusion.
        
        Args:
            result: RetrievalResult to format
            visited_identifiers: List of already visited source identifiers
            
        Returns:
            Formatted context string
        """
        if result.is_empty():
            return ""

        visited_set = set(visited_identifiers or [])

        # Get primary source identifier for formatting
        primary_source = ""
        if result.sources:
            primary_source = result.sources[0].identifier

        # Prepare content with chunking and limits
        formatted = self.chunker.prepare_context(
            content=result.content,
            source_type=result.source_type,
            source_identifier=primary_source,
            enforce_limit=True
        )

        return formatted

    def format_citation_instructions(self, sources: List[Source]) -> str:
        """
        Generate citation instructions for the LLM.
        
        Args:
            sources: List of sources to cite
            
        Returns:
            Citation instruction string
        """
        if not sources:
            return ""

        formatted_sources = []
        for source in sources:
            if source.source_type == RetrievalSource.INTERNAL:
                formatted_sources.append({
                    "title": source.title or f"Internal Document: {source.identifier}",
                    "url": f"internal/{source.identifier}"
                })
            else:
                formatted_sources.append({
                    "title": source.title or source.identifier,
                    "url": source.identifier
                })

        instruction = (
            f"\n\nIMPORTANT: You have been provided with information from the following sources: "
            f"{[s.get('title', 'Unknown') for s in formatted_sources]}. "
            f"If you use this information, you MUST include it in the `sources` argument of `submit_findings`. "
            f"Suggested sources format: {formatted_sources}"
        )

        return instruction

    def combine_contexts(
        self,
        internal: Optional[RetrievalResult],
        web: Optional[RetrievalResult]
    ) -> str:
        """
        Combine internal and web contexts into a single formatted string.
        
        Args:
            internal: Internal knowledge base result
            web: Web search result
            
        Returns:
            Combined formatted context
        """
        parts = []

        if internal and not internal.is_empty():
            parts.append(self.format_retrieval_result(internal))

        if web and not web.is_empty():
            parts.append(self.format_retrieval_result(web))

        return "\n\n".join(parts)

    def format_for_analysis(
        self,
        task: str,
        internal_result: Optional[RetrievalResult] = None,
        web_result: Optional[RetrievalResult] = None,
        visited_identifiers: Optional[List[str]] = None
    ) -> tuple[str, str, List[Source]]:
        """
        Format all contexts for analysis node prompt.
        
        This is the main entry point for preparing context for the analysis node.
        
        Args:
            task: Task description
            internal_result: Internal knowledge base result
            web_result: Web search result
            visited_identifiers: List of already visited identifiers
            
        Returns:
            Tuple of (formatted_context, citation_instructions, all_sources)
        """
        all_sources = []
        context_parts = []

        # Format internal context
        if internal_result and not internal_result.is_empty():
            formatted_internal = self.format_retrieval_result(
                internal_result,
                visited_identifiers
            )
            if formatted_internal:
                context_parts.append(formatted_internal)
                all_sources.extend(internal_result.sources)

        # Format web context
        if web_result and not web_result.is_empty():
            formatted_web = self.format_retrieval_result(
                web_result,
                visited_identifiers
            )
            if formatted_web:
                context_parts.append(formatted_web)
                all_sources.extend(web_result.sources)

        # Combine contexts
        full_context = "\n\n".join(context_parts)

        # Generate citation instructions
        citation_instructions = self.format_citation_instructions(all_sources)

        return full_context, citation_instructions, all_sources

    def format_search_results_summary(self, search_results: List[dict]) -> str:
        """
        Format web search results into a summary string.
        
        This is used when we have search results but haven't scraped yet.
        
        Args:
            search_results: List of search result dicts with 'title', 'url', 'content'
            
        Returns:
            Formatted summary string
        """
        if not search_results:
            return ""

        formatted = "\n\n".join([
            f"{i+1}. {r.get('title', 'Unknown')}\n{r.get('content', '')[:150]}"
            for i, r in enumerate(search_results)
        ])

        return formatted

