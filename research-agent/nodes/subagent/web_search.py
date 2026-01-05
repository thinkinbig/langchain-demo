"""Web search node: Perform web search and paper search if RAG failed"""

import asyncio

from retrieval import RetrievalResult, RetrievalService, RetrievalSource
from schemas import SubagentState


async def web_search_node(state: SubagentState):
    """
    Node: Perform web search and optionally paper search if RAG failed.
    
    This node searches both web (Tavily) and academic papers (arXiv, Semantic Scholar).
    The LLM in analysis_node will decide which sources to use based on the task.
    """
    task_description = state.get("task_description", "")

    if not task_description:
        # Get from tasks if not cached
        tasks = state.get("subagent_tasks", [])
        if tasks:
            rs_task = tasks[0]
            task_description = rs_task.description

    print(
        f"\n🔎 [Subagent-Web] Internal knowledge empty. "
        f"Performing Web & Paper Search: {task_description[:60]}..."
    )

    # Get visited URLs from unified format
    visited_sources = state.get("visited_sources", [])
    visited_web_urls = [
        vs.identifier
        for vs in visited_sources
        if vs.source_type == RetrievalSource.WEB.value
    ]
    visited_paper_urls = [
        vs.identifier
        for vs in visited_sources
        if vs.source_type == RetrievalSource.PAPER.value
    ]
    # Also include citation sources (extracted citations) to avoid duplicate searches
    visited_citation_urls = [
        vs.identifier
        for vs in visited_sources
        if vs.source_type == "citation"
    ]
    # Merge citation URLs into paper URLs (citations are typically papers)
    visited_paper_urls = list(set(visited_paper_urls + visited_citation_urls))

    # Use unified retrieval service (Async) with timeout
    from config import settings

    # Search both web and papers in parallel
    web_result = None
    paper_result = None

    # Web search
    try:
        web_result = await asyncio.wait_for(
            RetrievalService.aretrieve_web(
                query=task_description,
                visited_urls=visited_web_urls,
                max_results=5,
                scrape_top_result=True
            ),
            timeout=settings.TIMEOUT_WEB_SEARCH
        )
        print(f"  ✅ [WebSearch] Found {len(web_result.sources)} web sources")
    except asyncio.TimeoutError:
        print(f"  ⚠️  [WebSearch] Web search timeout ({settings.TIMEOUT_WEB_SEARCH}s)")
        web_result = RetrievalResult(
            content="(Web search timed out)",
            sources=[],
            source_type=RetrievalSource.WEB,
            has_content=False
        )
    except Exception as e:
        print(f"  ⚠️  [WebSearch] Error: {e}")
        web_result = RetrievalResult(
            content="(Web search failed)",
            sources=[],
            source_type=RetrievalSource.WEB,
            has_content=False
        )

    # Paper search (optional, based on config)
    use_semantic_scholar = getattr(settings, "USE_SEMANTIC_SCHOLAR", False)
    try:
        paper_result = await asyncio.wait_for(
            RetrievalService.aretrieve_papers(
                query=task_description,
                visited_urls=visited_paper_urls,
                max_results=5,
                use_semantic_scholar=use_semantic_scholar
            ),
            timeout=settings.TIMEOUT_WEB_SEARCH
        )
        if paper_result.has_content:
            print(f"  ✅ [PaperSearch] Found {len(paper_result.sources)} papers")
        else:
            print("  ℹ️  [PaperSearch] No papers found")
    except asyncio.TimeoutError:
        print(f"  ⚠️  [PaperSearch] Paper search timeout ({settings.TIMEOUT_WEB_SEARCH}s)")
        paper_result = RetrievalResult(
            content="(Paper search timed out)",
            sources=[],
            source_type=RetrievalSource.PAPER,
            has_content=False
        )
    except Exception as e:
        print(f"  ⚠️  [PaperSearch] Error: {e}")
        paper_result = RetrievalResult(
            content="(Paper search failed)",
            sources=[],
            source_type=RetrievalSource.PAPER,
            has_content=False
        )

    # Store both results - LLM will decide which to use
    return {
        "web_result": web_result,
        "paper_result": paper_result,
    }

