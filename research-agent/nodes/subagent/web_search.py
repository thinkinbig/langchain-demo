"""Web search node: Perform web search if RAG failed"""

from retrieval import RetrievalService
from schemas import SubagentState


async def web_search_node(state: SubagentState):
    """Node: Perform web search if RAG failed"""
    task_description = state.get("task_description", "")

    if not task_description:
        # Get from tasks if not cached
        tasks = state.get("subagent_tasks", [])
        if tasks:
            rs_task = tasks[0]
            task_description = rs_task.description

    print(
        f"\n🔎 [Subagent-Web] Internal knowledge empty. "
        f"Performing Web Search: {task_description[:60]}..."
    )

    # Get visited URLs from unified format
    visited_sources = state.get("visited_sources", [])
    visited_urls = [
        vs.identifier
        for vs in visited_sources
        if vs.source_type == "web"
    ]

    # Use unified retrieval service (Async) with timeout
    import asyncio

    from config import settings
    from retrieval import RetrievalResult
    try:
        result = await asyncio.wait_for(
            RetrievalService.aretrieve_web(
                query=task_description,
                visited_urls=visited_urls,
                max_results=5,
                scrape_top_result=True
            ),
            timeout=settings.TIMEOUT_WEB_SEARCH
        )
    except asyncio.TimeoutError:
        print(f"  ⚠️  [WebSearch] Web search timeout ({settings.TIMEOUT_WEB_SEARCH}s)")
        # Return empty result on timeout
        result = RetrievalResult(
            content="(Web search timed out)",
            sources=[],
            source_type="web",
            has_content=False
        )

    # Store in state using new unified format
    return {
        "web_result": result,
    }

