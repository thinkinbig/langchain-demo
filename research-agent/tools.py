"""Web search tools using Tavily"""

import os
from typing import Dict, List

from tavily import TavilyClient


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Simple web search using Tavily API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        List of search results, each containing:
        - title: Result title
        - url: Result URL
        - content: Result content/snippet
    """
    try:
        # Get API key from environment
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        # Initialize client
        client = TavilyClient(api_key=api_key)

        # Perform search
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        # Extract results
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            })

        return results

    except Exception as e:
        # Graceful error handling
        print(f"⚠️  Search error: {e}")
        return []

