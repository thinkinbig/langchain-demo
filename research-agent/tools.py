"""Web search and scraping tools"""

import os
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
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
    # Simple retry logic
    max_retries = 3
    for attempt in range(max_retries):
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
            # Graceful error handling with retry
            print(f"  ⚠️  Search error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)  # Brief backoff
            else:
                print(f"  ❌ All search attempts failed for query: {query[:50]}...")
                return []


def scrape_web_page(url: str) -> str:
    """
    Scrape text content from a web page.

    Args:
        url: URL to scrape

    Returns:
        str: Extracted text content, or error message
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text[:5000]  # Limit content length

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"
