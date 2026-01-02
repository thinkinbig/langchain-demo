"""Web search and scraping tools"""

import ast
import contextlib
import os
from io import StringIO
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Simple web search using Tavily API. Returns a list of results.
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


def python_repl(code: str) -> str:
    """
    Execute Python code and return the output (stdout).
    Useful for analysis, filtering data, or calculations.

    WARNING: This executes code locally. Use with caution.
    """
    # Create a string buffer to capture stdout
    io_buffer = StringIO()

    try:
        # Shared environment for exec and eval
        env = {"__builtins__": __builtins__}

        # Parse the code to check if the last node is an expression
        tree = ast.parse(code)
        last_node = tree.body[-1] if tree.body else None

        # If last node is an expression, we want to print its result
        if isinstance(last_node, ast.Expr):
            # Remove the last node from the tree
            tree.body.pop()
            # Compile and exec the rest
            if tree.body:
                exec(compile(tree, filename="<ast>", mode="exec"), env)

            # Eval the last expression and print it
            last_expr = compile(
                ast.Expression(last_node.value),
                filename="<ast>",
                mode="eval",
            )
            with contextlib.redirect_stdout(io_buffer):
                result = eval(last_expr, env)
                if result is not None:
                    print(result)
        else:
            # Just exec everything
            with contextlib.redirect_stdout(io_buffer):
                exec(code, env)

        output = io_buffer.getvalue()
        if not output:
             return "(No output)"
        return output.strip()

    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        io_buffer.close()


def read_local_file(file_path: str) -> str:
    """
    Read a local file from the filesystem.
    Supports plain text (.txt, .md, .py, etc.) and PDF (.pdf).
    """
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."

    try:
        # Check for PDF
        if file_path.lower().endswith(".pdf"):
            try:
                # Lazy import to avoid hard dependency if not used
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text.append(f"--- Page {i+1} ---\n{page_text}")
                return "\n".join(text)
            except ImportError:
                return "Error: pypdf not installed. Please install it to read PDFs."
            except Exception as e:
                return f"Error reading PDF {file_path}: {str(e)}"

        # Default to text
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"
