"""Web search and scraping tools"""

import ast
import contextlib
import logging
import os
import tempfile
import time
from io import StringIO
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient

# Try to import docker, but allow fallback if not available
try:
    import docker  # type: ignore
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

# Configure logging
logger = logging.getLogger(__name__)

# Docker sandbox configuration
DOCKER_IMAGE = os.getenv("PYTHON_SANDBOX_IMAGE", "python:3.11-slim")
MAX_EXECUTION_TIME = int(os.getenv("PYTHON_SANDBOX_TIMEOUT", "30"))
MEMORY_LIMIT = os.getenv("PYTHON_SANDBOX_MEMORY", "256m")
CPU_QUOTA = int(os.getenv("PYTHON_SANDBOX_CPU_QUOTA", "50000"))  # 50% of CPU
CPU_PERIOD = 100000
ENABLE_DOCKER = os.getenv("ENABLE_DOCKER_SANDBOX", "true").lower() == "true"


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

    SECURITY: This function implements multi-layer sandboxing:
    1. AST-based validation to block dangerous operations (pre-execution)
    2. Docker containerization with resource limits (production mode)
    3. Restricted builtins fallback (if Docker unavailable)

    For production use, Docker sandboxing provides complete isolation.
    See SECURITY_ANALYSIS.md for details.
    """
    start_time = time.time()
    code_preview = code[:100] + "..." if len(code) > 100 else code

    # Security: Limit code length
    MAX_CODE_LENGTH = 10000
    if len(code) > MAX_CODE_LENGTH:
        logger.warning(
            f"Code execution rejected: too long ({len(code)} chars)"
        )
        return f"Error: Code too long (max {MAX_CODE_LENGTH} characters)"

    # Security: AST validation (first layer of defense)
    try:
        tree = ast.parse(code)
        violations = _validate_code_safety(tree)
        if violations:
            logger.warning(
                f"Code execution blocked: security violations detected. "
                f"Code preview: {code_preview}"
            )
            return f"Security violation: {', '.join(violations)}"
    except SyntaxError as e:
        logger.warning(f"Code execution blocked: syntax error. Code: {code_preview}")
        return f"Syntax error: {e}"

    # Try Docker sandbox first (if enabled and available)
    if ENABLE_DOCKER and DOCKER_AVAILABLE:
        try:
            result = _execute_in_docker(code)
            execution_time = time.time() - start_time
            logger.info(
                f"Code executed in Docker sandbox. "
                f"Time: {execution_time:.2f}s, "
                f"Code length: {len(code)} chars"
            )
            return result
        except Exception as e:
            logger.warning(
                f"Docker execution failed, falling back to restricted execution: {e}"
            )
            # Fall through to restricted execution

    # Fallback: Restricted execution (if Docker unavailable or disabled)
    logger.info(
        f"Using restricted execution (Docker not available or disabled). "
        f"Code preview: {code_preview}"
    )
    return _execute_restricted(code, start_time)


def _execute_in_docker(code: str) -> str:
    """
    Execute Python code in an isolated Docker container.

    Security features:
    - Complete process isolation
    - Resource limits (CPU, memory, time)
    - Network disabled
    - Read-only volume mount
    - Automatic cleanup
    """
    client = None
    temp_file = None
    temp_dir = None

    try:
        # Initialize Docker client
        client = docker.from_env()
        client.ping()  # Verify Docker daemon is running

        # Create temporary directory and file
        temp_dir = tempfile.mkdtemp(prefix="python_sandbox_")
        script_name = "code.py"
        temp_file = os.path.join(temp_dir, script_name)

        # Write code to temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code)

        # Prepare code wrapper to capture output
        wrapper_code = """
import sys
from io import StringIO

# Capture stdout
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    exec(open('/code/code.py').read())
    output = sys.stdout.getvalue()
    if not output:
        print("(No output)")
    print(output, end='')
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
finally:
    sys.stdout = old_stdout
"""

        wrapper_file = os.path.join(temp_dir, "wrapper.py")
        with open(wrapper_file, "w", encoding="utf-8") as f:
            f.write(wrapper_code)

        # Run container with security constraints
        try:
            result = client.containers.run(
                image=DOCKER_IMAGE,
                command="python /code/wrapper.py",
                volumes={temp_dir: {"bind": "/code", "mode": "ro"}},
                mem_limit=MEMORY_LIMIT,
                cpu_period=CPU_PERIOD,
                cpu_quota=CPU_QUOTA,
                network_disabled=True,
                remove=True,
                timeout=MAX_EXECUTION_TIME,
                stderr=True,
                stdout=True,
            )

            # Decode output
            if isinstance(result, bytes):
                output = result.decode("utf-8", errors="replace")
            else:
                output = str(result)

            return output.strip() if output else "(No output)"

        except docker.errors.ContainerError as e:
            # Container returned non-zero exit code
            error_msg = (
                e.stderr.decode("utf-8", errors="replace")
                if e.stderr
                else str(e)
            )
            return f"Error executing code: {error_msg.strip()}"

        except docker.errors.Timeout:
            logger.warning(
                f"Docker execution timeout after {MAX_EXECUTION_TIME}s"
            )
            return f"Error: Execution timeout (max {MAX_EXECUTION_TIME}s)"

    except docker.errors.APIError as e:
        logger.error(f"Docker API error: {e}")
        raise Exception(f"Docker API error: {e}") from e

    except Exception as e:
        logger.error(f"Docker execution failed: {e}")
        raise

    finally:
        # Cleanup temporary files
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        if temp_dir and os.path.exists(temp_dir):
            try:
                # Remove wrapper file
                wrapper_file = os.path.join(temp_dir, "wrapper.py")
                if os.path.exists(wrapper_file):
                    os.unlink(wrapper_file)
                os.rmdir(temp_dir)
            except Exception:
                pass


def _execute_restricted(code: str, start_time: float) -> str:
    """
    Execute code in restricted environment (fallback when Docker unavailable).

    This provides basic sandboxing but is less secure than Docker.
    """
    # Create a string buffer to capture stdout
    io_buffer = StringIO()

    try:
        # Security: Restricted environment with safe builtins only
        safe_builtins = {
            # Math operations
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'divmod': divmod,
            # Type conversions
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            # Iteration
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'sorted': sorted, 'reversed': reversed,
            # String operations
            'ord': ord, 'chr': chr,
            # Basic functions
            'print': print, 'isinstance': isinstance, 'type': type,
            # Constants
            'True': True, 'False': False, 'None': None,
        }

        env = {"__builtins__": safe_builtins}

        # Parse AST again (already validated, but need for execution)
        tree = ast.parse(code)

        # Check if the last node is an expression
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
                exec(compile(tree, filename="<ast>", mode="exec"), env)

        output = io_buffer.getvalue()
        execution_time = time.time() - start_time
        logger.info(
            f"Code executed in restricted mode. "
            f"Time: {execution_time:.2f}s"
        )

        if not output:
            return "(No output)"
        return output.strip()

    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(
            f"Code execution failed after {execution_time:.2f}s: {e}"
        )
        return f"Error executing code: {e}"
    finally:
        io_buffer.close()


def _validate_code_safety(tree: ast.AST) -> list[str]:
    """
    Validate AST for dangerous operations.
    Returns list of violation messages, empty if safe.
    """
    violations = []

    # Dangerous imports to block
    DANGEROUS_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests',
        'urllib', 'http', 'ftplib', 'smtplib', 'telnetlib',
        'pickle', 'marshal', 'ctypes', 'multiprocessing',
        'threading', 'importlib', '__builtin__', 'builtins',
    }

    # Dangerous function calls to block
    DANGEROUS_FUNCTIONS = {
        'eval', 'exec', 'compile', '__import__', 'open',
        'file', 'input', 'raw_input', 'reload',
    }

    class SecurityChecker(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                module_name = alias.name.split('.')[0]  # Get root module
                if module_name in DANGEROUS_IMPORTS:
                    violations.append(
                        f"Dangerous import blocked: {alias.name}"
                    )
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in DANGEROUS_IMPORTS:
                    violations.append(
                        f"Dangerous import blocked: {node.module}"
                    )
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check for dangerous function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in DANGEROUS_FUNCTIONS:
                    violations.append(
                        f"Dangerous function call blocked: {node.func.id}"
                    )
            # Check for attribute access that might be dangerous
            elif isinstance(node.func, ast.Attribute):
                # Block __import__, eval, exec as attributes
                if node.func.attr in DANGEROUS_FUNCTIONS:
                    violations.append(
                        f"Dangerous method call blocked: {node.func.attr}"
                    )
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # Block access to dangerous attributes like __builtins__, __import__
            if isinstance(node.attr, str) and node.attr.startswith('__'):
                if node.attr in ('__import__', '__builtins__', '__globals__'):
                    violations.append(
                        f"Dangerous attribute access blocked: {node.attr}"
                    )
            self.generic_visit(node)

    checker = SecurityChecker()
    checker.visit(tree)
    return violations


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
