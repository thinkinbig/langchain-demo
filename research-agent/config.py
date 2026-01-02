"""
Configuration management for the Research Agent.
Uses Pydantic BaseSettings for environment variable management and validation.
"""


from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """
    Centralized configuration for the Research Agent.
    Values can be overridden by environment variables (e.g., AGENT_RETRIEVAL_K=5).
    """
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # =========================================================================
    # Retrieval Hyperparameters
    # =========================================================================

    # Threshold for vector similarity (L2 distance).
    # Lower is stricter (0.0 = identical).
    # 1.0 is default strictness (approx cosine 0.5).
    # 1.6 is loose (approx cosine 0.2), ensuring high recall for diverse queries.
    RETRIEVAL_L2_THRESHOLD: float = 1.6

    # Number of chunks to retrieve per query
    RETRIEVAL_K: int = 4

    # =========================================================================
    # LLM Configuration
    # =========================================================================

    # Model Names
    # Common options: "qwen-plus", "qwen-turbo", "qwen-max"
    # Note: Only two LLM instances are used:
    #   - lead_llm (for planning, synthesis, verification) uses MODEL_PLANNER
    #   - subagent_llm (for extraction, sub-tasks) uses MODEL_EXTRACTOR
    MODEL_PLANNER: str = "qwen-plus"
    MODEL_EXTRACTOR: str = "qwen-turbo"

    # Temperatures
    TEMP_PLANNER: float = 0.0      # Determinism for cost optimization
    TEMP_EXTRACTOR: float = 0.0    # Determinism

    # Base URL for API
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # =========================================================================
    # Budget Controls
    # =========================================================================

    MAX_TOKENS_PER_QUERY: int = 100_000
    MAX_ITERATIONS: int = 3
    MAX_SEARCH_CALLS: int = 20

    # =========================================================================
    # Timeout Configuration (in seconds)
    # =========================================================================

    # Overall timeout for the entire research process
    TIMEOUT_MAIN: float = 600.0  # 10 minutes

    # Timeout for subagent subgraph execution
    TIMEOUT_SUBAGENT: float = 180.0  # 3 minutes

    # Timeout for web search requests
    TIMEOUT_WEB_SEARCH: float = 60.0  # 1 minute

    # Timeout for internal retrieval (RAG)
    TIMEOUT_RETRIEVAL: float = 30.0  # 30 seconds

    # Timeout for Python code execution
    TIMEOUT_PYTHON_REPL: float = 30.0  # 30 seconds

    # Timeout for LLM calls (optional, LLM may have built-in timeout)
    TIMEOUT_LLM_CALL: float = 120.0  # 2 minutes

    # =========================================================================
    # Checkpointer Configuration
    # =========================================================================

    # Checkpointer backend: 'memory', 'sqlite', or 'postgres'
    # Note: SQLite/Postgres support may require additional packages or
    # newer langgraph version
    CHECKPOINTER_BACKEND: str = "memory"

    # SQLite database path (for sqlite backend)
    CHECKPOINTER_DB_PATH: str = "checkpoints.db"

    # PostgreSQL connection string (for postgres backend)
    # Format: postgresql://user:password@host:port/database
    CHECKPOINTER_CONNECTION_STRING: str = ""


# Global singleton
settings = AgentSettings()
