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
    # Three models available for autonomous selection based on task complexity:
    #   - MODEL_TURBO: Low cost, fast, suitable for simple tasks
    #   - MODEL_PLUS: Balanced, suitable for medium tasks
    #   - MODEL_MAX: High quality, suitable for complex tasks (production)
    MODEL_TURBO: str = "qwen-turbo"
    MODEL_PLUS: str = "qwen-plus"
    MODEL_MAX: str = "qwen-max"

    # Temperatures
    TEMP_PLANNER: float = 0.0      # Determinism for cost optimization
    TEMP_EXTRACTOR: float = 0.0    # Determinism

    # Base URL for API
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Enable MAX model for complex tasks (only in production)
    # Set AGENT_ENABLE_MAX_MODEL=true to enable max model for complex tasks
    # Default: False (use plus model instead to save costs)
    ENABLE_MAX_MODEL: bool = False

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
    # Increased to accommodate multiple iterations with refinement
    TIMEOUT_MAIN: float = 1200.0  # 20 minutes (increased from 10)

    # Timeout for subagent subgraph execution
    TIMEOUT_SUBAGENT: float = 300.0  # 5 minutes (increased from 3)

    # Timeout for web search requests
    TIMEOUT_WEB_SEARCH: float = 90.0  # 1.5 minutes (increased from 1)

    # Timeout for internal retrieval (RAG)
    TIMEOUT_RETRIEVAL: float = 45.0  # 45 seconds (increased from 30)

    # Timeout for Python code execution
    TIMEOUT_PYTHON_REPL: float = 60.0  # 1 minute (increased from 30)

    # Timeout for LLM calls (optional, LLM may have built-in timeout)
    # Increased to accommodate complex refinement steps
    TIMEOUT_LLM_CALL: float = 180.0  # 3 minutes (increased from 2)

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

    # =========================================================================
    # Synthesis Configuration
    # =========================================================================

    # Use SCR (Situation-Complication-Resolution) structure for synthesis
    USE_SCR_STRUCTURE: bool = True

    # Early decision optimization: enable decision after partial synthesis
    # When enabled, decision is made after Situation+Complication, skipping
    # Resolution if continuing research (saves ~33% synthesis time)
    ENABLE_EARLY_DECISION: bool = True

    # Early decision point: "situation" or "complication"
    # "complication" = after Situation+Complication (recommended, balanced)
    # "situation" = after Situation only (faster but less information)
    EARLY_DECISION_AFTER: str = "complication"

    # =========================================================================
    # GraphRAG Configuration
    # =========================================================================

    # Enable/disable GraphRAG (Knowledge Graph integration)
    GRAPH_ENABLED: bool = True

    # Path for graph persistence (relative to research-agent directory)
    GRAPH_PERSIST_PATH: str = "graph_store.json"

    # Weight for graph signals in reranking (0.0-1.0)
    # Higher values give more weight to graph connectivity
    GRAPH_RERANK_WEIGHT: float = 0.25

    # Maximum hops for neighbor expansion in graph context retrieval
    GRAPH_MAX_HOPS: int = 2

    # =========================================================================
    # HippoRAG/PPR Configuration
    # =========================================================================

    # Enable/disable PPR-based retrieval (HippoRAG methodology)
    # When enabled, replaces BFS-based neighborhood expansion with
    # Personalized PageRank for multi-hop associative retrieval
    USE_PPR_RETRIEVAL: bool = True

    # Damping factor for PPR (restart probability)
    # Standard PageRank value: 0.85
    # Lower values = more localized search, higher = more global
    PPR_ALPHA: float = 0.85

    # Maximum iterations for PPR convergence
    PPR_MAX_ITER: int = 100

    # Convergence tolerance for PPR
    PPR_TOL: float = 1e-6

    # Number of top nodes to retrieve from PPR
    PPR_TOP_K_NODES: int = 20

    # Number of documents to retrieve from top nodes
    PPR_TOP_K_DOCS: int = 10

    # Weight for PPR results in hybrid retrieval (vs vector search)
    # 0.0 = only vector search, 1.0 = only PPR
    PPR_WEIGHT: float = 0.4


# Global singleton
settings = AgentSettings()
