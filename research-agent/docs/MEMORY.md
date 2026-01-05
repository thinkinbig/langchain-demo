# Research Agent Memory System

The Research Agent employs a sophisticated **GraphRAG (Graph Retrieval-Augmented Generation)** system to manage knowledge, prevent hallucinations, and enable "deep research" capabilities.

## Core Components

### 1. The Knowledge Graph (`GraphStore`)
Instead of just storing chunks of text in a vector database, we maintain a **Knowledge Graph** where:
- **Nodes** represent Entities (concepts, people, organizations) or Papers/Documents.
- **Edges** represent relationships (e.g., `related_to`, `cites`, `authored_by`).

**Implementation**:
- **Backend**: `NetworkX` (in-memory graph operations)
- **Persistence**: JSON-based storage (`graph_store.json`).
- **Schema**:
    - `Paper`: Represents a scientific paper or article.
    - `Entity`: Extracted tokens or concepts from the text.

### 2. HippoRAG: Associative Memory
We implement a variant of **HippoRAG (Hippocampus-inspired RAG)** to model human-like associative memory.

**The Problem**:
Standard Vector RAG retrieves documents that are *semantically similar* to the query. It fails when the answer requires connecting two distant pieces of information that aren't semantically close to the query but are structurally related (e.g., "Paper A cites Paper B which solves Problem C").

**The Solution (PPR)**:
We use **Personalized PageRank (PPR)** to spread "activation" from the user's query through the knowledge graph.

**Algorithm Steps**:
1.  **Entity Extraction**: The system extracts entities from the user's query (e.g., "Transformer", "Attention").
2.  **Seed Activation**: These entities in the graph are "activated" (assigned a probability mass).
3.  **Spreading Activation**: We run the PageRank algorithm (with `alpha=0.85`), allowing this probability to flow to neighbors.
    - *Example*: "Transformer" -> activates "Attention Is All You Need" -> activates "Bahdanau Attention".
4.  **Retrieval**: We retrieve the top entities and the documents linked to them.

**Benefits**:
- **Multi-hop Reasoning**: Can retrieve documents that don't mention the query terms but are highly relevant via 1-2 degrees of separation.
- **Contextual Anchoring**: Provides a "bigger picture" view of the research landscape.

### 3. Vector Database (`ChromaDB`)
While the Graph handles structure and association, **ChromaDB** handles strict semantic similarity search for specific text chunks. It acts as the "cortical" long-term memory.

## Trade-offs and Design Decisions

| Feature | Design Choice | Rationale | Cost/Complexity |
| :--- | :--- | :--- | :--- |
| **Graph Engine** | `NetworkX` (In-Memory) | Simpler than Neo4j for an MVP. Easy to iterate on Python. | High memory usage for large graphs (>100k nodes). |
| **Retrieval** | Hybrid (PPR + Vector) | Captures both precise answers (Vector) and broad context (PPR). | Slower query latency due to dual tracking. |
| **Persistence** | JSON | Human-readable, easy to debug. | Slow load times at scale; no transaction support. |

## Future Improvements
- **Neo4j Migration**: For scaling beyond memory limits.
- **Dynamic Graph Updates**: Allowing the agent to prune incorrect edges during research.
