"""Voting Pattern: Run the same task multiple times for diverse outputs

This demonstrates the Voting parallelization pattern:
- Run the same task multiple times in parallel
- Get diverse perspectives/evaluations
- Aggregate votes to reach consensus
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from schemas import VotingState

# Initialize LLM with different temperatures for diversity
# Using separate client instances for true parallel execution
llm_strict = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.1,  # More deterministic
    max_retries=2,
)

llm_creative = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,  # More creative
    max_retries=2,
)

llm_balanced = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.4,  # Balanced
    max_retries=2,
)


def voting_worker_node(state: VotingState):
    """Each worker evaluates the same task independently"""
    import time
    task = state.get("task", "")
    worker_id = state.get("worker_id", "unknown")
    start_time = time.time()
    print(
        f"\n🗳️  [Voting Worker {worker_id}] Evaluating task... "
        f"(started at {start_time:.2f})"
    )

    # Different prompts for different perspectives
    prompts = {
        "strict": (
            "Evaluate this task with strict, conservative criteria:\n\n"
            f"{task}\n\n"
            "Focus on: accuracy, safety, potential risks, and strict "
            "adherence to requirements."
        ),

        "creative": (
            "Evaluate this task with creative, innovative thinking:\n\n"
            f"{task}\n\n"
            "Focus on: novel approaches, creative solutions, and "
            "out-of-the-box ideas."
        ),
        "balanced": (
            "Evaluate this task with balanced, practical criteria:\n\n"
            f"{task}\n\n"
            "Focus on: practicality, feasibility, and balanced trade-offs."
        ),
    }

    llms = {
        "strict": llm_strict,
        "creative": llm_creative,
        "balanced": llm_balanced,
    }

    # Use different LLM based on worker_id
    perspective = worker_id if worker_id in prompts else "balanced"
    prompt = prompts.get(perspective, prompts["balanced"])
    llm = llms.get(perspective, llm_balanced)

    response = llm.invoke([
        SystemMessage(
            content=f"You are an expert evaluator with a {perspective} perspective."
        ),
        HumanMessage(content=prompt),
    ])

    end_time = time.time()
    elapsed = end_time - start_time
    vote = f"[{perspective.upper()} Perspective]\n{response.content}"
    print(f"  ✅ Vote from {perspective} perspective completed in {elapsed:.2f}s")
    return {"votes": [vote]}


def assign_voters(state: VotingState):
    """Fan-out: Assign task to multiple voting workers in parallel"""
    task = state["task"]
    print("\n🔀 [Fan-out] Distributing task to 3 voting workers in parallel...")

    # Return list of Send() to create parallel execution
    return [
        Send("voting_worker", {"task": task, "worker_id": "strict"}),
        Send("voting_worker", {"task": task, "worker_id": "creative"}),
        Send("voting_worker", {"task": task, "worker_id": "balanced"}),
    ]


def consensus_node(state: VotingState):
    """Aggregate votes to reach consensus"""
    votes = state.get("votes", [])
    task = state["task"]
    print(f"\n📊 [Consensus] Analyzing {len(votes)} votes...")

    all_votes = "\n\n---\n\n".join(votes)

    prompt = f"""Analyze these independent evaluations and reach a consensus:

Original Task: {task}

Independent Votes:
{all_votes}

Provide:
1. A consensus evaluation/answer
2. Key points of agreement
3. Important considerations from different perspectives
4. Final recommendation"""
    response = llm_balanced.invoke([
        SystemMessage(
            content="You are a consensus builder. Synthesize diverse perspectives."
        ),
        HumanMessage(content=prompt),
    ])

    print(f"  ✅ Consensus reached ({len(response.content)} chars)")
    return {"consensus": response.content}


# Build graph
workflow = StateGraph(VotingState)

# Add nodes
workflow.add_node("voting_worker", voting_worker_node)
workflow.add_node("consensus", consensus_node)

# Add edges
# Use conditional_edges with Send() to fan-out to multiple workers
workflow.add_conditional_edges(
    START,
    assign_voters,
    ["voting_worker"],
)
workflow.add_edge("voting_worker", "consensus")
workflow.add_edge("consensus", END)

# Compile app
voting_app = workflow.compile()

