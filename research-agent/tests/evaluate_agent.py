"""
Evaluation script for Deep Research Agent.
Runs a set of "Golden Questions" and saves the results for review.
Now supports LangSmith evaluation tracking.
"""

import asyncio
import os
import sys
import uuid

# Add project root/research-agent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402
from langsmith import Client, evaluate  # noqa: E402
from langsmith.schemas import Example, Run  # noqa: E402
from schemas import ResearchState  # noqa: E402

# =============================================================================
# Golden Dataset
# =============================================================================

DATASET_NAME = "Deep Research Agent Checkpoints"
GOLDEN_DATASET = [
    {
        "id": "tech_business_01",
        "category": "Tech/Business",
        "query": (
            "Who is the primary supplier of the LiDAR sensors for the "
            "Xiaomi SU7, and how did their stock price react on the week "
            "of the car's official launch in 2024?"
        ),
        "expected_findings": [
            "Identifying Hesai as supplier",
            "Stock price trend of Hesai during launch week"
        ]
    },
    {
        "id": "health_science_01",
        "category": "Health/Science",
        "query": (
            "Compare the weight loss efficacy of Semaglutide vs Tirzepatide "
            "in non-diabetic adults based on Phase 3 trials published "
            "between 2022-2024. Focus on % body weight reduction."
        ),
        "expected_findings": [
            "Semaglutide ~15%",
            "Tirzepatide ~20%",
            "Comparison favors Tirzepatide"
        ]
    },
    {
        "id": "policy_regulation_01",
        "category": "Policy/Regulation",
        "query": (
            "What are the key differences between the EU AI Act and the "
            "US Executive Order on AI regarding 'General Purpose AI' models?"
        ),
        "expected_findings": [
            "EU AI Act: Risk-based, binding",
            "US EO: Report-based, safety focus"
        ]
    },
    {
        "id": "memory_multi_hop_01",
        "category": "Memory/Multi-hop Reasoning",
        "query": (
            "Earlier you researched the relationship between attention mechanisms "
            "and computational efficiency in transformers. Now, analyze how the "
            "specific efficiency improvements mentioned in those papers (like "
            "Flash Attention, Sparse Attention, or Linear Attention variants) "
            "have been applied in production LLMs released between 2022-2024. "
            "For each efficiency technique, identify: (1) which papers first "
            "proposed it, (2) which LLM implementations adopted it first, "
            "(3) what performance gains were reported, and (4) what trade-offs "
            "or limitations were discovered in practice. Cross-reference this "
            "with any evaluation benchmarks that measured these LLMs' efficiency."
        ),
        "expected_findings": [
            "References to previous research on attention mechanisms",
            "Connection between theoretical efficiency papers and practical implementations",
            "Specific LLM adoptions (e.g., Flash Attention in GPT-4, etc.)",
            "Performance metrics and trade-offs",
            "Multi-hop reasoning: Papers -> Techniques -> Implementations -> Evaluations"
        ],
        "memory_requirements": [
            "Must recall previous query about attention mechanisms and efficiency",
            "Must connect theoretical papers to production systems",
            "Must demonstrate temporal understanding (2022-2024 timeline)",
            "Must show causal reasoning (technique -> adoption -> results -> trade-offs)",
            "Must link multiple information sources across different research sessions"
        ]
    }
]

# =============================================================================
# LangSmith Setup
# =============================================================================

client = Client()

def ensure_dataset():
    """Create dataset in LangSmith if it doesn't exist."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"✅ Dataset '{DATASET_NAME}' exists.")
        return

    print(f"Creates dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(dataset_name=DATASET_NAME,
    description=(
        "Golden questions for Deep Research Agent"
    )
    )

    for item in GOLDEN_DATASET:
        client.create_example(
            inputs={"query": item["query"]},
            outputs={"expected_findings": item.get("expected_findings")},
            metadata={"category": item["category"], "id": item["id"]},
            dataset_id=dataset.id,
        )
    print("✅ Dataset populated.")

# =============================================================================
# Evaluation Logic
# =============================================================================

async def predict_async(inputs: dict) -> dict:
    """Run the agent on a single input (async version)."""
    try:
        # Initialize state with Pydantic validation
        initial_state = ResearchState(
            query=inputs["query"],
            research_plan="",
            subagent_tasks=[],
            subagent_findings=[],
            iteration_count=0,
            needs_more_research=False,
            synthesized_results="",
            citations=[],
            final_report="",
        )

        # Generate a unique thread ID for checkpointer
        thread_id = str(uuid.uuid4())

        # Set recursion limit based on MAX_ITERATIONS
        # Each iteration involves multiple nodes, so we set a safe limit
        from config import settings
        recursion_limit = (settings.MAX_ITERATIONS + 1) * 15

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }

        # Use async invoke with timeout
        result = await asyncio.wait_for(
            app.ainvoke(initial_state, config=config),
            timeout=settings.TIMEOUT_MAIN
        )

        citations = result.get("citations", [])
        return {
            "final_report": result.get("final_report", ""),
            "citations_count": len(citations),
            "citations": [c.dict() if hasattr(c, 'dict') else c for c in citations]
        }
    except asyncio.TimeoutError:
        from config import settings
        return {
            "error": f"Timeout after {settings.TIMEOUT_MAIN}s",
            "final_report": "",
            "citations_count": 0,
            "citations": []
        }
    except Exception as e:
        return {
            "error": str(e),
            "final_report": "",
            "citations_count": 0,
            "citations": []
        }


def predict(inputs: dict) -> dict:
    """Run the agent on a single input (LangSmith interface - sync wrapper)."""
    return asyncio.run(predict_async(inputs))

def correctness_evaluator(run: Run, example: Example) -> dict:
    """Simple evaluator to check if a report was generated."""
    outputs = run.outputs or {}
    report = outputs.get("final_report", "")
    citations = outputs.get("citations_count", 0)

    score = 0.0
    reasoning = []

    # Basic Checks
    if report and len(report) > 100:
        score += 0.5
        reasoning.append("Generated a substantial report.")
    else:
        reasoning.append("Report missing or too short.")

    if citations >= 5:
        score += 0.5
        reasoning.append("Includes >5 citations.")
    else:
        reasoning.append("Insufficient citations.")

    return {"key": "basic_quality", "score": score, "comment": "; ".join(reasoning)}


def main():
    """Run LangSmith evaluation."""
    ensure_dataset()

    print(f"\n🚀 Starting LangSmith Evaluation on '{DATASET_NAME}'...")

    experiment_results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator],
        experiment_prefix="deep-research-v2",
        max_concurrency=1,  # Sequential to avoid rate limits
    )

    print("\n✅ Evaluation complete. View results in LangSmith.")
    print(experiment_results)

    # Legacy: Generate local report for user convenience
    # We can reconstruct it from the experiment results if needed,
    # but for now, rely on LangSmith UI for detailed view.

if __name__ == "__main__":
    main()
