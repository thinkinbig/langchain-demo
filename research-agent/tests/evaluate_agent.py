"""
Evaluation script for Deep Research Agent.
Runs a set of "Golden Questions" and saves the results for review.
Now supports LangSmith evaluation tracking.
"""

import os
import sys

# Add project root/research-agent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402
from langsmith import Client, evaluate  # noqa: E402
from langsmith.schemas import Example, Run  # noqa: E402

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

def predict(inputs: dict) -> dict:
    """Run the agent on a single input (LangSmith interface)."""
    try:
        result = app.invoke({"query": inputs["query"]})
        citations = result.get("citations", [])
        return {
            "final_report": result.get("final_report", ""),
            "citations_count": len(citations),
            "citations": [c.dict() for c in citations]
        }
    except Exception as e:
        return {"error": str(e)}

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
