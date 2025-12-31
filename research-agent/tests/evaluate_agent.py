"""
Evaluation script for Deep Research Agent.
Runs a set of "Golden Questions" and saves the results for review.
"""

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from graph import app

# =============================================================================
# Golden Dataset
# =============================================================================

GOLDEN_DATASET = [
    {
        "id": "tech_business_01",
        "category": "Tech/Business",
        "query": "Who is the primary supplier of the LiDAR sensors for the Xiaomi SU7, and how did their stock price react on the week of the car's official launch in 2024?"
    },
    {
        "id": "health_science_01",
        "category": "Health/Science",
        "query": "Compare the weight loss efficacy of Semaglutide vs Tirzepatide in non-diabetic adults based on Phase 3 trials published between 2022-2024. Focus on % body weight reduction."
    },
    {
        "id": "policy_regulation_01",
        "category": "Policy/Regulation",
        "query": "What are the key differences between the EU AI Act and the US Executive Order on AI regarding 'General Purpose AI' models?"
    }
]

# =============================================================================
# Evaluation Logic
# =============================================================================

def run_evaluation(question: Dict) -> Dict:
    """Run a single question through the agent."""
    print(f"\n🚀 Starting Evaluation: {question['id']}")
    print(f"   Query: {question['query']}")
    
    start_time = time.time()
    try:
        # Invoke agent
        initial_state = {"query": question["query"]}
        result = app.invoke(initial_state)
        
        duration = time.time() - start_time
        
        final_report = result.get("final_report", "No report generated.")
        citations = result.get("citations", [])
        
        return {
            "id": question["id"],
            "query": question["query"],
            "status": "success",
            "duration_seconds": round(duration, 2),
            "final_report": final_report,
            "citations_count": len(citations),
            "citations": [c.dict() for c in citations]
        }
        
    except Exception as e:
        print(f"❌ Error in {question['id']}: {e}")
        return {
            "id": question["id"],
            "query": question["query"],
            "status": "error",
            "error": str(e)
        }

def main():
    """Run all evaluations and save report."""
    results = []
    
    # Run sequentially to avoid rate limits/context window issues
    for q in GOLDEN_DATASET:
        res = run_evaluation(q)
        results.append(res)
        print(f"✅ Finished {q['id']} in {res.get('duration_seconds', 'N/A')}s")
        
    # Save results to Markdown
    output_file = "evaluation_report.md"
    with open(output_file, "w") as f:
        f.write("# Deep Research Agent Evaluation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for res in results:
            f.write(f"## {res['id']}\n")
            f.write(f"**Query:** {res['query']}\n")
            f.write(f"**Status:** {res['status']}\n")
            if res['status'] == 'success':
                f.write(f"**Duration:** {res['duration_seconds']}s\n")
                f.write(f"**Citations:** {res['citations_count']}\n\n")
                f.write("### Final Report\n\n")
                
                # Indent headers in the report to nest correctly (Level 1 -> Level 3)
                report_content = res['final_report']
                # Increase header level by 2
                report_content = report_content.replace("# ", "### ").replace("## ", "#### ").replace("### ", "##### ")
                
                f.write(report_content)
                f.write("\n\n---\n\n")
            else:
                f.write(f"**Error:** {res.get('error')}\n\n")
    
    print(f"\n📄 Evaluation report saved to {output_file}")

if __name__ == "__main__":
    main()
