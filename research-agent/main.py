"""Main entry point for research agent"""

from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402


def main():
    """Run research agent on a query"""
    query = "What are the key differences between Python and Rust programming languages?"

    print("=" * 80)
    print("MULTI-AGENT RESEARCH SYSTEM")
    print("=" * 80)
    print(f"\n🔍 Research Query: {query}\n")

    # Initialize state
    initial_state = {
        "query": query,
        "research_plan": "",
        "subagent_tasks": [],
        "subagent_findings": [],
        "iteration_count": 0,
        "needs_more_research": False,
        "synthesized_results": "",
        "citations": [],
        "final_report": "",
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    # Display results
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(final_state.get("final_report", "No report generated"))
    print("=" * 80)

    print("\n📊 Summary:")
    print(f"   Iterations: {final_state.get('iteration_count', 0)}")
    print(f"   Findings: {len(final_state.get('subagent_findings', []))}")
    print(f"   Citations: {len(final_state.get('citations', []))}")


if __name__ == "__main__":
    main()

