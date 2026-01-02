"""Main entry point for research agent"""

import asyncio
import uuid

from dotenv import load_dotenv

load_dotenv()


from graph import app  # noqa: E402
from schemas import ResearchState  # noqa: E402


async def main():
    """Run research agent on a query"""
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = (
            "What are the key differences between Python and Rust "
            "programming languages?"
        )

    print("=" * 80)
    print("MULTI-AGENT RESEARCH SYSTEM")
    print("=" * 80)
    print(f"\n🔍 Research Query: {query}\n")

    # Initialize Cost Controller
    from cost_control import (
        CostController,
        CostLimitExceeded,
        CostTrackingCallback,
        QueryBudget,
    )

    cost_controller = CostController()
    query_budget = QueryBudget()

    # 1. Check Daily Limit (Pre-flight check)
    # Estimate typical query cost (e.g., 50k tokens approx $0.10)
    can_accept, message = cost_controller.check_daily_limit(
        estimated_tokens=50_000, estimated_cost=0.10
    )
    if not can_accept:
        print(f"⛔ QUERY REJECTED: {message}")
        print("Daily budget exhausted. Please increase budget or wait for reset.")
        return

    print("✅ Daily budget check passed.")

    # Initialize state with Pydantic validation
    initial_state = ResearchState(
        query=query,
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

    try:
        # Run the graph with Cost Tracking Callback
        # This will auto-update query_budget on every LLM call
        cost_callback = CostTrackingCallback(query_budget)

        config = {
            "callbacks": [cost_callback],
            "configurable": {"thread_id": thread_id}
        }

        # Use async invoke for the graph
        final_state = await app.ainvoke(initial_state, config=config)

        # Display results
        print("\n" + "=" * 80)
        print("FINAL REPORT")
        print("=" * 80)
        print(final_state['final_report'] or "No report generated")
        print("=" * 80)

        print("\n📊 Summary:")
        print(f"   Iterations: {final_state['iteration_count']}")
        print(f"   Findings: {len(final_state['subagent_findings'])}")
        print(f"   Citations: {len(final_state['citations'])}")

    except CostLimitExceeded as e:
        print(f"\n⛔ EXCEPTION: Research stopped due to cost limit: {e}")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
    finally:
        # 2. Record actual usage to Daily Budget
        # Only what was consumed
        print("\n💰 Cost Report:")
        print(f"   Tokens Used: {query_budget.current_tokens}")
        print(f"   Est. Cost: ${query_budget.current_cost:.4f}")

        cost_controller.record_daily_usage(
            tokens=query_budget.current_tokens,
            cost=query_budget.current_cost
        )
        print("   (Recorded to daily budget)")


if __name__ == "__main__":
    asyncio.run(main())
