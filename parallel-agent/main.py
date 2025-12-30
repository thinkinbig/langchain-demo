import time

from dotenv import load_dotenv

load_dotenv()

from sectioning import sectioning_app  # noqa: E402
from voting import voting_app  # noqa: E402

if __name__ == "__main__":
    print("=" * 80)
    print("PARALLELIZATION PATTERNS DEMONSTRATION")
    print("=" * 80)

    # Sectioning examples
    sectioning_tasks = [
        "Design a comprehensive security system for a cloud-based application",
        "Create a marketing strategy for a new AI product launch",
        "Evaluate the pros and cons of microservices vs monolithic architecture",
    ]

    print("\n" + "=" * 80)
    print("PATTERN 1: SECTIONING (Breaking into parallel subtasks)")
    print("=" * 80)

    for i, task in enumerate(sectioning_tasks, 1):
        print(f"\n{'─' * 80}")
        print(f"Sectioning Example {i}: {task}")
        print("─" * 80)

        initial_state = {
            "task": task,
            "sections": [],
            "section_results": [],
            "final_summary": "",
        }

        start_time = time.time()
        final_state = sectioning_app.invoke(initial_state)
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")

        print("\n" + "=" * 80)
        print("SECTIONING RESULTS")
        print("=" * 80)
        print(f"Sections created: {len(final_state['sections'])}")
        print(f"Results aggregated: {len(final_state['section_results'])}")

        print("\nFinal Summary:")
        print("-" * 80)
        print(final_state["final_summary"])
        print("=" * 80)

    # Voting examples
    voting_tasks = [
        (
            "Review this code for security vulnerabilities: "
            "def process_user_data(data): return data.upper()"
        ),
        (
            "Evaluate whether this content is appropriate for a professional setting: "
            "'I think we should consider alternative approaches'"
        ),
        (
            "Assess the feasibility of implementing a real-time recommendation "
            "system for 10M users"
        ),
    ]

    print("\n\n" + "=" * 80)
    print("PATTERN 2: VOTING (Multiple perspectives in parallel)")
    print("=" * 80)

    for i, task in enumerate(voting_tasks, 1):
        print(f"\n{'─' * 80}")
        print(f"Voting Example {i}: {task}")
        print("─" * 80)

        initial_state = {
            "task": task,
            "perspectives": ["strict", "creative", "balanced"],
            "vote_results": [],
            "final_decision": "",
        }

        start_time = time.time()
        final_state = voting_app.invoke(initial_state)
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")

        print("\n" + "=" * 80)
        print("VOTING RESULTS")
        print("=" * 80)
        print(f"Votes collected: {len(final_state['vote_results'])}")
        print(f"Perspectives: {', '.join(final_state['perspectives'])}")

        print("\nFinal Consensus:")
        print("-" * 80)
        print(final_state["final_decision"])
        print("=" * 80)
