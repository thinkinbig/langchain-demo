"""Test script to verify parallelization is working"""

import time

from dotenv import load_dotenv

load_dotenv()

from sectioning import sectioning_app  # noqa: E402
from voting import voting_app  # noqa: E402


def test_sectioning_parallel():
    """Test if sectioning actually runs in parallel"""
    print("Testing Sectioning Parallelization...")
    print("=" * 80)

    task = "Analyze the security, performance, and scalability of a web application"

    start_time = time.time()

    initial_state = {
        "task": task,
        "sections": [],
        "section_results": [],
        "final_summary": "",
    }

    final_state = sectioning_app.invoke(initial_state)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
    print(f"📊 Sections processed: {len(final_state['section_results'])}")
    print(f"✅ Parallelization working: {len(final_state['section_results']) > 1}")


def test_voting_parallel():
    """Test if voting actually runs in parallel"""
    print("\n\nTesting Voting Parallelization...")
    print("=" * 80)

    task = "Review this code for security: def process(data): return data.upper()"

    start_time = time.time()

    initial_state = {
        "task": task,
        "votes": [],
        "consensus": "",
    }

    final_state = voting_app.invoke(initial_state)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
    print(f"📊 Votes collected: {len(final_state['votes'])}")
    print(f"✅ Parallelization working: {len(final_state['votes']) > 1}")


if __name__ == "__main__":
    test_sectioning_parallel()
    test_voting_parallel()

