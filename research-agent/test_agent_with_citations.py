#!/usr/bin/env python
"""
Test the research agent with citation extraction enabled.
This tests the problematic query from the LangSmith trace.
"""

import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from graph import app
from schemas import ResearchState


def test_citation_extraction():
    """Test that agent extracts and follows citations from PDF"""

    query = "What is the difference between Cognitive Planning and State Consolidation?"

    print("="*80)
    print("TESTING RESEARCH AGENT WITH CITATION EXTRACTION")
    print("="*80)
    print(f"\nQuery: {query}\n")
    print("Expected behavior:")
    print("  1. Retrieve content from PDF containing citations")
    print("  2. Extract citations (Song et al., Zhang et al., etc.)")
    print("  3. Generate follow-up tasks to investigate those papers")
    print("  4. Perform deeper research across 2-3 iterations")
    print("\n" + "="*80 + "\n")

    # Initialize state
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

    # Run with unique thread ID
    thread_id = "test_citation_extraction_v1"
    config = {"configurable": {"thread_id": thread_id}}

    final_state = app.invoke(initial_state, config=config)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\nFinal Report Length: {len(final_state.get('final_report', ''))} chars")
    print(f"Iterations: {final_state.get('iteration_count', 0)}")
    print(f"Citations Found: {len(final_state.get('all_extracted_citations', []))}")

    # Show extracted citations
    citations = final_state.get('all_extracted_citations', [])
    if citations:
        print(f"\n📚 Extracted Citations ({len(citations)}):")
        for i, cite in enumerate(citations[:5], 1):
            print(f"  {i}. {cite.get('title', 'Unknown')}")

    print("\n✅ Test completed! Check LangSmith trace for details.")
    return final_state

if __name__ == "__main__":
    test_citation_extraction()
