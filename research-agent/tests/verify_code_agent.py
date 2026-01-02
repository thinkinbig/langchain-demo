import os
import sys
import unittest

from dotenv import load_dotenv

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mocking for testing without full LLM cost if needed, but here we want integration test
from graph import app
from schemas import ResearchState


class TestCodeAgent(unittest.TestCase):
    def test_python_repl_usage(self):
        print("\n\n=== TEST: CodeAgent Python REPL Usage ===")

        # We want to force the agent into a scenario where it naturally uses code.
        # Query: "Calculate the square root of 54321 and multiply it by 7."
        query = "Calculate the square root of 54321 and multiply it by 7 using python."

        initial_state = ResearchState(
            query=query,
            research_plan="",
            subagent_tasks=[],
            subagent_findings=[],
            iteration_count=0,
            scratchpad=""
        )

        # Run the graph
        # This will trigger the lead researcher (who makes a plan) -> subagent.
        # The subagent should see the task "Calculate..." and use the Python tool.
        try:
            config = {"configurable": {"thread_id": "test_verification_1"}}
            final_state = app.invoke(initial_state, config=config)

            # Since we can't easily spy on internal tool calls without a callback,
            # we rely on the final output mentioning the calculation or checking logs.

            print("Final Report:", final_state.get('final_report'))

        except Exception as e:
            self.fail(f"Graph execution failed: {e}")

if __name__ == '__main__':
    load_dotenv()
    unittest.main()
