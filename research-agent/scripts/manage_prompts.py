
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langsmith import Client
from prompts import (
    LEAD_RESEARCHER_INITIAL,
    LEAD_RESEARCHER_REFINE,
    SUBAGENT_STRUCTURED_ANALYSIS,
    SYNTHESIZER_MAIN,
    VERIFIER_MAIN,
)

load_dotenv()

def push_prompts():
    """Push local prompts to LangChain Hub via LangSmith SDK."""
    print("🚀 Pushing prompts to LangChain Hub...")

    client = Client()

    prompts_to_push = {
        "lead-researcher-initial": LEAD_RESEARCHER_INITIAL,
        "lead-researcher-refine": LEAD_RESEARCHER_REFINE,
        "subagent-analysis": SUBAGENT_STRUCTURED_ANALYSIS,
        "synthesizer-main": SYNTHESIZER_MAIN,
        "verifier-main": VERIFIER_MAIN,
    }

    for name, prompt_obj in prompts_to_push.items():
        try:
            target_repo = name

            print(f"  - Pushing '{name}'...")
            repo_url = client.push_prompt(target_repo, object=prompt_obj)
            print(f"    ✅ Pushed to: {repo_url}")

        except Exception as e:
            print(f"    ❌ Failed to push '{name}': {e}")
            print("       (Ensure LANGCHAIN_API_KEY is set correctly)")

if __name__ == "__main__":
    if "LANGCHAIN_API_KEY" not in os.environ:
        print("❌ LANGCHAIN_API_KEY is not set. Please set it in .env or environment.")
        sys.exit(1)

    push_prompts()
    print("\n✨ Done! You can now edit these prompts in the LangSmith UI.")
    print(
        "   To use them in code (once `from langchain import hub` is fixed "
        "or using langsmith):"
    )
    print('   prompt = client.pull_prompt("<your-handle>/lead-researcher-initial")')

