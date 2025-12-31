"""
Script to manage prompts via LangSmith Hub.
Allows pushing local prompts to the Hub and demonstration of pulling them.

Usage:
    python research-agent/scripts/manage_prompts.py
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain import hub
from prompts import (
    LEAD_RESEARCHER_INITIAL,
    LEAD_RESEARCHER_REFINE,
    SUBAGENT_ANALYSIS,
    SYNTHESIZER_MAIN,
    VERIFIER_MAIN
)

load_dotenv()

def push_prompts():
    """Push local prompts to LangChain Hub."""
    print("🚀 Pushing prompts to LangChain Hub...")
    
    # Mapping of local variable name to Hub repo name
    # Format: "handle/repo-name"
    # We'll use a placeholder handle 'deep-research' or rely on the user's default handle if not specified.
    # Note: Using 'push' without a specific handle often pushes to your own namespace.
    
    prompts_to_push = {
        "lead-researcher-initial": LEAD_RESEARCHER_INITIAL,
        "lead-researcher-refine": LEAD_RESEARCHER_REFINE,
        "subagent-analysis": SUBAGENT_ANALYSIS,
        "synthesizer-main": SYNTHESIZER_MAIN,
        "verifier-main": VERIFIER_MAIN,
    }

    for name, prompt_obj in prompts_to_push.items():
        try:
            # Pushing to the user's handle with specific repo name
            # We assume the user has a handle. If not, this might need an explicit handle.
            # hub.push(f"{handle}/{name}", prompt_obj)
            
            # Simple push (interactive or default)
            # Usually creates/updates <user-handle>/<repo-name>
            target_repo = name  # e.g., "lead-researcher-initial"
            
            print(f"  - Pushing '{name}'...")
            repo_url = hub.push(target_repo, prompt_obj)
            print(f"    ✅ Pushed to: {repo_url}")
            
        except Exception as e:
            print(f"    ❌ Failed to push '{name}': {e}")
            print("       (Ensure you are logged in via `langchainhub login` or have LANGCHAIN_API_KEY set)")

def check_pull_demo():
    """Demonstrate how to pull a prompt."""
    print("\n⬇️  Demonstrating Pull (Verification)...")
    try:
        # Pulling back the first one as a test
        repo_name = "lead-researcher-initial" # Assuming default handle
        # NOTE: In real usage, you'd use the full handle returned by push, e.g., "user123/lead-researcher-initial"
        # Since we don't know the handle, we'll try pulling the one we just (hopefully) pressed.
        
        # This is strictly illustrative if we don't capture the handle dynamically.
        pass 
    except Exception as e:
        pass

if __name__ == "__main__":
    if "LANGCHAIN_API_KEY" not in os.environ:
        print("❌ LANGCHAIN_API_KEY is not set. Please set it in .env or environment.")
        sys.exit(1)
        
    push_prompts()
    print("\n✨ Done! You can now edit these prompts in the LangSmith UI.")
    print("   To use them in code, replace the local constant with:")
    print('   prompt = hub.pull("<your-handle>/lead-researcher-initial")')
