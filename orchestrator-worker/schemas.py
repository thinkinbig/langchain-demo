import operator
from typing import Annotated, List, TypedDict


# 1. The Schema (What the Orchestrator outputs)
class Plan(TypedDict):
    steps: List[str]

# 2. The Agent State (The Shared Memory)
class AgentState(TypedDict):
    task: str
    plan: List[str]
    results: Annotated[List[str], operator.add]
