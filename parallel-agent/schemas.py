import operator
from typing import Annotated, List, TypedDict


class SectioningState(TypedDict):
    """State for sectioning (breaking task into parallel subtasks)"""
    task: str  # Original task
    sections: List[str]  # Independent subtasks
    section_results: Annotated[List[str], operator.add]  # Results from each section
    final_summary: str  # Aggregated final result


class VotingState(TypedDict):
    """State for voting (same task run multiple times)"""
    task: str  # Original task
    votes: Annotated[List[str], operator.add]  # Multiple independent evaluations
    consensus: str  # Final consensus or aggregated result

