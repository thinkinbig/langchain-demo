from typing import TypedDict


class OptimizerState(TypedDict):
    task: str
    code: str
    feedback: str
    retry_count: int
