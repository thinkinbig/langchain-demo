import operator
from typing import Annotated, List, TypedDict


class DecisionState(TypedDict):
    """State for multi-stage decision agent"""
    request: str  # Original user request
    stage: str  # Current processing stage
    extracted_data: dict  # Extracted structured data
    validation_results: Annotated[List[str], operator.add]  # Validation results
    processing_log: Annotated[List[str], operator.add]  # Processing log
    decisions: Annotated[List[str], operator.add]  # Decision history
    final_output: str  # Final processed output
    error_count: int  # Error counter
    retry_stage: str  # Stage to retry if errors occur

