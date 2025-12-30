import os
import sys
import re

# Add parent directory to path to import graph
# This assumes the script is located in research-agent/tests/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import from the application graph
from graph import app, get_lead_llm

# 1. Dataset Configuration
DATASET_NAME = "Research Agent - Simple Queries"
DATASET_DESCRIPTION = "Simple fact-finding queries to test end-to-end research capabilities."

# Initial examples to seed the dataset
SIMPLE_EXAMPLES = [
    ("What is LangGraph?", "LangGraph is a library for building stateful, multi-actor applications with LLMs..."),
    ("Who created Python?", "Python was created by Guido van Rossum..."),
    ("What is the capital of France?", "The capital of France is Paris..."),
]

def ensure_dataset(client: Client):
    """Check if dataset exists, if not create it."""
    # Check if dataset exists (by listing) - naive check
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print(f"Dataset '{DATASET_NAME}' already exists.")
        return datasets[0]
    
    print(f"Creating dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION
    )
    
    inputs = [{"query": q} for q, _ in SIMPLE_EXAMPLES]
    outputs = [{"reference_answer": a} for _, a in SIMPLE_EXAMPLES]
    
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id
    )
    return dataset

# 2. Target Definition
def research_agent_target(inputs: dict) -> dict:
    """Wrapper for the LangGraph app."""
    query = inputs["query"]
    # Initialize state
    initial_state = {
        "query": query,
        "research_plan": "",
        "subagent_tasks": [],
        "subagent_findings": [],
        "iteration_count": 0,
        "needs_more_research": False,
        "synthesized_results": "",
        "citations": [],
        "final_report": "",
    }
    
    # Run graph with Budget Control
    # Import locally to avoid circular dependencies if any
    from cost_control import QueryBudget, CostTrackingCallback, CostLimitExceeded
    
    query_budget = QueryBudget()
    cost_callback = CostTrackingCallback(query_budget)
    
    try:
        # Note: We use the compiled app directly. 
        result = app.invoke(
            initial_state,
            config={"callbacks": [cost_callback]}
        )
        
        return {
            "final_report": result.get("final_report", ""),
            "synthesized_results": result.get("synthesized_results", ""),
            "iteration_count": result.get("iteration_count", 0),
            # Optional: Return cost metrics to LangSmith (as custom outputs)
            "tokens_used": query_budget.current_tokens,
            "estimated_cost": (query_budget.current_tokens / 1000) * 0.002
        }
    except CostLimitExceeded as e:
        return {
            "final_report": f"Research stopped due to cost limit: {str(e)}",
            "synthesized_results": "PARTIAL/STOPPED",
            "iteration_count": 0,
            "error": "CostLimitExceeded"
        }

# 3. Evaluators
def completeness_evaluator(run: Run, example: Example) -> dict:
    """LLM-as-judge to check if the answer is complete based on reference."""
    query = example.inputs["query"]
    reference = example.outputs["reference_answer"]
    prediction = run.outputs["synthesized_results"]
    
    # Simple check for empty prediction
    if not prediction:
         return {"key": "completeness", "score": 0.0, "comment": "No result produced."}

    llm = get_lead_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert grader."),
        ("human", """Compare the predicted answer to the reference answer for the given query.
        Query: {query}
        Reference: {reference}
        Prediction: {prediction}
        
        Does the prediction cover the key facts mentioned in the reference? 
        Respond with a score between 0 and 1, where 1 is fully complete relative to the reference.
        Only return the number.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({
            "query": query,
            "reference": reference,
            "prediction": prediction
        })
        # Extract number
        match = re.search(r"(\d+(\.\d+)?)", response)
        score = float(match.group(1)) if match else 0.0
        # Clamp score
        score = min(max(score, 0.0), 1.0)
        return {"key": "completeness", "score": score}
    except Exception as e:
        return {"key": "completeness", "score": 0.0, "comment": str(e)}

def coherence_evaluator(run: Run, example: Example) -> dict:
    """Check if the final report has Markdown headers."""
    report = run.outputs["final_report"]
    if not report:
        return {"key": "coherence", "score": 0.0}
        
    has_headers = "# " in report and "## " in report
    return {"key": "coherence", "score": 1.0 if has_headers else 0.0}

def run_evaluation():
    print("Initializing LangSmith Evaluation...")
    client = Client()
    ensure_dataset(client)
    
    print(f"Running evaluation on '{DATASET_NAME}'...")
    # Using evaluate directly
    results = evaluate(
        research_agent_target,
        data=DATASET_NAME,
        evaluators=[completeness_evaluator, coherence_evaluator],
        experiment_prefix="research-agent-eval",
        max_concurrency=2, # Limit concurrency
    )
    
    # Basic results summary
    print("\nEvaluation Complete!")
    # LangSmith evaluate returns an ExperimentResult object
    print(f"Experiment Name: {results.experiment_name}")
    
    # We can perform simple aggregations if needed, but the UI is best.
    print(f"View results at the URL provided in logs above (usually https://smith.langchain.com/...)")
    return results

def test_evaluation():
    """Run the evaluation as a pytest test."""
    results = run_evaluation()
    assert results is not None
    assert results.experiment_name is not None

if __name__ == "__main__":
    run_evaluation()
