from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402

if __name__ == "__main__":
    # Complex requests that require multi-stage processing
    requests = [
        (
            "Generate a comprehensive market analysis report for AI technology "
            "trends in 2024, including market size, key players, and growth "
            "projections"
        ),
        (
            "Analyze the performance metrics: sales increased 15%, customer "
            "satisfaction is 4.5/5, and calculate the ROI if we invest $100k "
            "in marketing"
        ),
        (
            "Process this text: 'The quarterly revenue was $2.5M, up from "
            "$1.8M last quarter. Customer churn rate decreased by 12%. "
            "Calculate the growth rate and provide insights.'"
        ),
    ]

    for i, request in enumerate(requests, 1):
        print(f"\n{'='*80}")
        print(f"Request {i}: {request[:70]}...")
        print("="*80)

        # Initialize state
        initial_state = {
            "request": request,
            "stage": "input_parser",
            "extracted_data": {},
            "validation_results": [],
            "processing_log": [],
            "decisions": [],
            "final_output": "",
            "error_count": 0,
            "retry_stage": "",
        }

        # Run agent
        final_state = app.invoke(initial_state)

        # Display results
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        print(f"Stages executed: {len(final_state['processing_log'])}")
        print(f"Decisions made: {len(final_state['decisions'])}")
        print(f"Validation checks: {len(final_state['validation_results'])}")
        print(f"Errors encountered: {final_state['error_count']}")
        print("\nProcessing Log:")
        for log in final_state["processing_log"]:
            print(f"  {log}")
        print("\nDecisions:")
        for decision in final_state["decisions"]:
            print(f"  {decision}")
        print("\n" + "="*80)
        print("FINAL OUTPUT")
        print("="*80)
        print(final_state["final_output"])
        print("="*80 + "\n")

