from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402

if __name__ == "__main__":
    # Diverse tasks that should route to different experts
    tasks = [
        # Code tasks
        "Write a Python function to sort a list of dictionaries by a specific key",
        "Debug this JavaScript code: function add(a,b) { return a - b; }",
        # Analysis tasks
        (
            "Analyze the sales trends from Q1 to Q4: "
            "Q1=$100k, Q2=$120k, Q3=$135k, Q4=$150k"
        ),
        "Research the pros and cons of microservices architecture",
        # Writing tasks
        "Write a professional email to request a meeting with a client",
        "Create a blog post introduction about the benefits of remote work",
        # Calculation tasks
        "Calculate the compound interest on $10,000 at 5% annual rate for 10 years",
        "Solve: If a train travels 300km in 2 hours, what's its average speed?",
        # General tasks
        "Explain how photosynthesis works in simple terms",
        "What are the key differences between REST and GraphQL APIs?",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"Task {i}: {task}")
        print("="*80)

        # Initialize state
        initial_state = {
            "task": task,
            "task_type": "",
            "routed_to": "",
            "expert_output": "",
            "routing_reason": "",
        }

        # Run agent
        final_state = app.invoke(initial_state)

        # Display results
        print("\n" + "="*80)
        print("ROUTING SUMMARY")
        print("="*80)
        print(f"Task Type: {final_state['task_type']}")
        print(f"Routed To: {final_state['routed_to']}")
        print(f"Routing Reason: {final_state['routing_reason']}")
        print("\n" + "="*80)
        print("FINAL OUTPUT")
        print("="*80)
        print(final_state["expert_output"])
        print("="*80 + "\n")

