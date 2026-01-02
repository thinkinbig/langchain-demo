import os

from langsmith import Client


def create_dataset():
    """Create a LangSmith dataset for evaluating context engineering."""
    client = Client()
    dataset_name = "Research Agent Context Benchmark"

    # Check if dataset exists
    ds_iterator = client.list_datasets(dataset_name=dataset_name)
    existing_ds = next(ds_iterator, None)

    if existing_ds:
        print(f"Dataset '{dataset_name}' already exists. ID: {existing_ds.id}")
        return existing_ds

    # Create dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Benchmark for evaluating context recall and isolation efficiency."
    )
    print(f"Created dataset '{dataset_name}'. ID: {dataset.id}")

    # Add examples
    examples = [
        {
            "inputs": {"query": "What is the budget for Project Alpha?"},
            "outputs": {"answer": "The budget for Project Alpha is $2.5 million."},
            "metadata": {"type": "retrieval", "difficulty": "easy"}
        },
        {
            "inputs": {
                "query": (
                    "Calculate the risk exposure if Project Alpha is "
                    "delayed by 3 months."
                )
            },
            "outputs": {
                "answer": "Dependent on specific risk factors in internal docs."
            },
            "metadata": {"type": "reasoning", "difficulty": "medium"}
        },
        {
            "inputs": {
                "query": (
                    "Compare Project Alpha budget with standard "
                    "operating costs."
                )
            },
            "outputs": {
                "answer": (
                    "Project Alpha budget is $2.5M. Standard costs needed "
                    "from other docs."
                )
            },
            "metadata": {"type": "integration", "difficulty": "hard"}
        }
    ]

    for example in examples:
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata=example["metadata"],
            dataset_id=dataset.id
        )

    print(f"Added {len(examples)} examples to dataset.")
    return dataset

if __name__ == "__main__":
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("Skipping dataset creation: LANGCHAIN_API_KEY not set.")
    else:
        try:
            create_dataset()
        except Exception as e:
            print(f"Failed to create dataset: {e}")
