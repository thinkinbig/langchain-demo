from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402

if __name__ == "__main__":
    query = "Research the history of Python, Rust, and Go."
    print(f"🚀 Starting Agent on: {query}")

    # Run the graph imported from graph.py
    final_state = app.invoke({
        "task": query,
        "plan": [],
        "results": [],
    })

    print("\n✅ Final Results:")
    for res in final_state["results"]:
        print(f"- {res}")
