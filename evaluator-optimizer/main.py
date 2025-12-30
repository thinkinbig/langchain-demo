from dotenv import load_dotenv

load_dotenv()

from graph import app  # noqa: E402

if __name__ == "__main__":
    # Test task
    task = (
        "Write a Python function to calculate the nth Fibonacci number. "
        "Handle edge cases where n < 0."
    )

    print(f"🚀 Starting Task: {task}")

    final_state = app.invoke({
        "task": task,
        "code": "",
        "feedback": "",
        "retry_count": 0
    })

    print("\n📄 Final Code:")
    print(final_state["code"])
