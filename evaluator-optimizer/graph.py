from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from schemas import OptimizerState

llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0
)


# 4. Generator Node (Writes the code)
def generator_node(state: OptimizerState):
    task = state["task"]
    feedback = state.get("feedback", "")
    current_code = state.get("code", "")

    if feedback:
        attempt = state['retry_count'] + 1
        print(
            f"\n🔄 (Generator) Feedback received. Refining code... "
            f"(Attempt {attempt})"
        )
        prompt = f"""
        You are an expert Python developer.
        Original Task: {task}
        Your Previous Code:
        {current_code}

        Feedback from Reviewer (Must Fix): {feedback}

        Please generate the fully corrected Python code.
        Output ONLY the code, no markdown explanations.
        """
    else:
        print("\n✍️ (Generator) Writing initial draft...")
        prompt = (
            f"You are an expert Python developer. "
            f"Write a Python script to accomplish this task: {task}. "
            f"Output ONLY the code."
        )

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "code": response.content,
        "feedback": "",
        "retry_count": state["retry_count"] + 1 if feedback else 0
    }

# 5. Evaluator Node (Reviews the code)
def evaluator_node(state: OptimizerState):
    code = state["code"]
    task = state["task"]

    print("\n🧐 (Evaluator) Reviewing code...")

    prompt = f"""
    You are a strict QA Engineer. Review the following Python code for
    correctness, bugs, and completeness.

    Task: {task}
    Code to Review:
    {code}

    If the code is perfect and solves the task, reply exactly with "PASS".
    If there are errors or missing requirements, briefly explain what needs to be fixed.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"feedback": response.content}

# 6. Router Logic
def should_continue(state: OptimizerState):
    feedback = state["feedback"]
    retry_count = state["retry_count"]

    # Check for "PASS" (Case insensitive is safer)
    if "PASS" in feedback.upper():
        print("\n✅ Code accepted! Task complete.")
        return END

    if retry_count >= 3:
        print("\n⚠️ Maximum retries reached. Stopping.")
        return END

    print(f"\n❌ Review failed: {feedback}")
    return "generator"


workflow = StateGraph(OptimizerState)

workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "evaluator")
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {END: END, "generator": "generator"}
)

app = workflow.compile()
