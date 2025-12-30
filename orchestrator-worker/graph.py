
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from schemas import AgentState, Plan

# 1. Setup LLM (Reads OPENAI_API_KEY from .env automatically)
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0
)

# 2. Define Nodes
def orchestrator_node(state: AgentState):
    task = state["task"]

    planner = llm.with_structured_output(Plan)
    plan = planner.invoke([
        SystemMessage(
            content="Break this task into 3 distinct, parallelizeable sub-tasks."
        ),
        HumanMessage(content=task),
    ])

    return {"plan": plan["steps"]}

def worker_node(state: AgentState):
    step = state["plan"][0]

    response = llm.invoke([
        SystemMessage(content="You are a precise worker. Execute the task concisely."),
        HumanMessage(content=step),
    ])

    return {
        "results": [
            f"Finised {step}: {response.content}"
        ]
    }

def assign_workers(state: AgentState):
    return [Send("worker", {"plan": [step]}) for step in state["plan"]]

# 3. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("worker", worker_node)
workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", assign_workers, ["worker"])
workflow.add_edge("worker", END)

# 4. Export the compiled app
app = workflow.compile()
