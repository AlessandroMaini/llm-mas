from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# --- State ---
class HierarchicalState(TypedDict):
    task: str
    worker_outputs: list[str]
    status: Literal["delegating", "synthesizing", "done"]
    final_output: str


# --- Structured Output for Orchestrator ---
class OrchestratorDecision(BaseModel):
    next_action: Literal["call_researcher", "call_coder", "synthesize"]
    instruction: str


orchestrator_llm = llm.with_structured_output(OrchestratorDecision)


# --- Nodes ---
def orchestrator_node(state: HierarchicalState):
    history = "\n".join(state["worker_outputs"])
    prompt = f"Task: {state['task']}\nHistory:\n{history}\nDecide the next step."

    decision = orchestrator_llm.invoke([{"role": "user", "content": prompt}])

    if decision.next_action == "synthesize":
        return {"status": "synthesizing"}
    else:
        # Route to a worker and pass the specific instruction
        return {
            "status": decision.next_action,
            "task": decision.instruction,
        }  # Overloading task temporarily for the worker


def worker_node(state: HierarchicalState, role: str):
    # Mock worker execution (replace with create_agent)
    instruction = state["task"]
    response = llm.invoke(
        [{"role": "system", "content": f"You are a {role}. Execute: {instruction}"}]
    )
    return {
        "worker_outputs": state["worker_outputs"] + [f"[{role}]: {response.content}"],
        "status": "delegating",
    }


def synthesize_node(state: HierarchicalState):
    history = "\n".join(state["worker_outputs"])
    prompt = f"Synthesize this data to answer: {state['task']}\nData: {history}"
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"final_output": response.content, "status": "done"}


# --- Graph ---
workflow = StateGraph(HierarchicalState)
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("researcher", lambda state: worker_node(state, "Researcher"))
workflow.add_node("coder", lambda state: worker_node(state, "Coder"))
workflow.add_node("synthesizer", synthesize_node)

workflow.set_entry_point("orchestrator")


def router(state: HierarchicalState):
    if state["status"] == "call_researcher":
        return "researcher"
    if state["status"] == "call_coder":
        return "coder"
    if state["status"] == "synthesizing":
        return "synthesizer"
    if state["status"] == "done":
        return END
    return "orchestrator"


workflow.add_conditional_edges("orchestrator", router)
workflow.add_edge("researcher", "orchestrator")  # Star topology: always return to orchestrator
workflow.add_edge("coder", "orchestrator")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# Run Demo
result = app.invoke(
    {
        "task": "What is the capital of France?",
        "worker_outputs": [],
        "status": "delegating",
        "final_output": "",
    }
)
