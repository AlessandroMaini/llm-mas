# hierarchical.py
from typing import TypedDict

from dotenv import load_dotenv
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.openai_functions_agent.base import (
    create_openai_functions_agent,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# --- Sub-agents (same as flat, abbreviated) ---
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Results: {query}]"


@tool
def run_python(code: str) -> str:
    """Execute Python code."""
    return f"[Output: {code}]"


@tool
def write_file(name: str, text: str) -> str:
    """Write text to a file."""
    return f"[Saved: {name}]"

def make_agent(name, desc, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are {name}. {desc}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    return AgentExecutor(
        agent=create_openai_functions_agent(llm, tools, prompt),
        tools=tools, verbose=True
    )

researcher = make_agent("Researcher", "Find information.", [search_web])
coder      = make_agent("Coder",      "Write code.",       [run_python])
writer     = make_agent("Writer",     "Write documents.",  [write_file])

# --- Supervisor: plans a sequential list of steps ---
SUPERVISOR_PROMPT = ChatPromptTemplate.from_template(
    "You are a supervisor. Decompose the task into an ordered list of steps.\n"
    "Each step must assign work to one of: researcher, coder, writer.\n"
    "Return JSON: {{\"steps\": [{{\"agent\": \"...\", \"instruction\": \"...\"}}]}}\n\n"
    "Task: {task}\nCompleted so far:\n{history}"
)
plan_chain = SUPERVISOR_PROMPT | llm | JsonOutputParser()

# --- LangGraph state ---
class MASState(TypedDict):
    task: str
    steps: list
    history: str
    current_step: int
    final_output: str

AGENTS_MAP = {"researcher": researcher, "coder": coder, "writer": writer}

def supervisor_node(state: MASState) -> MASState:
    plan = plan_chain.invoke({"task": state["task"], "history": state["history"]})
    return {**state, "steps": plan["steps"], "current_step": 0}

def worker_node(state: MASState) -> MASState:
    idx   = state["current_step"]
    step  = state["steps"][idx]
    agent = AGENTS_MAP[step["agent"]]
    result = agent.invoke({"input": step["instruction"]})["output"]
    history = state["history"] + f"\n[{step['agent']}]: {result}"
    return {**state, "history": history, "current_step": idx + 1}

def should_continue(state: MASState) -> str:
    return "worker" if state["current_step"] < len(state["steps"]) else END

graph = StateGraph(MASState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("worker",     worker_node)
graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "worker")
graph.add_conditional_edges("worker", should_continue)
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "task": "Research transformer architectures, implement a toy model, then write a report.",
        "steps": [], "history": "", "current_step": 0, "final_output": ""
    })
    print(result["history"])