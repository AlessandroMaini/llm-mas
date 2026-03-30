# team_modular.py
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

# ── Tools ──────────────────────────────────────────────────────────────────
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Web: {query}]"


@tool
def scrape_url(url: str) -> str:
    """Scrape content from a URL."""
    return f"[Page: {url}]"


@tool
def run_python(code: str) -> str:
    """Execute Python code."""
    return f"[Exec: {code}]"


@tool
def query_database(sql: str) -> str:
    """Run a SQL query against the database."""
    return f"[DB: {sql}]"


@tool
def generate_chart(spec: str) -> str:
    """Generate a chart from a specification."""
    return f"[Chart: {spec}]"


@tool
def write_section(title: str, body: str) -> str:
    """Write a report section with title and body."""
    return f"[Section '{title}' written]"

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

# ── Team definitions ───────────────────────────────────────────────────────
TEAMS = {
    "research_team": {
        "description": "Handles information gathering and web research.",
        "agents": {
            "web_researcher": make_agent("Web Researcher", "Search the web.", [search_web]),
            "scraper":        make_agent("Scraper", "Scrape web pages.",       [scrape_url]),
        }
    },
    "data_team": {
        "description": "Handles data analysis and visualization.",
        "agents": {
            "analyst":    make_agent("Data Analyst",    "Analyze data with SQL.", [query_database]),
            "visualizer": make_agent("Visualizer", "Create charts.",             [generate_chart]),
            "programmer": make_agent("Programmer", "Write Python scripts.",      [run_python]),
        }
    },
    "writing_team": {
        "description": "Handles drafting, editing, and document production.",
        "agents": {
            "writer": make_agent("Writer", "Write document sections.", [write_section]),
        }
    }
}

# ── Team supervisor: routes within a team ─────────────────────────────────
TEAM_SUPERVISOR_PROMPT = ChatPromptTemplate.from_template(
    "You are the {team_name} supervisor. Route the instruction to ONE "
    "of your agents: {agent_names}.\n"
    "Return JSON: {{\"agent\": \"<name>\", \"instruction\": \"<refined instruction>\"}}\n\n"
    "Instruction: {instruction}"
)
team_supervisor_chain = TEAM_SUPERVISOR_PROMPT | llm | JsonOutputParser()

def run_team(team_name: str, instruction: str) -> str:
    team = TEAMS[team_name]
    agent_names = list(team["agents"].keys())
    decision = team_supervisor_chain.invoke({
        "team_name": team_name,
        "agent_names": ", ".join(agent_names),
        "instruction": instruction,
    })
    agent = team["agents"].get(decision["agent"], list(team["agents"].values())[0])
    return agent.invoke({"input": decision["instruction"]})["output"]

# ── Top-level coordinator: routes between teams ────────────────────────────
COORDINATOR_PROMPT = ChatPromptTemplate.from_template(
    "You are the project coordinator. Break the task into steps, each assigned to one team.\n"
    "Teams: research_team, data_team, writing_team.\n"
    "Return JSON: {{\"plan\": [{{\"team\": \"...\", \"instruction\": \"...\"}}]}}\n\n"
    "Task: {task}"
)
coordinator_chain = COORDINATOR_PROMPT | llm | JsonOutputParser()

class CoordState(TypedDict):
    task: str
    plan: list[dict]
    current: int
    log: str

def coordinator_node(state: CoordState) -> CoordState:
    result = coordinator_chain.invoke({"task": state["task"]})
    return {**state, "plan": result["plan"], "current": 0}

def team_dispatch_node(state: CoordState) -> CoordState:
    step   = state["plan"][state["current"]]
    output = run_team(step["team"], step["instruction"])
    log    = state["log"] + f"\n[{step['team']}] → {output}"
    return {**state, "log": log, "current": state["current"] + 1}

def more_steps(state: CoordState) -> str:
    return "dispatch" if state["current"] < len(state["plan"]) else END

graph = StateGraph(CoordState)
graph.add_node("coordinator", coordinator_node)
graph.add_node("dispatch",    team_dispatch_node)
graph.set_entry_point("coordinator")
graph.add_edge("coordinator", "dispatch")
graph.add_conditional_edges("dispatch", more_steps)
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "task": "Research EV market trends, analyze sales data, and write a report with charts.",
        "plan": [], "current": 0, "log": ""
    })
    print(result["log"])