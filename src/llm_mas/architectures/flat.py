# flat.py
from dotenv import load_dotenv
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.openai_functions_agent.base import (
    create_openai_functions_agent,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# --- Define peer agents ---
def make_agent(name: str, description: str, tools: list) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are {name}. {description}"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Web results for: {query}]"

@tool
def run_python(code: str) -> str:
    """Execute Python code."""
    return f"[Output of code: {code}]"

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    return f"[Written to {filename}]"

researcher  = make_agent("Researcher",  "You find and summarize information.", [search_web])
coder       = make_agent("Coder",       "You write and execute Python code.",   [run_python])
writer      = make_agent("Writer",      "You produce well-written documents.",  [write_file])

AGENTS = {
    "researcher": researcher,
    "coder":      coder,
    "writer":     writer,
}

# --- Flat router: LLM picks the right peer ---
ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Given the task below, choose one agent: researcher, coder, writer.\n"
    "Reply with ONLY the agent name.\n\nTask: {task}"
)
router_chain = ROUTER_PROMPT | llm | StrOutputParser()

def run_flat_mas(task: str) -> str:
    agent_name = router_chain.invoke({"task": task}).strip().lower()
    agent = AGENTS.get(agent_name, researcher)
    print(f"[Router] → {agent_name}")
    result = agent.invoke({"input": task})
    return result["output"]

if __name__ == "__main__":
    print(run_flat_mas("Find the latest Python 3.13 features and write a summary."))