import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


# --- Tools ---
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Web: {query}]"


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return f"[Math: {expression}]"


# --- State ---
class PeerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# --- Peer Nodes ---
def create_peer_node(agent_name: str, system_prompt: str, tools: list):
    # We compile a standard tool-calling agent
    agent = create_agent(llm, tools, system_prompt=system_prompt)

    def node(state: PeerState):
        result = agent.invoke({"messages": state["messages"]})
        final_message = result["messages"][-1]
        # Wrap it to preserve sender info
        msg = AIMessage(content=f"[{agent_name}]: {final_message.content}", name=agent_name)
        return {"messages": [msg], "sender": agent_name}

    return node


researcher = create_peer_node(
    "Researcher",
    "You are a Researcher. Contribute facts. End your turn by asking the Critic for thoughts. If the task is fully solved, output 'TERMINATE'.",
    [search_web],
)
critic = create_peer_node(
    "Critic",
    "You are a Critic. Analyze the Researcher's facts for flaws or logic gaps. End your turn by passing back to the Researcher. If the task is fully solved, output 'TERMINATE'.",
    [calculate],
)


# --- Routing ---
def peer_router(state: PeerState) -> str:
    last_message = state["messages"][-1].content
    if "TERMINATE" in last_message:
        return END
    # Simple turn-based routing (could be expanded to LLM-based mesh routing)
    if state["sender"] == "Researcher":
        return "critic"
    else:
        return "researcher"


# --- Graph ---
workflow = StateGraph(PeerState)
workflow.add_node("researcher", researcher)
workflow.add_node("critic", critic)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges("researcher", peer_router)
workflow.add_conditional_edges("critic", peer_router)

app = workflow.compile()

# Run Demo
result = app.invoke(
    {
        "messages": [
            HumanMessage(content="Brainstorm and evaluate the feasibility of a space elevator.")
        ],
        "sender": "user",
    }
)
