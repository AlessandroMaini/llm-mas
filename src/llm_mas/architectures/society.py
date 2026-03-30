# society.py
import uuid
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# ── Shared blackboard ──────────────────────────────────────────────────────
class Message(TypedDict):
    id: str
    sender: str
    content: str
    msg_type: str          # "task" | "result" | "question" | "answer"
    addressed_to: str | None
    timestamp: str

class SocietyState(TypedDict):
    goal: str
    blackboard: list[Message]
    iteration: int
    max_iterations: int
    final_answer: str

def post_message(board: list[Message], sender: str, content: str,
                 msg_type: str, addressed_to: str = "all") -> list[Message]:
    return board + [{
        "id":           str(uuid.uuid4())[:8],
        "sender":       sender,
        "content":      content,
        "msg_type":     msg_type,
        "addressed_to": addressed_to,
        "timestamp":    datetime.now().isoformat(),
    }]

def board_summary(board: list[Message]) -> str:
    return "\n".join(
        f"[{m['sender']} → {m['addressed_to']} | {m['msg_type']}]: {m['content']}"
        for m in board[-10:]   # last 10 messages for context window efficiency
    )

# ── Agent personas ─────────────────────────────────────────────────────────
PERSONAS = {
    "Socrates": (
        "You are Socrates — a critical thinker who asks clarifying questions, "
        "challenges assumptions, and exposes contradictions in reasoning."
    ),
    "Darwin": (
        "You are Darwin — an empirical scientist who looks for evidence, "
        "patterns, and causal mechanisms."
    ),
    "Turing": (
        "You are Turing — a computational thinker who models problems formally, "
        "designs algorithms, and evaluates feasibility."
    ),
    "Curie": (
        "You are Curie — a rigorous experimenter who proposes concrete experiments "
        "and evaluates hypotheses against data."
    ),
}

AGENT_PROMPT = ChatPromptTemplate.from_template(
    "{persona}\n\n"
    "The society's shared goal: {goal}\n\n"
    "Recent blackboard messages:\n{board}\n\n"
    "Decide what to contribute. You may:\n"
    "  - Post a 'result' with your insight\n"
    "  - Post a 'question' addressed to a specific agent or 'all'\n"
    "  - Post an 'answer' to an open question\n"
    "  - Post 'done' as msg_type if the goal is fully solved\n\n"
    "Return JSON:\n"
    "{{\"msg_type\": \"...\", \"addressed_to\": \"...\", \"content\": \"...\"}}"
)
agent_chain = AGENT_PROMPT | llm | JsonOutputParser()

# ── Termination judge ──────────────────────────────────────────────────────
JUDGE_PROMPT = ChatPromptTemplate.from_template(
    "Given the goal and the discussion below, has the society reached a sufficient answer?\n"
    "Goal: {goal}\nDiscussion:\n{board}\n\n"
    "Return JSON: {{\"solved\": true/false, \"summary\": \"...\"}}"
)
judge_chain = JUDGE_PROMPT | llm | JsonOutputParser()

# ── LangGraph nodes ────────────────────────────────────────────────────────
def society_round(state: SocietyState) -> SocietyState:
    """Each agent reads the board and contributes once per round."""
    board = state["blackboard"]
    for name, persona in PERSONAS.items():
        response = agent_chain.invoke({
            "persona":  persona,
            "goal":     state["goal"],
            "board":    board_summary(board),
        })
        board = post_message(
            board,
            sender=name,
            content=response.get("content", ""),
            msg_type=response.get("msg_type", "result"),
            addressed_to=response.get("addressed_to", "all"),
        )
    return {**state, "blackboard": board, "iteration": state["iteration"] + 1}

def judge_node(state: SocietyState) -> SocietyState:
    verdict = judge_chain.invoke({
        "goal":  state["goal"],
        "board": board_summary(state["blackboard"]),
    })
    final = verdict.get("summary", "") if verdict.get("solved") else ""
    return {**state, "final_answer": final}

def should_stop(state: SocietyState) -> str:
    if state["final_answer"]:
        return END
    if state["iteration"] >= state["max_iterations"]:
        return END
    return "society_round"

graph = StateGraph(SocietyState)
graph.add_node("society_round", society_round)
graph.add_node("judge",         judge_node)
graph.set_entry_point("society_round")
graph.add_edge("society_round", "judge")
graph.add_conditional_edges("judge", should_stop)
app = graph.compile()

if __name__ == "__main__":
    init_board = post_message([], "System", "Debate begins now.", "task")
    result = app.invoke({
        "goal": (
            "Can artificial general intelligence be achieved with "
            "current transformer architectures?"
        ),
        "blackboard":     init_board,
        "iteration":      0,
        "max_iterations": 4,
        "final_answer":   "",
    })
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"] or "Max iterations reached.")
    print("\n=== FULL BLACKBOARD ===")
    for m in result["blackboard"]:
        print(f"[{m['sender']} → {m['addressed_to']}] {m['content']}")