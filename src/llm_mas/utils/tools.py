# src/llm_mas/utils/tools.py
from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"[Web results for: {query}]"

@tool
def run_python(code: str) -> str:
    """Execute Python code and return output."""
    return f"[Code output for: {code}]"

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    return f"[Written to {filename}]"

@tool
def query_database(sql: str) -> str:
    """Run a SQL query against the database."""
    return f"[DB result for: {sql}]"

@tool
def generate_chart(spec: str) -> str:
    """Generate a chart from a specification."""
    return f"[Chart generated: {spec}]"