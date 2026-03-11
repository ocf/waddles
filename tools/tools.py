"""Function tools for the OCF Agent Workflow."""

from llama_index.core import VectorStoreIndex

from tools.docs_search import create_docs_search_tool
from tools.python_run import create_python_run_tool
from tools.user_info import create_user_info_tool
from tools.web_search import create_web_search_tool


def get_tool_prompt(question: str) -> str:
    """Generate the prompt for deciding which tools to use.

    Args:
        question: The user's question.

    Returns:
        The formatted prompt string for tool decision.
    """
    return (
        "You must decide what information to search for or what actions to take to answer the user's question. "
        "Call 'search_web' for general internet facts, current events, and news. "
        "Call 'search_docs' for internal OCF rules, services, policies, or any OCF-related question. "
        "Call 'get_ocf_user_info' if the user asks for details or printing quotas for a specific OCF username. "
        "Call 'run_python' if you need to perform complex math calculations, manipulate data, or run custom logic. "
        "You should call as many tools as necessary to fully address the user's question. "
        "Unless it is definitely unrelated, ALWAYS call 'search_docs' to check internal documentation first. "
        f"\n\nUser Question: {question}"
    )


def get_all_tools(index: VectorStoreIndex) -> dict:
    """Central registry of all tools available to the workflow."""
    return {
        "search_web": create_web_search_tool(),
        "search_docs": create_docs_search_tool(index),
        "get_ocf_user_info": create_user_info_tool(),
        "run_python": create_python_run_tool(),
    }
