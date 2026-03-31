"""Function tools for the OCF Agent Workflow."""

from datetime import datetime
from zoneinfo import ZoneInfo

from llama_index.core import VectorStoreIndex

from tools.docs_search import create_docs_search_tool
from tools.python_run import create_python_run_tool
from tools.user_info import create_user_info_tool
from tools.web_search import create_web_search_tool
from tools.web_scrape import create_web_scrape_tool
from tools.delegate import create_delegation_tool


def get_tool_prompt(user_name: str = "unknown", use_thinking: bool = False) -> str:
    """Generate the prompt for deciding which tools to use.

    Args:
        question: The user's question.
        use_thinking: Whether the agent is in thinking mode.

    Returns:
        The formatted prompt string for tool decision.
    """
    prompt = (
        "You are a helpful assistant in a multi-user Discord thread. "
        "User messages are formatted as `[Username @ Timestamp]: Content`. "
        "Pay close attention to who is speaking and address them by name if necessary. "
        "Multiple different users may be participating in this same thread.\n\n"
        "You must decide what information to search for or what actions to take to answer the user's question.\n"
        "- Call 'search_web' for general internet facts, current events, and news. You can use the 'page' parameter to see more results.\n"
        "- Call 'scrape_url' to read the content of a specific webpage if you are given a URL or need to extract text from an external link.\n"
        "- Call 'delegate_task' for complex, multi-step research, processing large amounts of text, or exploring many links at once. This runs a specialized sub-agent to handle the heavy lifting.\n"
        "- Call 'search_docs' for internal OCF rules, services, policies, or any OCF-related question.\n"
        "- Call 'get_ocf_user_info' if the user asks for details or printing quotas for a specific OCF username.\n"
        "- Call 'run_python' if you need to perform complex math calculations, manipulate data, or run custom logic.\n"
        "You should call as many tools as necessary to fully address the user's question.\n"
        "Unless it is definitely unrelated, ALWAYS call 'search_docs' to check internal documentation first.")

    if use_thinking:
        prompt += ("\n\nCRITICAL: You are in 'Thinking Mode'. Before making ANY tool calls, you MUST use your thinking space to:\n"
                   "1. Analyze the user's request carefully.\n"
                   "2. Outline a step-by-step 'Research Plan' explaining which tools you will use and why.\n"
                   "3. If a tool call reveals new information (like a URL or a specific service), stop and re-evaluate your plan in the next thinking step before proceeding.\n"
                   "4. Only provide a final answer once you have gathered all necessary context or exhausted your options.")

    return prompt


def get_all_tools(index: VectorStoreIndex, depth: int = 0) -> dict:
    """Central registry of all tools available to the workflow.

    Args:
        index: The document index.
        depth: The current recursion depth of the agent (used to prevent infinite delegation).
    """
    from llm import get_llm
    llm_standard = get_llm(thinking=False)
    llm_thinking = get_llm(thinking=True)

    tools = {
        "search_web": create_web_search_tool(),
        "scrape_url": create_web_scrape_tool(),
        "search_docs": create_docs_search_tool(index),
        "get_ocf_user_info": create_user_info_tool(),
        "run_python": create_python_run_tool(),
    }

    # Only add delegation tool if we haven't reached max depth
    if depth < 2:
        tools["delegate_task"] = create_delegation_tool(
            llm_standard, llm_thinking, index, current_depth=depth
        )

    return tools
