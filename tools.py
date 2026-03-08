"""Function tools for the OCF Agent Workflow."""

import asyncio
from typing import TYPE_CHECKING
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS

if TYPE_CHECKING:
    from llama_index.core import VectorStoreIndex


def create_web_search_tool() -> FunctionTool:
    """Web search tool updated for the post-v7 'Simplified' DDGS."""

    def sync_ddgs_call(query: str):
        """The actual blocking call."""
        # Note: deedy5 added mandatory delays, so we keep max_results low
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=3))

    async def search_web(query: str) -> str:
        try:
            # Official workaround: offload the sync call to a worker thread
            results = await asyncio.to_thread(sync_ddgs_call, query)

            if not results:
                return "No web results found."

            formatted = [f"Source: {r['href']}\n{r['body']}" for r in results]
            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            return f"Web search snag (v7+): {e}"

    return FunctionTool.from_defaults(
        async_fn=search_web,
        name="search_web",
        description="Search the internet for recent news and facts not related to OCF. Use for general knowledge, current events, and external information."
    )


def create_docs_search_tool(index: "VectorStoreIndex") -> FunctionTool:
    """Create the internal documentation search tool.

    Args:
        index: The VectorStoreIndex containing OCF documentation.
    """

    async def search_docs(query: str) -> str:
        """
        Search internal OCF documentation for rules, services, and policies.
        Use this for questions about OCF-specific information, lab policies,
        printing, computing resources, and other OCF services.

        Args:
            query: The search query to look up in OCF documentation.

        Returns:
            Relevant documentation passages.
        """
        try:
            retriever = index.as_retriever(similarity_top_k=5)
            nodes = await retriever.aretrieve(query)
            if not nodes:
                return "No internal documentation found."
            return "\n---------------------\n".join([
                n.get_content() for n in nodes
            ])
        except Exception as e:
            return f"Documentation search error: {e}"

    return FunctionTool.from_defaults(
        async_fn=search_docs,
        name="search_docs",
        description="Search internal OCF documentation about rules, services, and policies. Use for OCF-specific questions about lab policies, printing, computing resources, etc."
    )


def get_tool_decision_prompt(question: str) -> str:
    """Generate the prompt for deciding which tools to use.

    Args:
        question: The user's question.

    Returns:
        The formatted prompt string for tool decision.
    """
    return (
        "You must decide what information to search for to answer the user's question. "
        "Call 'search_web' for general internet facts, current events, and news. "
        "Call 'search_docs' for internal OCF rules, services, policies, or any OCF-related question. "
        "You may call multiple tools if needed. "
        "When in doubt, ALWAYS call 'search_docs' to check the internal documentation first. "
        f"\n\nUser Question: {question}"
    )


def get_all_tools(index: "VectorStoreIndex") -> dict:
    """Central registry of all tools available to the workflow."""
    return {
        "search_web": create_web_search_tool(),
        "search_docs": create_docs_search_tool(index),
    }
