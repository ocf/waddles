import asyncio
from ddgs import DDGS
from llama_index.core.tools import FunctionTool

from config import WEB_SEARCH_MAX_RESULTS


def create_web_search_tool() -> FunctionTool:
    """Web search tool with pagination."""

    def sync_ddgs_call(query: str, page: int = 1):
        """The actual blocking call with simulated pagination."""
        start_index = (page - 1) * WEB_SEARCH_MAX_RESULTS
        end_index = page * WEB_SEARCH_MAX_RESULTS

        with DDGS() as ddgs:
            # We fetch up to the end of the requested page and slice the results.
            # DDGS doesn't have a direct skip/offset, so we fetch the total count needed.
            results = list(ddgs.text(query, max_results=end_index))
            return results[start_index:end_index]

    async def search_web(query: str, page: int = 1) -> str:
        """
        Search the internet for news and facts.

        Args:
            query: The search terms.
            page: The page number of results (each page contains 3 results). 
                  Increase this if you didn't find what you need on the first page.
        """
        try:
            if page < 1:
                page = 1

            # Official workaround: offload the sync call to a worker thread
            results = await asyncio.to_thread(sync_ddgs_call, query, page)

            if not results:
                return f"No more web results found on page {page} for '{query}'."

            formatted = [f"Source: {r['href']}\n{r['body']}" for r in results]
            header = f"--- Web Search Results for '{query}' (Page {page}) ---\n"
            return header + "\n\n---\n\n".join(formatted)

        except Exception as e:
            return f"Web search snag (v7+): {e}"

    return FunctionTool.from_defaults(
        async_fn=search_web,
        name="search_web",
        description=(
            "Search the internet for recent news and facts not related to OCF. "
            "Supports pagination via the 'page' parameter (e.g., page 2 for results 4-6)."
        ),
    )