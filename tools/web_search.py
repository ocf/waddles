import asyncio
from ddgs import DDGS
from llama_index.core.tools import FunctionTool


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
        description=(
            "Search the internet for recent news and facts not related to OCF. "
            "Use for general knowledge, current events, and external information."
        ),
    )