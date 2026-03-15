from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool

from config import DOCS_DIR, DOCS_TOP_N


def create_docs_search_tool(index: VectorStoreIndex) -> FunctionTool:
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
            # Initial retrieval
            retriever = index.as_retriever(similarity_top_k=DOCS_TOP_N)
            nodes = await retriever.aretrieve(query)

            if not nodes:
                return "No internal documentation found."

            return "\n---------------------\n".join([
                f"Source: {n.metadata.get('file_path', 'Unknown').removeprefix(DOCS_DIR).lstrip('/')}\n"
                f"Last modified: {n.metadata.get('last_modified_date', 'Unknown')}\n"
                f"Content: {n.get_content()}"
                for n in nodes
            ])
        except Exception as e:
            return f"Documentation search error: {e}"

    return FunctionTool.from_defaults(
        async_fn=search_docs,
        name="search_docs",
        description=(
            "Search internal OCF documentation about rules, services, and policies. "
            "Use for OCF-specific questions about lab policies, printing, computing resources, etc."
        ),
    )
