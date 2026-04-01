from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from config import DOCS_DIR, DOCS_TOP_N
from database import get_nodes


def create_docs_search_tool(index: VectorStoreIndex) -> FunctionTool:
    """Create the internal documentation search tool.

    Args:
        index: The VectorStoreIndex containing OCF documentation.
    """

    # Pre-fetch nodes for BM25 (this happens once when the tool is created)
    all_nodes = get_nodes(index)

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
            # 1. Semantic (Vector) Retriever
            vector_retriever = index.as_retriever(similarity_top_k=DOCS_TOP_N)

            # 2. Keyword (BM25) Retriever
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=all_nodes,
                similarity_top_k=DOCS_TOP_N
            )

            # 3. Combined Hybrid Retriever (Fusion)
            retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=DOCS_TOP_N,
                num_queries=1,  # We only need the original query for standard hybrid
                mode="reciprocal_rerank",
                use_async=True,
            )

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
