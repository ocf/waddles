import datetime
from typing import List, Optional

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import FunctionTool

from config import DOCS_RECENCY_WEIGHT, DOCS_RETRIEVE_CHUNKS, DOCS_TOP_N


class RecencyReranker(BaseNodePostprocessor):
    """A node postprocessor that reranks nodes based on their recency.

    Weights recency at 30% and vector similarity at 70%.
    """

    top_n: int = 5
    recency_weight: float = 0.3

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        now = datetime.datetime.now()
        for node in nodes:
            # Use last_modified_date (populated via Git for OCF docs, or file system fallback)
            date_str = node.node.metadata.get("last_modified_date")
            recency_score = 0.0
            if date_str:
                try:
                    if isinstance(date_str, str) and len(date_str) > 10:
                        date_str = date_str[:10]

                    # SimpleDirectoryReader might provide datetime object or string
                    if isinstance(date_str, datetime.datetime):
                        modified_date = date_str
                    else:
                        modified_date = datetime.datetime.strptime(str(date_str), "%Y-%m-%d")

                    days_old = max(0, (now - modified_date).days)
                    # 1.0 for brand new, 0.0 for 1 year old
                    recency_score = max(0.0, 1.0 - (days_old / 365.0))
                except Exception:
                    pass

            sim_score = node.score or 0.0
            # Combine scores
            node.score = (1.0 - self.recency_weight) * sim_score + self.recency_weight * recency_score

        # Sort by the new score
        nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
        return nodes[: self.top_n]


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
            # Initial retrieval of a larger pool of candidates
            retriever = index.as_retriever(similarity_top_k=DOCS_RETRIEVE_CHUNKS)
            nodes = await retriever.aretrieve(query)

            if not nodes:
                return "No internal documentation found."

            # Rerank nodes based on recency
            reranker = RecencyReranker(top_n=DOCS_TOP_N, recency_weight=DOCS_RECENCY_WEIGHT)
            nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query))

            return "\n---------------------\n".join([n.get_content() for n in nodes])
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
