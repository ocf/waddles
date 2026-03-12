from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike


def create_delegation_tool(
    llm_standard: OpenAILike,
    llm_thinking: OpenAILike,
    index: VectorStoreIndex,
    current_depth: int = 0,
    max_depth: int = 2
) -> FunctionTool:
    """Creates a tool that allows the agent to delegate tasks to a sub-agent."""

    async def delegate_task(task: str, use_thinking: bool = False) -> str:
        """
        Delegates a specific, complex research or processing task to a specialized sub-agent.
        Use this for:
        1. Scraping multiple links found on a page.
        2. Searching for and synthesizing information from many documents.
        3. Handling long-running tasks that would clutter your main context.

        Args:
            task: A detailed description of what the sub-agent should do.
            use_thinking: Whether the sub-agent should use 'thinking' mode (slower but more thorough).
        """
        if current_depth >= max_depth:
            return "Error: Maximum delegation depth reached. You must handle this task yourself."

        # Import locally to avoid circular dependency
        from workflow import OCFAgentWorkflow
        from events import ResponseCompleteEvent

        # Create a worker workflow instance
        worker = OCFAgentWorkflow(
            llm_standard=llm_standard,
            llm_thinking=llm_thinking,
            index=index,
            timeout=180.0,  # Shorter timeout for workers
        )

        # Override the tool map for the worker to include the same tools
        # but with an incremented depth
        from tools.tools import get_all_tools
        worker.tool_map = get_all_tools(index, depth=current_depth + 1)
        worker.tools = list(worker.tool_map.values())

        # Worker-specific persona: Focus on execution, skip conversational filler.
        worker_persona = (
            "You are a specialized research sub-agent. Your goal is to execute the following task "
            "efficiently and return a structured summary of your findings.\n"
            "Task: {query_str}\n\n"
            "Guidelines:\n"
            "- Be concise and factual.\n"
            "- If you encounter links, scrape them if necessary to fulfill the task.\n"
            "- Return ONLY the final synthesized answer. No intro/outro."
        )

        try:
            # Run the worker without a message callback to avoid Discord spam
            result = await worker.run(
                question=task,
                user_name="Manager Agent",
                persona_prompt=worker_persona,
                use_thinking=use_thinking,
                message_callback=None
            )

            if isinstance(result, ResponseCompleteEvent):
                return f"Sub-agent Summary:\n{result.final_text}"
            return str(result)

        except Exception as e:
            return f"Delegation failed: {e}"

    return FunctionTool.from_defaults(
        async_fn=delegate_task,
        name="delegate_task",
        description=(
            "Spawns a specialized worker agent to handle complex sub-tasks, multi-link scraping, "
            "or heavy document research. Returns a concise summary of the worker's findings."
        )
    )
