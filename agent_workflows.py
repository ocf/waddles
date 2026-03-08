"""OCF Agent Workflow using LlamaIndex Workflow system."""

import asyncio
import json
import re
import time
from typing import Optional, List, Callable, Awaitable, Any

from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike

from events import (
    QueryReceivedEvent,
    ToolDecisionEvent,
    ToolResultEvent,
    ContextGatheredEvent,
    StreamChunkEvent,
    ResponseCompleteEvent,
)
from tools import get_all_tools, get_tool_decision_prompt


# Type alias for the message update callback
MessageCallback = Callable[[str], Awaitable[None]]


class OCFAgentWorkflow(Workflow):
    """Main agent workflow for processing OCF queries.

    This workflow handles:
    1. Receiving a query from a user
    2. Deciding which tools to call (web search, docs search, or both)
    3. Executing the tools and gathering context
    4. Generating a streaming response using the persona prompt
    """

    def __init__(
        self,
        llm_standard: OpenAILike,
        llm_thinking: OpenAILike,
        index: VectorStoreIndex,
        timeout: float = 300.0,
        verbose: bool = False,
    ):
        """Initialize the workflow.

        Args:
            llm_standard: The standard LLM without thinking mode.
            llm_thinking: The LLM with thinking mode enabled.
            index: The VectorStoreIndex for document retrieval.
            timeout: Workflow timeout in seconds.
            verbose: Whether to enable verbose logging.
        """
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm_standard = llm_standard
        self.llm_thinking = llm_thinking
        self.index = index

        # Create tools
        self.tool_map = get_all_tools(index)

        # Per-run state (set when run() is called)
        self._message_callback: Optional[MessageCallback] = None
        self._cancelled: bool = False

    @step
    async def handle_query(
        self, ctx: Context, ev: StartEvent
    ) -> ToolDecisionEvent:
        """Entry point: receive query and decide on tools to call."""
        question = ev.get("question", "")
        user_name = ev.get("user_name", "User")
        persona_prompt = ev.get("persona_prompt", "")
        use_thinking = ev.get("use_thinking", False)
        self._message_callback = ev.get("message_callback")
        self._cancelled = False

        if self._message_callback:
            await self._message_callback("🔍 Deciding how to answer...")

        response = await self.llm_standard.achat_with_tools(
            list(self.tool_map.values()),
            user_msg=get_tool_decision_prompt(question)
        )

        # 1. Start with the fallback (Search Docs)
        tool_calls = [{"name": "search_docs", "kwargs": {"query": question}}]

        # 2. Check if Qwen 3.5 sent its special XML tags
        content = response.message.content
        if content and "<tool_call>" in content:
            parsed_calls = []

            # UPDATED: Use finditer to grab EVERY function and parameter pair
            # This regex looks for <function=X> followed by <parameter=Y>Z</parameter>
            matches = re.finditer(r"<function=(\w+)>\s*<parameter=(\w+)>(.*?)</parameter>", content, re.DOTALL)

            for match in matches:
                fn_name = match.group(1)
                param_name = match.group(2)
                param_val = match.group(3).strip()

                parsed_calls.append({
                    "name": fn_name,
                    "kwargs": {param_name: param_val}
                })

            # If we successfully parsed any tool calls, override the fallback
            if parsed_calls:
                tool_calls = parsed_calls

        return ToolDecisionEvent(
            tool_calls=tool_calls,
            original_question=question,
            user_name=user_name,
            persona_prompt=persona_prompt,
            use_thinking=use_thinking,
        )

    @step
    async def execute_tools(self, ctx: Context, ev: ToolDecisionEvent) -> ContextGatheredEvent:
        tasks = []
        labels = []

        for tool_call in ev.tool_calls:
            if self._cancelled: break

            name = tool_call.get("name")
            kwargs = tool_call.get("kwargs", {})

            if name in self.tool_map:
                # Format a nice label using the dynamic kwargs
                param_str = ", ".join([f"{v}" for k, v in kwargs.items()])
                labels.append(f"{name}: {param_str}")

                # UPDATED: Unpack the kwargs into acall
                tasks.append(self.tool_map[name].acall(**kwargs))
            else:
                print(f"Warning: LLM tried to call unknown tool {name}")

        if self._message_callback and labels:
            await self._message_callback("🔍 Searching: " + " & ".join([f"`{l}`" for l in labels]))

        results = await asyncio.gather(*tasks)

        context_pieces = [f"--- {l.upper()} ---\n{r}" for l, r in zip(labels, results)]

        return ContextGatheredEvent(
            context_str="\n\n".join(context_pieces),
            query_str=f"[{ev.user_name}] says: \n{ev.original_question}",
            persona_prompt=ev.persona_prompt,
            use_thinking=ev.use_thinking,
        )

    @step
    async def generate_response(
        self, ctx: Context, ev: ContextGatheredEvent
    ) -> StopEvent:
        """Generate the final streaming response.

        Args:
            ctx: The workflow context.
            ev: The context gathered event.

        Returns:
            StopEvent with the final response.
        """
        # Format the prompt with context
        formatted_prompt = ev.persona_prompt.format(
            context_str=ev.context_str,
            query_str=ev.query_str
        )

        # Select the appropriate LLM
        active_llm = self.llm_thinking if ev.use_thinking else self.llm_standard

        # Start streaming
        if self._message_callback:
            status = "💭 Thinking deeply..." if ev.use_thinking else "✨ Generating response..."
            await self._message_callback(status)

        response_stream = await active_llm.astream_complete(formatted_prompt)

        thinking_text = ""
        answer_text = ""
        display_text = ""
        last_edit_time = time.time()

        async for chunk in response_stream:
            # Check for cancellation
            if self._cancelled:
                stop_msg = "\n\n*[Stopped by user]*"

                # Format appropriately based on what we have
                if thinking_text and not answer_text:
                    truncated_thoughts = (
                        thinking_text if len(thinking_text) <= 1850
                        else "..." + thinking_text[-1850:]
                    )
                    display_text = f"💭 **Thinking...**\n```text\n{truncated_thoughts}\n```{stop_msg}"
                else:
                    answer_text += stop_msg
                    display_text = answer_text

                # Close the stream
                if hasattr(response_stream, "aclose"):
                    await response_stream.aclose()

                break

            # Process the chunk
            think_delta = chunk.additional_kwargs.get("thinking_delta", "")

            if think_delta:
                thinking_text += think_delta
                truncated_thoughts = (
                    thinking_text if len(thinking_text) <= 1850
                    else "..." + thinking_text[-1850:]
                )
                display_text = f"💭 **Thinking...**\n```text\n{truncated_thoughts}\n```"
            elif chunk.delta:
                answer_text += chunk.delta
                display_text = answer_text

            # Update Discord message periodically to avoid rate limits
            current_time = time.time()
            if self._message_callback and current_time - last_edit_time > 1.2:
                if display_text:
                    await self._message_callback(display_text[:2000])
                last_edit_time = current_time

        # Final update
        final_content = display_text if display_text else "I couldn't think of anything to say."

        if self._message_callback:
            await self._message_callback(final_content[:2000])

        return StopEvent(result=ResponseCompleteEvent(
            final_text=final_content,
            was_stopped=self._cancelled
        ))

    def cancel(self) -> None:
        """Request cancellation of the workflow."""
        self._cancelled = True


async def run_query_workflow(
    workflow: OCFAgentWorkflow,
    question: str,
    user_name: str,
    persona_prompt: str,
    use_thinking: bool,
    message_callback: Optional[MessageCallback] = None,
) -> ResponseCompleteEvent:
    """Convenience function to run a query through the workflow.

    Args:
        workflow: The OCFAgentWorkflow instance.
        question: The user's question.
        user_name: The user's display name.
        persona_prompt: The persona prompt template.
        use_thinking: Whether to use thinking mode.
        message_callback: Optional callback for status updates.

    Returns:
        The ResponseCompleteEvent with the final response.
    """
    result = await workflow.run(
        question=question,
        user_name=user_name,
        persona_prompt=persona_prompt,
        use_thinking=use_thinking,
        message_callback=message_callback,
    )

    return result
