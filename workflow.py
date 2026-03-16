"""OCF Agent Workflow using LlamaIndex Workflow system."""

import asyncio
import json
import time
from typing import Optional, List, Callable, Awaitable, Any, Dict, Union

from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike

from events import (
    AgentInputEvent,
    ToolDecisionEvent,
    ContextGatheredEvent,
    ResponseCompleteEvent,
)
from tools.tools import get_all_tools, get_tool_prompt


# Type alias for the message update callback
MessageCallback = Callable[[str], Awaitable[None]]


class OCFAgentWorkflow(Workflow):
    """Main agent workflow for processing OCF queries.

    This workflow handles a cyclic tool-calling loop where the LLM can call
    multiple tools, receive their output, and decide whether to call more
    tools or finalize its response.
    """

    def __init__(
        self,
        llm_standard: OpenAILike,
        llm_thinking: OpenAILike,
        index: VectorStoreIndex,
        timeout: float = 300.0,
        verbose: bool = False,
        depth: int = 0,
        memory: Optional[Any] = None,
    ):
        """Initialize the workflow."""
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm_standard = llm_standard
        self.llm_thinking = llm_thinking
        self.index = index
        self.depth = depth
        self.memory = memory

        # Create tools
        self.tool_map = get_all_tools(index, depth=depth)
        self.tools = list(self.tool_map.values())

        # Per-run state
        self._message_callback: Optional[MessageCallback] = None
        self._cancelled: bool = False
        self._chat_history: List[ChatMessage] = []
        self._max_loops = 20
        self._loop_count = 0
        self._question = ""
        self._user_name = "User"
        self._persona_prompt = ""
        self._use_thinking = False

    @step
    async def handle_start(self, ctx: Context, ev: StartEvent) -> AgentInputEvent:
        """Initializes the reasoning loop from a fresh start."""
        self._question = ev.get("question", "")
        self._user_name = ev.get("user_name", "User")
        self._persona_prompt = ev.get("persona_prompt", "")
        self._use_thinking = ev.get("use_thinking", False)
        self._message_callback = ev.get("message_callback")
        image_urls = ev.get("image_urls", [])
        self._cancelled = False
        self._loop_count = 0

        # Construct the system persona including tool instructions
        system_content = self._persona_prompt + "\n\n"
        system_content += get_tool_prompt(self._user_name, self._use_thinking)

        # Retrieve relevant past history from memory blocks
        past_history = []
        if self.memory:
            # For multimodal queries, we only search memory with the text part
            past_history = await self.memory.aget(input=self._question)

        # Construct multimodal message if images are provided
        user_blocks: List[Union[TextBlock, ImageBlock]] = [TextBlock(text=self._question)]
        user_blocks.extend([ImageBlock(url=url) for url in image_urls])
        user_msg = ChatMessage(role=MessageRole.USER, blocks=user_blocks)

        self._chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            *past_history,
            user_msg
        ]

        if self._message_callback:
            await self._message_callback("🔍 Deciding how to answer...")

        return AgentInputEvent()

    @step
    async def handle_context(self, ctx: Context, ev: ContextGatheredEvent) -> AgentInputEvent:
        """Processes tool results and triggers the next reasoning step."""
        # NEW: Properly append individual tool results to satisfy OpenAI schema
        for tool_res in ev.tool_results:
            self._chat_history.append(ChatMessage(
                role=MessageRole.TOOL,
                content=str(tool_res["content"]),
                additional_kwargs={
                    "tool_call_id": tool_res["id"],
                    "name": tool_res["name"]
                }
            ))
        return AgentInputEvent()

    @step
    async def agent_step(
        self, ctx: Context, ev: AgentInputEvent
    ) -> ToolDecisionEvent | StopEvent:
        """The central reasoning node: calls the LLM and decides whether to tool-call or finish."""
        if self._cancelled or self._loop_count >= self._max_loops:
            stop_reason = "Workflow stopped by user." if self._cancelled else "Reached maximum reasoning steps."
            return StopEvent(result=ResponseCompleteEvent(final_text=stop_reason, was_stopped=self._cancelled))

        self._loop_count += 1
        active_llm = self.llm_thinking if self._use_thinking else self.llm_standard

        if self._message_callback:
            status = "💭 Thinking..." if self._use_thinking else "✨ Generating response..."
            await self._message_callback(status)

        response_stream = await active_llm.astream_chat_with_tools(
            self.tools,
            chat_history=self._chat_history
        )

        thinking_text = ""
        full_content = ""
        display_text = ""
        last_edit_time = time.time()

        # Assemble streaming tool call fragments
        streaming_tool_calls = {}

        async for chunk in response_stream:
            if self._cancelled:
                if hasattr(response_stream, "aclose"):
                    await response_stream.aclose()
                break

            # Handle thinking mode deltas
            think_delta = chunk.additional_kwargs.get("thinking_delta", "")
            if think_delta:
                thinking_text += think_delta
                truncated = thinking_text if len(thinking_text) <= 1800 else "..." + thinking_text[-1800:]
                display_text = f"💭 **Thinking...**\n```text\n{truncated}\n```"

            # Handle regular content deltas
            elif chunk.delta:
                full_content += chunk.delta
                display_text = full_content

            # Capture standard OpenAI-style tool calls coming from SGLang
            tool_call_deltas = chunk.additional_kwargs.get("tool_calls") or []
            for tc in tool_call_deltas:
                # Safely extract values whether LlamaIndex returns dicts or objects
                if isinstance(tc, dict):
                    idx = tc.get("index")
                    tc_id = tc.get("id")
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args = tc.get("function", {}).get("arguments", "")
                else:
                    idx = tc.index
                    tc_id = getattr(tc, "id", None)
                    fn_name = getattr(tc.function, "name", "") if hasattr(tc, "function") else ""
                    fn_args = getattr(tc.function, "arguments", "") if hasattr(tc, "function") else ""

                if idx not in streaming_tool_calls:
                    streaming_tool_calls[idx] = {
                        "id": tc_id or f"call_{idx}",
                        "name": fn_name,
                        "arguments": fn_args
                    }
                else:
                    if fn_args:
                        streaming_tool_calls[idx]["arguments"] += fn_args

            # Update Discord message periodically
            now = time.time()
            if self._message_callback and now - last_edit_time > 1.2:
                if display_text:
                    await self._message_callback(display_text[:2000])
                last_edit_time = now

        # Parse the assembled JSON tool calls
        tool_calls = []
        openai_history_tools = []
        for tc in streaming_tool_calls.values():
            try:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "kwargs": json.loads(tc["arguments"] or "{}")
                })
                # Format required for standard OpenAI assistant history
                openai_history_tools.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"]
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to parse tool call {tc['name']}: {e}")

        # Store assistant response in history
        kwargs = {}
        if thinking_text:
            kwargs["thinking_text"] = thinking_text
        if openai_history_tools:
            kwargs["tool_calls"] = openai_history_tools

        self._chat_history.append(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=full_content,
            additional_kwargs=kwargs,
        ))

        if tool_calls and not self._cancelled:
            return ToolDecisionEvent(
                tool_calls=tool_calls,
                original_question=self._question,
                user_name=self._user_name,
                persona_prompt=self._persona_prompt,
                use_thinking=self._use_thinking,
            )

        # Final Cleanup for User Display
        final_display = display_text if display_text else "I couldn't generate a response."

        if self._message_callback:
            await self._message_callback(final_display[:2000])

        # Update persistent memory blocks with this interaction
        if self.memory and not self._cancelled:
            await self.memory.aput(ChatMessage(role=MessageRole.USER, content=self._question))
            await self.memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=full_content))

        return StopEvent(result=ResponseCompleteEvent(
            final_text=final_display,
            was_stopped=self._cancelled
        ))

    @step
    async def execute_tools(self, ctx: Context, ev: ToolDecisionEvent) -> ContextGatheredEvent:
        """Executes one or more tools in parallel and returns their results."""
        tasks = []
        tool_meta = []  # Keep track of ID and Name for the results

        for tool_call in ev.tool_calls:
            if self._cancelled:
                break

            name = tool_call.get("name")
            kwargs = tool_call.get("kwargs", {})
            call_id = tool_call.get("id")

            if name in self.tool_map:
                tool_meta.append({"id": call_id, "name": name, "kwargs": kwargs})
                tasks.append(self.tool_map[name].acall(**kwargs))
            else:
                print(f"Warning: LLM tried to call unknown tool {name}")
                # We must return a result for the ID even if the tool is missing to satisfy the API
                tool_meta.append({"id": call_id, "name": name, "kwargs": kwargs})
                async def fallback(): return f"Error: Tool {name} not found."
                tasks.append(fallback())

        if not tasks:
            return ContextGatheredEvent(
                tool_results=[],
                query_str=self._question,
                persona_prompt=self._persona_prompt,
                use_thinking=self._use_thinking,
            )

        if self._message_callback:
            labels = [f"{m['name']}" for m in tool_meta]
            await self._message_callback("🔍 Searching: " + " & ".join([f"`{l}`" for l in labels]))

        results = await asyncio.gather(*tasks)

        # Map results back to their IDs for the history schema
        formatted_results = []
        for meta, res in zip(tool_meta, results):
            formatted_results.append({
                "id": meta["id"],
                "name": meta["name"],
                "content": res
            })

        return ContextGatheredEvent(
            tool_results=formatted_results,
            query_str=self._question,
            persona_prompt=self._persona_prompt,
            use_thinking=self._use_thinking,
        )

    def cancel(self) -> None:
        """Request cancellation of the workflow."""
        self._cancelled = True


async def run_query_workflow(
    workflow: OCFAgentWorkflow,
    question: str,
    user_name: str,
    persona_prompt: str,
    use_thinking: bool,
    image_urls: Optional[List[str]] = None,
    message_callback: Optional[MessageCallback] = None,
) -> ResponseCompleteEvent:
    """Convenience function to run a query through the workflow."""
    return await workflow.run(
        question=question,
        user_name=user_name,
        persona_prompt=persona_prompt,
        use_thinking=use_thinking,
        image_urls=image_urls,
        message_callback=message_callback,
    )
