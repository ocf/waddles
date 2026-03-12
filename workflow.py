"""OCF Agent Workflow using LlamaIndex Workflow system."""

import asyncio
import re
import time
from typing import Optional, List, Callable, Awaitable, Any, Dict

from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core.llms import ChatMessage, MessageRole
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
    ):
        """Initialize the workflow."""
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm_standard = llm_standard
        self.llm_thinking = llm_thinking
        self.index = index
        self.depth = depth

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

    def _parse_qwen_tools(self, content: str) -> List[Dict[str, Any]]:
        """Parses Qwen's XML-style tool calls from a string."""
        if not content or "<tool_call>" not in content:
            return []

        parsed_calls = []
        # Matches <function=name> followed by <parameter=name>value</parameter>
        matches = re.finditer(
            r"<function=(\w+)>\s*<parameter=(\w+)>(.*?)</parameter>",
            content,
            re.DOTALL
        )

        for match in matches:
            fn_name = match.group(1)
            param_name = match.group(2)
            param_val = match.group(3).strip("\n")

            parsed_calls.append({
                "name": fn_name,
                "kwargs": {param_name: param_val}
            })

        return parsed_calls

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
        system_content = self._persona_prompt.format(query_str=self._question) + "\n\n" + get_tool_prompt(self._question, use_thinking=self._use_thinking)

        # Construct multimodal message if images are provided
        if image_urls:
            user_blocks = [{"block_type": "text", "text": f"[{self._user_name}] says: \n{self._question}"}]
            for url in image_urls:
                user_blocks.append({"block_type": "image", "url": url})
            
            user_msg = ChatMessage(role=MessageRole.USER, blocks=user_blocks)
        else:
            user_msg = ChatMessage(role=MessageRole.USER, content=f"[{self._user_name}] says: \n{self._question}")

        self._chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            user_msg
        ]

        if self._message_callback:
            await self._message_callback("🔍 Deciding how to answer...")

        return AgentInputEvent()

    @step
    async def handle_context(self, ctx: Context, ev: ContextGatheredEvent) -> AgentInputEvent:
        """Processes tool results and triggers the next reasoning step."""
        # Wrap tool results in a user message for the LLM to observe
        self._chat_history.append(ChatMessage(role=MessageRole.USER, content=ev.context_str))
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

        # We use astream_chat_with_tools to ensure the LLM receives the tool definitions
        response_stream = await active_llm.astream_chat_with_tools(
            self.tools,
            chat_history=self._chat_history
        )

        thinking_text = ""
        full_content = ""
        display_text = ""
        last_edit_time = time.time()

        async for chunk in response_stream:
            if self._cancelled:
                if hasattr(response_stream, "aclose"): await response_stream.aclose()
                break

            # Handle thinking mode deltas
            think_delta = chunk.additional_kwargs.get("thinking_delta", "")
            if think_delta:
                thinking_text += think_delta
                # Limit thinking block length for Discord
                truncated = thinking_text if len(thinking_text) <= 1800 else "..." + thinking_text[-1800:]
                display_text = f"💭 **Thinking...**\n```text\n{truncated}\n```"

            # Handle regular content deltas
            elif chunk.delta:
                full_content += chunk.delta
                display_text = full_content

            # Update Discord message periodically
            now = time.time()
            if self._message_callback and now - last_edit_time > 1.2:
                if display_text:
                    await self._message_callback(display_text[:2000])
                last_edit_time = now

        # Parse potential tool calls from the finalized content
        tool_calls = self._parse_qwen_tools(full_content)

        # Store assistant response in history
        self._chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=full_content))

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

        return StopEvent(result=ResponseCompleteEvent(
            final_text=final_display,
            was_stopped=self._cancelled
        ))

    @step
    async def execute_tools(self, ctx: Context, ev: ToolDecisionEvent) -> ContextGatheredEvent:
        """Executes one or more tools in parallel and returns their results."""
        tasks = []
        labels = []

        for tool_call in ev.tool_calls:
            if self._cancelled: break

            name = tool_call.get("name")
            kwargs = tool_call.get("kwargs", {})

            if name in self.tool_map:
                param_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                labels.append(f"{name}: {param_str}")
                tasks.append(self.tool_map[name].acall(**kwargs))
            else:
                print(f"Warning: LLM tried to call unknown tool {name}")

        if not tasks:
            return ContextGatheredEvent(
                context_str="No valid tools were called.",
                query_str=self._question,
                persona_prompt=self._persona_prompt,
                use_thinking=self._use_thinking,
            )

        if self._message_callback:
            await self._message_callback("🔍 Searching: " + " & ".join([f"`{l}`" for l in labels]))

        results = await asyncio.gather(*tasks)

        # Format observations for the LLM
        context_pieces = [f"--- Tool Result: {l} ---\n{r}" for l, r in zip(labels, results)]

        return ContextGatheredEvent(
            context_str="Tool Results:\n" + "\n\n".join(context_pieces),
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
