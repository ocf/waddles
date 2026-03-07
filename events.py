"""Event classes for the OCF Agent Workflow."""

from typing import Optional, List, Any, Dict
from llama_index.core.workflow import Event


class QueryReceivedEvent(Event):
    """Initial event when a user asks a question."""
    question: str
    user_name: str
    persona_prompt: str
    use_thinking: bool


class ToolDecisionEvent(Event):
    """Event after deciding which tools to call."""
    tool_calls: List[Dict[str, Any]]
    original_question: str
    user_name: str
    persona_prompt: str
    use_thinking: bool


class ToolExecutionEvent(Event):
    """Event to trigger individual tool execution."""
    tool_name: str
    query: str


class ToolResultEvent(Event):
    """Event when a tool returns results."""
    tool_name: str
    query: str
    result: str


class ContextGatheredEvent(Event):
    """Event when all context has been gathered."""
    context_str: str
    query_str: str
    persona_prompt: str
    use_thinking: bool


class StreamingStartedEvent(Event):
    """Event when streaming response starts."""
    pass


class StreamChunkEvent(Event):
    """Event for each streaming chunk - used for progress updates."""
    thinking_delta: Optional[str] = None
    answer_delta: Optional[str] = None
    display_text: str = ""
    is_final: bool = False


class ResponseCompleteEvent(Event):
    """Final event when response generation is complete."""
    final_text: str
    was_stopped: bool = False


class StopRequestedEvent(Event):
    """Event when user requests to stop generation."""
    user_id: int
    query_id: int
