"""Event classes for the OCF Agent Workflow."""

from typing import List, Any, Dict
from llama_index.core.workflow import Event


class AgentInputEvent(Event):
    """Event to trigger the agent reasoning step with the current chat history."""
    pass


class ToolDecisionEvent(Event):
    """Event after deciding which tools to call."""
    tool_calls: List[Dict[str, Any]]
    original_question: str
    user_name: str
    persona_prompt: str
    use_thinking: bool


class ContextGatheredEvent(Event):
    """Event when all context has been gathered."""
    context_str: str
    query_str: str
    persona_prompt: str
    use_thinking: bool


class ResponseCompleteEvent(Event):
    """Final event when response generation is complete."""
    final_text: str
    was_stopped: bool = False
