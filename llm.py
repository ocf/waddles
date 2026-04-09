from llama_index.llms.openai_like import OpenAILike
from config import (
    LLM_REPETITION_PENALTY,
    VLLM_URL,
    MODEL_NAME,
    LLM_CONTEXT_WINDOW,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_TOP_K,
)


def get_llm(thinking: bool) -> OpenAILike:
    """Create an LLM instance with optional thinking mode.

    Args:
        thinking: Whether to enable thinking mode.

    Returns:
        Configured OpenAILike LLM instance.
    """
    return OpenAILike(
        model=MODEL_NAME,
        api_base=VLLM_URL,
        api_key="fake-key",
        context_window=LLM_CONTEXT_WINDOW,
        is_chat_model=True,
        is_function_calling_model=True,
        timeout=LLM_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        additional_kwargs={
            "stop": ["<turn|>", "<|turn>", "<|tool_response>"],
            "top_p": LLM_TOP_P,
            "extra_body": {
                "top_k": LLM_TOP_K,
                "chat_template_kwargs": {"enable_thinking": thinking}
            }
        }
    )
