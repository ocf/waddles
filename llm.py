from llama_index.llms.openai_like import OpenAILike
from config import (
    LLM_REPETITION_PENALTY,
    SGLANG_URL,
    LLM_CONTEXT_WINDOW,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_FREQUENCY_PENALTY,
    LLM_PRESENCE_PENALTY,
    LLM_MIN_P,
)


def get_llm(thinking: bool) -> OpenAILike:
    """Create an LLM instance with optional thinking mode.

    Args:
        thinking: Whether to enable thinking mode.

    Returns:
        Configured OpenAILike LLM instance.
    """
    return OpenAILike(
        model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        api_base=SGLANG_URL,
        api_key="fake-key",
        context_window=LLM_CONTEXT_WINDOW,
        is_chat_model=True,
        is_function_calling_model=True,
        timeout=LLM_TIMEOUT,
        temperature=LLM_TEMPERATURE,
        additional_kwargs={
            "stop": ["<|im_end|>", "<|im_start|>", "\nUser:"],
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": thinking},
                "repetition_penalty": LLM_REPETITION_PENALTY,
                "min_p": LLM_MIN_P
            }
        }
    )