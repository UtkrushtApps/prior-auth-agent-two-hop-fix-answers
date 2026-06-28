"""LLM client wrapper using LiteLLM for provider-agnostic model access."""

import litellm
from agent.config import config


def complete(messages: list[dict], response_format: dict | None = None) -> str:
    """Call the configured LLM model and return the text content.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        response_format: Optional format dict, e.g. {"type": "json_object"}.

    Returns:
        The model's text response.
    """
    kwargs: dict = {
        "model": config.model,
        "messages": messages,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content
