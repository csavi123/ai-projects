from __future__ import annotations

import json

import anthropic

from vivaprep.config import LLM_MODEL

# The one place that calls Claude. Everything else asks this file, so the model,
# settings and error handling live in one spot. The API key comes from .env
# (config.py loads it); anthropic.Anthropic() reads ANTHROPIC_API_KEY.

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def ask_claude(system: str, user_message: str, max_tokens: int = 4000) -> str:
    """Send one question to Claude and return its text answer."""
    return _call(system, user_message, max_tokens, {"effort": "low"})  # short grounded answers need little thinking


def ask_claude_json(system: str, user_message: str, schema: dict, max_tokens: int = 4000) -> dict:
    """Like ask_claude, but the API forces the reply to be JSON matching `schema` (structured output)."""
    output_config = {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    return json.loads(_call(system, user_message, max_tokens, output_config))


def _call(system: str, user_message: str, max_tokens: int, output_config: dict) -> str:
    """The one API call, with clear messages for the errors you're likely to hit."""
    try:
        response = _get_client().messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            output_config=output_config,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )  # no temperature: Haiku 5.5 rejects sampling settings
    except anthropic.AuthenticationError:
        raise SystemExit("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        raise SystemExit("Too many requests to Claude. Wait a minute and try again.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise SystemExit("Could not reach the Claude API. Check your internet connection.")

    if response.stop_reason == "refusal":
        raise SystemExit("Claude declined to answer this question.")
    if response.stop_reason == "max_tokens":
        raise SystemExit("Claude's answer was cut off (hit max_tokens).")
    # The response can also hold (empty) thinking blocks; keep only the text.
    return "".join(block.text for block in response.content if block.type == "text").strip()
