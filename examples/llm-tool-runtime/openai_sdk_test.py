#!/usr/bin/env python3
"""Exercise AgentGateway builtin tools through the OpenAI Python SDK."""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Sequence


DEFAULT_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_MODEL = "qwen3.6-flash"
EXPECTED_MARKER = "OPENAI_SDK_TOOLS_OK_391"


class SDKTestFailure(RuntimeError):
    """The AgentGateway Responses stream did not satisfy the test contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SDKTestFailure(message)


def consume_stream(events: Iterable[Any]) -> tuple[str, Any]:
    event_types: list[str] = []
    deltas: list[str] = []
    completed_responses: list[Any] = []

    for event in events:
        event_type = getattr(event, "type", "")
        event_types.append(event_type)
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            require(isinstance(delta, str), "output_text delta was not a string")
            deltas.append(delta)
        elif event_type == "response.completed":
            completed_responses.append(getattr(event, "response", None))

    require(bool(event_types), "Responses stream returned no events")
    require(event_types[0] == "response.created", "first event was not response.created")
    require(bool(deltas), "Responses stream omitted response.output_text.delta")
    require(
        len(completed_responses) == 1,
        "Responses stream did not contain exactly one response.completed event",
    )
    require(event_types[-1] == "response.completed", "last event was not response.completed")
    return "".join(deltas), completed_responses[0]


def run_sdk_case(client: Any, model: str) -> str:
    stream = client.responses.create(
        model=model,
        input=(
            "Before answering, you must call both tools. Use web_search to find one current "
            "fact about the AgentGateway open-source project, and use code_interpreter to "
            "calculate 17 * 23. Call both tools in the same turn if possible. Then include "
            f"the exact marker {EXPECTED_MARKER} and the searched fact in the final answer."
        ),
        tools=[
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ],
        parallel_tool_calls=True,
        stream=True,
    )
    streamed_text, response = consume_stream(stream)

    require(response is not None, "response.completed omitted its response")
    require(getattr(response, "status", None) == "completed", "response did not complete")
    final_text = getattr(response, "output_text", None)
    require(isinstance(final_text, str) and bool(final_text), "response returned no output_text")
    require(streamed_text == final_text, "streamed deltas did not match final output_text")
    require(EXPECTED_MARKER in final_text, "final output omitted the verification marker")
    return final_text


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AGENTGATEWAY_BASE_URL", DEFAULT_BASE_URL),
        help="AgentGateway OpenAI-compatible base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("AGENTGATEWAY_API_KEY", "agentgateway-local-client"),
        help="AgentGateway client API key; AGENTGATEWAY_API_KEY is used when set",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AGENTGATEWAY_LIVE_MODEL", DEFAULT_MODEL),
        help="model name configured in AgentGateway (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else ())
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SDKTestFailure(
            "the openai package is required; run with `uv run --with openai` or install it"
        ) from error

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/") + "/",
        timeout=180.0,
        max_retries=0,
    )
    output = run_sdk_case(client, args.model)
    print("PASS openai-sdk-streaming-tools")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
