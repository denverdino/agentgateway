#!/usr/bin/env python3
"""Exercise AgentGateway tool-runtime cases through the Anthropic Python SDK."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Mapping, Sequence


CASE_NAMES = (
    "web-search",
    "code-interpreter",
    "combined",
    "programmatic-web-search",
    "streaming-tool-runtime",
)
DEFAULT_BASE_URL = "http://127.0.0.1:4000"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class SDKTestFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SDKTestFailure(message)


def case_payloads(model: str) -> dict[str, dict[str, Any]]:
    web_tool = {"type": "web_search_20250305", "name": "web_search"}
    code_tool = {"type": "code_execution_20250825", "name": "code_execution"}
    return {
        "web-search": {
            "model": model,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": (
                    "You must use web_search before answering. Search for the AgentGateway "
                    "open-source project and include the exact marker WEB_SEARCH_OK plus one fact."
                ),
            }],
            "tools": [web_tool],
            "stream": False,
        },
        "code-interpreter": {
            "model": model,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": (
                    "You must use code_execution before answering. Execute Python to calculate "
                    "the sum of squares from 1 through 10 and include CODE_INTERPRETER_OK_385."
                ),
            }],
            "tools": [code_tool],
            "stream": False,
        },
        "combined": {
            "model": model,
            "max_tokens": 768,
            "messages": [{
                "role": "user",
                "content": (
                    "Use both web_search and code_execution before answering. Find one fact about "
                    "AgentGateway and calculate 17 * 23, then include COMBINED_OK_391."
                ),
            }],
            "tools": [web_tool, code_tool],
            "stream": False,
        },
        "programmatic-web-search": {
            "model": model,
            "max_tokens": 768,
            "messages": [{
                "role": "user",
                "content": (
                    "Use code execution to call web_search programmatically for one fact about "
                    "AgentGateway. Do not call web_search directly. Include the exact marker "
                    "PROGRAMMATIC_WEB_SEARCH_OK in the final answer."
                ),
            }],
            "tools": [
                {"type": "code_execution_20260120", "name": "code_execution"},
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "allowed_callers": ["code_execution_20260120"],
                },
            ],
            "stream": False,
        },
        "streaming-tool-runtime": {
            "model": model,
            "max_tokens": 512,
            "messages": [{
                "role": "user",
                "content": (
                    "Use web_search before answering and include STREAMING_TOOL_RUNTIME_OK "
                    "in the final response."
                ),
            }],
            "tools": [web_tool],
            "stream": True,
        },
    }


def expected_marker(case_name: str) -> str:
    return {
        "web-search": "WEB_SEARCH_OK",
        "code-interpreter": "CODE_INTERPRETER_OK_385",
        "combined": "COMBINED_OK_391",
        "programmatic-web-search": "PROGRAMMATIC_WEB_SEARCH_OK",
        "streaming-tool-runtime": "STREAMING_TOOL_RUNTIME_OK",
    }[case_name]


def extract_message_text(response: Any) -> str:
    texts = []
    for block in getattr(response, "content", ()):
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
    return "".join(texts)


def response_mapping(response: Any) -> Mapping[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")
        if isinstance(value, dict):
            return value
    return {}


def contains_reserved_structured_name(value: Any) -> bool:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.startswith("_agentgateway_"):
            return True
        return any(contains_reserved_structured_name(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_reserved_structured_name(item) for item in value)
    return False


def run_sdk_case(client: Any, case_name: str, payload: Mapping[str, Any]) -> str:
    request = dict(payload)
    streaming = bool(request.pop("stream"))
    streamed_text = None
    if streaming:
        with client.messages.stream(**request) as stream:
            deltas = list(stream.text_stream)
            require(bool(deltas), "Anthropic stream omitted a text delta")
            require(
                all(isinstance(delta, str) for delta in deltas),
                "Anthropic stream returned a non-string text delta",
            )
            streamed_text = "".join(deltas)
            response = stream.get_final_message()
    else:
        response = client.messages.create(**request)
    require(getattr(response, "type", None) == "message", f"{case_name} returned wrong type")
    require(
        getattr(response, "stop_reason", None) == "end_turn",
        f"{case_name} did not complete",
    )
    text = extract_message_text(response)
    require(bool(text), f"{case_name} returned no final text")
    require(expected_marker(case_name) in text, f"{case_name} omitted its verification marker")
    if streamed_text is not None:
        require(streamed_text == text, "streamed deltas did not match final message text")
    if case_name == "programmatic-web-search":
        require(
            not contains_reserved_structured_name(response_mapping(response)),
            "programmatic response leaked a reserved internal name",
        )
    return text


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("AGENTGATEWAY_ANTHROPIC_CLIENT_BASE_URL", DEFAULT_BASE_URL),
        help="AgentGateway root URL for the Anthropic SDK (default: %(default)s)",
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
    parser.add_argument(
        "--case",
        action="append",
        choices=CASE_NAMES,
        dest="cases",
        help="run one case; repeat to select several (default: all)",
    )
    parser.add_argument("--list-cases", action="store_true", help="list case names and exit")
    parser.add_argument("--show-output", action="store_true", help="print final model output")
    return parser.parse_args(argv)


def client_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "auth_token": args.api_key,
        "base_url": args.base_url.rstrip("/"),
        "timeout": 180.0,
        "max_retries": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else ())
    if args.list_cases:
        for case_name in CASE_NAMES:
            print(case_name)
        return 0
    try:
        from anthropic import Anthropic
    except ImportError as error:
        raise SDKTestFailure(
            "the anthropic package is required; activate the agent conda environment and install anthropic"
        ) from error
    client = Anthropic(**client_options(args))
    selected_cases = args.cases if args.cases else list(CASE_NAMES)
    payloads = case_payloads(args.model)
    for case_name in selected_cases:
        output = run_sdk_case(client, case_name, payloads[case_name])
        print(f"PASS {case_name}")
        if args.show_output:
            print(output)
    print(f"PASS {len(selected_cases)} Anthropic SDK case(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SDKTestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
