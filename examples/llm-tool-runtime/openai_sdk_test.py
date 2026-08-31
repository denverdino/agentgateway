#!/usr/bin/env python3
"""Exercise AgentGateway tool-runtime cases through the OpenAI Python SDK."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


CASE_NAMES = (
    "web-search",
    "code-interpreter",
    "combined",
    "programmatic-server-tools",
    "programmatic-mcp-weather",
    "streaming-tool-runtime",
    "remote-mcp-weather",
    "tool-search-weather-mcp",
)
DEFAULT_BASE_URL = "http://127.0.0.1:4000/v1"
DEFAULT_MODEL = "qwen3.8-flash"
WEATHER_CASES = {
    "programmatic-mcp-weather",
    "remote-mcp-weather",
    "tool-search-weather-mcp",
}


class SDKTestFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SDKTestFailure(message)


def case_payloads(
    model: str, weather_mcp_url: str, weather_mcp_token: str
) -> dict[str, dict[str, Any]]:
    code_tool = {"type": "code_interpreter", "container": {"type": "auto"}}
    web_tool = {"type": "web_search"}
    weather_mcp_tool = {
        "type": "mcp",
        "server_label": "weather",
        "server_url": weather_mcp_url,
        "authorization": weather_mcp_token,
        "allowed_tools": ["get_forecast", "get_current_weather"],
        "require_approval": "auto",
    }
    return {
        "web-search": {
            "model": model,
            "input": (
                "You must use web_search before answering. Search for the AgentGateway "
                "open-source project and answer with the exact marker WEB_SEARCH_OK plus "
                "one fact supported by the search result."
            ),
            "tools": [web_tool],
            "stream": False,
        },
        "code-interpreter": {
            "model": model,
            "input": (
                "You must use code_interpreter before answering. Execute Python to calculate "
                "the sum of the squares from 1 through 10 and print CODE_INTERPRETER_OK_385. "
                "Include that exact marker in the final answer."
            ),
            "tools": [code_tool],
            "stream": False,
        },
        "combined": {
            "model": model,
            "input": (
                "Before answering, you must call both tools: use web_search to find one fact "
                "about the AgentGateway open-source project, and use code_interpreter to "
                "calculate 17 * 23. Call both in the same turn if possible. Then answer with "
                "the exact marker COMBINED_OK_391 and the searched fact."
            ),
            "tools": [web_tool, code_tool],
            "parallel_tool_calls": True,
            "stream": False,
        },
        "programmatic-server-tools": {
            "model": model,
            "input": (
                "Use programmatic tool calling to run one program that calls both tools. "
                "Search the web for one fact about the AgentGateway open-source project, "
                "and execute Python to calculate 29 * 31. After both programmatic calls "
                "succeed, answer with the exact marker PROGRAMMATIC_SERVER_TOOLS_OK_899 "
                "and the searched fact. Do not call either tool directly."
            ),
            "tools": [
                {**web_tool, "allowed_callers": ["programmatic"]},
                {**code_tool, "allowed_callers": ["programmatic"]},
                {"type": "programmatic_tool_calling"},
            ],
            "parallel_tool_calls": True,
            "stream": False,
        },
        "programmatic-mcp-weather": {
            "model": model,
            "input": (
                "Use programmatic tool calling to run one Python program. In that program, "
                "call weather.get_forecast for Shanghai with days=3. From the returned "
                "3-day forecast, use Python to report the date and value with the highest "
                "daily maximum temperature, and the date and value with the lowest daily "
                "minimum temperature. If either result is tied, choose the earliest date. "
                "Do not call the MCP tool directly. Finish with the exact marker "
                "PROGRAMMATIC_MCP_WEATHER_3DAY_OK."
            ),
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": weather_mcp_url,
                    "authorization": weather_mcp_token,
                    "allowed_tools": ["get_forecast"],
                    "allowed_callers": ["programmatic"],
                    "require_approval": "never",
                },
                {"type": "programmatic_tool_calling"},
            ],
            "parallel_tool_calls": True,
            "stream": False,
        },
        "streaming-tool-runtime": {
            "model": model,
            "input": (
                "Use web_search before answering and include STREAMING_TOOL_RUNTIME_OK "
                "in the final response."
            ),
            "tools": [web_tool],
            "stream": True,
        },
        "remote-mcp-weather": {
            "model": model,
            "input": (
                "You must use the weather MCP server before answering. Get the current "
                "weather or forecast for Beijing, China. Summarize the observed weather "
                "data and include the exact marker WEATHER_MCP_BEIJING_OK in the final answer."
            ),
            "tools": [weather_mcp_tool],
            "tool_choice": {
                "type": "mcp",
                "server_label": "weather",
                "name": "get_current_weather",
            },
            "enable_thinking": False,
            "stream": False,
        },
        "tool-search-weather-mcp": {
            "model": model,
            "input": (
                "No weather tool is loaded yet. Use the tool search function to find a "
                "weather tool, then call the tool it returns to get the current weather "
                "for Shanghai, China. Summarize the observed weather data and include the "
                "exact marker TOOL_SEARCH_WEATHER_MCP_OK in the final answer."
            ),
            "tools": [
                {"type": "tool_search"},
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": weather_mcp_url,
                    "authorization": weather_mcp_token,
                    "allowed_tools": ["get_forecast", "get_current_weather"],
                    "require_approval": "never",
                    "defer_loading": True,
                },
            ],
            "enable_thinking": False,
            "stream": False,
        },
    }


def expected_marker(case_name: str) -> str:
    return {
        "web-search": "WEB_SEARCH_OK",
        "code-interpreter": "CODE_INTERPRETER_OK_385",
        "combined": "COMBINED_OK_391",
        "programmatic-server-tools": "PROGRAMMATIC_SERVER_TOOLS_OK_899",
        "programmatic-mcp-weather": "PROGRAMMATIC_MCP_WEATHER_3DAY_OK",
        "streaming-tool-runtime": "STREAMING_TOOL_RUNTIME_OK",
        "remote-mcp-weather": "WEATHER_MCP_BEIJING_OK",
        "tool-search-weather-mcp": "TOOL_SEARCH_WEATHER_MCP_OK",
    }[case_name]


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


def response_mapping(response: Any) -> Mapping[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_none=True)
        if isinstance(value, dict):
            return value
    return {}


def require_deferred_tool_contract(response: Any, payload: Mapping[str, Any]) -> None:
    expected = []
    for declaration in payload["tools"]:
        sanitized = dict(declaration)
        sanitized.pop("authorization", None)
        expected.append(sanitized)
    value = response_mapping(response)
    require(
        value.get("tools") == expected,
        "the deferred tool declarations did not round-trip to the client",
    )
    require(
        "_agentgateway" not in json.dumps(value, separators=(",", ":"), sort_keys=True),
        "a reserved internal name reached the client",
    )


def sdk_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    if "enable_thinking" in request:
        request["extra_body"] = {"enable_thinking": request.pop("enable_thinking")}
    return request


def run_sdk_case(client: Any, case_name: str, payload: Mapping[str, Any]) -> str:
    result = client.responses.create(**sdk_request(payload))
    streamed_text = None
    if payload.get("stream"):
        streamed_text, response = consume_stream(result)
    else:
        response = result
    require(response is not None, "response.completed omitted its response")
    require(getattr(response, "status", None) == "completed", f"{case_name} did not complete")
    text = getattr(response, "output_text", None)
    require(isinstance(text, str) and bool(text), f"{case_name} returned no output_text")
    require(expected_marker(case_name) in text, f"{case_name} omitted its verification marker")
    if streamed_text is not None:
        require(streamed_text == text, "streamed deltas did not match final output_text")
    if case_name == "programmatic-mcp-weather":
        dates = set(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text))
        lowered = text.lower()
        require(
            bool(dates)
            and ("highest" in lowered or "hottest" in lowered or "最高" in text)
            and ("lowest" in lowered or "coldest" in lowered or "最低" in text),
            "programmatic weather output omitted the highest or lowest temperature day",
        )
    if case_name == "tool-search-weather-mcp":
        require_deferred_tool_contract(response, payload)
        require(
            bool(re.search(r"-?\d+(?:\.\d+)?\s*(?:°|degrees?\b|C\b|F\b)", text)),
            "tool search weather output omitted an observed temperature",
        )
    return text


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
    parser.add_argument(
        "--weather-mcp-url",
        default=os.environ.get("FC_WEATHER_MCP_URL", ""),
        help="deployed weather MCP URL; FC_WEATHER_MCP_URL is used when set",
    )
    parser.add_argument(
        "--weather-mcp-token",
        default=os.environ.get("FC_WEATHER_MCP_TOKEN", ""),
        help="weather MCP bearer token; FC_WEATHER_MCP_TOKEN is used when set",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else ())
    if args.list_cases:
        for case_name in CASE_NAMES:
            print(case_name)
        return 0
    selected_cases = args.cases if args.cases else list(CASE_NAMES)
    if WEATHER_CASES.intersection(selected_cases):
        require(bool(args.weather_mcp_url), "weather cases require FC_WEATHER_MCP_URL")
        require(bool(args.weather_mcp_token), "weather cases require FC_WEATHER_MCP_TOKEN")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SDKTestFailure(
            "the openai package is required; activate the agent conda environment and install openai"
        ) from error
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/") + "/",
        timeout=180.0,
        max_retries=0,
    )
    payloads = case_payloads(args.model, args.weather_mcp_url, args.weather_mcp_token)
    for case_name in selected_cases:
        output = run_sdk_case(client, case_name, payloads[case_name])
        print(f"PASS {case_name}")
        if args.show_output:
            print(output)
    print(f"PASS {len(selected_cases)} OpenAI SDK case(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except SDKTestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
