#!/usr/bin/env python3
"""Run AgentGateway against real LLM, Web Search, E2B, and remote MCP backends."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from live_functional_test_harness import (
    GatewayStartupFailure,
    HarnessFailure,
    close_reservations,
    fetch_metrics,
    live_child_environment,
    load_known_dotenv,
    post_json,
    post_sse,
    require,
    reserve_loopback_ports,
    run_with_startup_retries,
    stop_process,
    tool_success_counts,
    wait_for_readiness,
)


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
DEFAULT_MODEL = "qwen3.6-flash"
DEFAULT_UPSTREAM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REQUIRED_KEYS = (
    "OPENAI_API_KEY",
    "FC_WEB_SEARCH_URL",
    "FC_WEB_SEARCH_TOKEN",
    "E2B_API_KEY",
    "E2B_API_URL",
    "E2B_DOMAIN",
    "FC_WEATHER_MCP_URL",
    "FC_WEATHER_MCP_TOKEN",
)
OPTIONAL_KEYS = (
    "AGENTGATEWAY_LIVE_MODEL",
    "AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL",
    "AGENTGATEWAY_UPSTREAM_MODEL",
    "AGENTGATEWAY_UPSTREAM_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
)
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


class LiveConfiguration:
    def __init__(self, values: Mapping[str, str]) -> None:
        self.openai_api_key = values["OPENAI_API_KEY"]
        self.web_search_url = values["FC_WEB_SEARCH_URL"]
        self.web_search_token = values["FC_WEB_SEARCH_TOKEN"]
        self.e2b_api_key = values["E2B_API_KEY"]
        self.e2b_api_url = values["E2B_API_URL"]
        self.e2b_domain = values["E2B_DOMAIN"]
        self.weather_mcp_url = values["FC_WEATHER_MCP_URL"]
        self.weather_mcp_token = values["FC_WEATHER_MCP_TOKEN"]
        self.model = (
            values.get("AGENTGATEWAY_LIVE_MODEL")
            or values.get("OPENAI_MODEL")
            or values.get("AGENTGATEWAY_UPSTREAM_MODEL")
            or DEFAULT_MODEL
        )
        self.upstream_base_url = (
            values.get("AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL")
            or values.get("OPENAI_BASE_URL")
            or values.get("AGENTGATEWAY_UPSTREAM_BASE_URL")
            or DEFAULT_UPSTREAM_BASE_URL
        )
        require(bool(MODEL_NAME_PATTERN.fullmatch(self.model)), "live model name is invalid")

    @classmethod
    def load(cls, dotenv: Path, environment: Mapping[str, str]) -> "LiveConfiguration":
        values = {
            key: value
            for key in REQUIRED_KEYS + OPTIONAL_KEYS
            if (value := environment.get(key))
        }
        load_known_dotenv(dotenv, values, REQUIRED_KEYS + OPTIONAL_KEYS)
        missing = sorted(key for key in REQUIRED_KEYS if not values.get(key))
        if missing:
            raise HarnessFailure("missing required configuration: " + ", ".join(missing))
        return cls(values)

    def child_environment(self, client_token: str) -> Dict[str, str]:
        return live_child_environment({
            "OPENAI_API_KEY": self.openai_api_key,
            "FC_WEB_SEARCH_URL": self.web_search_url,
            "FC_WEB_SEARCH_TOKEN": self.web_search_token,
            "E2B_API_KEY": self.e2b_api_key,
            "E2B_API_URL": self.e2b_api_url,
            "E2B_DOMAIN": self.e2b_domain,
            "AGENTGATEWAY_LIVE_CLIENT_TOKEN": client_token,
        })


def functional_config(gateway_port: int, readiness_port: int, stats_port: int = 0) -> str:
    return f"""\
# yaml-language-server: $schema=https://agentgateway.dev/schema/config
config:
  adminAddr: 127.0.0.1:0
  readinessAddr: 127.0.0.1:{readiness_port}
  statsAddr: 127.0.0.1:{stats_port}

gateways:
  default:
    port: {gateway_port}

llm:
  gateways: default
  policies:
    apiKey:
      keys:
      - key: $AGENTGATEWAY_LIVE_CLIENT_TOKEN
      mode: strict
      location:
        header:
          name: authorization
          prefix: 'Bearer '
  providers:
  - name: bailian
    provider: openAI
    params:
      apiKey: $OPENAI_API_KEY
      baseUrl: $AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL
  toolRuntime:
    maxRounds: 8
    maxToolCalls: 16
    maxParallelToolCalls: 4
    totalTimeout: 120s
    maxArgumentsBytes: 65536
    maxOutputBytes: 1048576
    tools:
    - name: web_search
      builtin: webSearch
      backend:
        type: http
        url: $FC_WEB_SEARCH_URL
        timeout: 20s
        bearerToken: $FC_WEB_SEARCH_TOKEN
    - name: code_interpreter
      builtin: codeInterpreter
      backend:
        type: e2b
        apiKey: $E2B_API_KEY
        apiUrl: $E2B_API_URL
        domain: $E2B_DOMAIN
        timeout: 120s
  models:
  - name: $AGENTGATEWAY_LIVE_MODEL
    provider:
      reference: bailian
"""


def case_payloads(
    model: str, weather_mcp_url: str, weather_mcp_token: str
) -> Dict[str, Dict[str, Any]]:
    code_tool = {"type": "code_interpreter", "container": {"type": "auto"}}
    web_tool = {"type": "web_search"}
    programmatic_code_tool = {
        **code_tool,
        "allowed_callers": ["programmatic"],
    }
    programmatic_web_tool = {
        **web_tool,
        "allowed_callers": ["programmatic"],
    }
    weather_mcp_tool = {
        "type": "mcp",
        "server_label": "weather",
        "server_url": weather_mcp_url,
        "authorization": weather_mcp_token,
        "allowed_tools": ["get_forecast", "get_current_weather"],
        "require_approval": "auto",
    }
    programmatic_weather_mcp_tool = {
        "type": "mcp",
        "server_label": "weather",
        "server_url": weather_mcp_url,
        "authorization": weather_mcp_token,
        "allowed_tools": ["get_forecast"],
        "allowed_callers": ["programmatic"],
        "require_approval": "never",
    }
    deferred_weather_mcp_tool = {
        "type": "mcp",
        "server_label": "weather",
        "server_url": weather_mcp_url,
        "authorization": weather_mcp_token,
        "allowed_tools": ["get_forecast", "get_current_weather"],
        "require_approval": "never",
        "defer_loading": True,
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
                programmatic_web_tool,
                programmatic_code_tool,
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
                programmatic_weather_mcp_tool,
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
            # No tool_choice: forcing a tool on a deferred server injects its declaration
            # eagerly, which is the one path that would let this case pass without a search.
            "tools": [{"type": "tool_search"}, deferred_weather_mcp_tool],
            "enable_thinking": False,
            "stream": False,
        },
    }


def extract_output_text(response: Mapping[str, Any]) -> str:
    texts: List[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
    return "\n".join(texts)


PROGRAM_SOURCE_MESSAGE = "programmatic tool call program source"


def extract_program_sources(logs: str) -> List[str]:
    sources: List[str] = []
    for line in logs.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("message") != PROGRAM_SOURCE_MESSAGE:
            continue
        code = event.get("code")
        if isinstance(code, str) and code:
            sources.append(code)
    return sources


def read_program_sources(log_path: Path, offset: int) -> Tuple[List[str], int]:
    try:
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError as error:
        raise HarnessFailure("could not read the AgentGateway log") from error
    # The child is still appending, so stop at the last newline and leave the rest for next time.
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    logs = data[: end + 1].decode("utf-8", "replace")
    return extract_program_sources(logs), offset + end + 1


def print_case_output(case_name: str, text: str, program_sources: Sequence[str]) -> None:
    if case_name == "programmatic-mcp-weather":
        require(
            bool(program_sources),
            "programmatic weather case did not capture generated program source",
        )
    for index, code in enumerate(program_sources, start=1):
        print("PROGRAM SOURCE {} (Python)".format(index))
        print(code)
        print("END PROGRAM SOURCE {}".format(index))
    print(text)


def expected_tool_backends(case_name: str) -> Tuple[Tuple[str, str], ...]:
    if case_name in ("web-search", "streaming-tool-runtime"):
        return (("web_search", "http"),)
    if case_name == "code-interpreter":
        return (("code_interpreter", "e2b"),)
    if case_name in (
        "remote-mcp-weather",
        "programmatic-mcp-weather",
        "tool-search-weather-mcp",
    ):
        return (("remote_mcp", "remote_mcp"),)
    if case_name in ("combined", "programmatic-server-tools"):
        return (
            ("web_search", "http"),
            ("code_interpreter", "e2b"),
        )
    raise HarnessFailure("unknown live functional case: {}".format(case_name))


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


def final_streaming_response(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    event_types = [event.get("type") for event in events]
    require("response.created" in event_types, "streaming response omitted response.created")
    require(
        "response.output_text.delta" in event_types,
        "streaming response omitted response.output_text.delta",
    )
    completed = [event for event in events if event.get("type") == "response.completed"]
    require(len(completed) == 1, "streaming response did not contain one response.completed")
    response = completed[0].get("response")
    require(isinstance(response, dict), "response.completed omitted its response object")
    return response


def require_deferred_tool_contract(
    response: Mapping[str, Any], payload: Mapping[str, Any]
) -> None:
    expected = []
    for declaration in payload["tools"]:
        declaration = dict(declaration)
        # The gateway strips the MCP credential before echoing; everything else round-trips.
        declaration.pop("authorization", None)
        expected.append(declaration)
    require(
        response.get("tools") == expected,
        "the deferred tool declarations did not round-trip to the client",
    )
    serialized = json.dumps(response, separators=(",", ":"), sort_keys=True)
    require(
        "_agentgateway" not in serialized,
        "a reserved internal name reached the client",
    )


def run_case(
    base_url: str,
    stats_port: int,
    token: str,
    case_name: str,
    payload: Mapping[str, Any],
) -> str:
    before = tool_success_counts(fetch_metrics(stats_port))
    if case_name == "streaming-tool-runtime":
        status, events = post_sse(base_url + "/v1/responses", token, payload)
        require(status == 200, "{} returned HTTP {}".format(case_name, status))
        response = final_streaming_response(events)
    else:
        status, response = post_json(base_url + "/v1/responses", token, payload)
    require(status == 200, "{} returned HTTP {}".format(case_name, status))
    require(response.get("status") == "completed", "{} did not complete".format(case_name))
    text = extract_output_text(response)
    require(bool(text), "{} returned no final output text".format(case_name))
    require(
        expected_marker(case_name) in text,
        "{} final output omitted its verification marker".format(case_name),
    )
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
        # The declaration was withheld from round one, so a successful remote_mcp call below is
        # only reachable through a search that injected it. Requiring a measured value keeps the
        # model from passing on the marker alone.
        require(
            bool(re.search(r"-?\d+(?:\.\d+)?\s*(?:°|degrees?\b|C\b|F\b)", text)),
            "tool search weather output omitted an observed temperature",
        )
    after = tool_success_counts(fetch_metrics(stats_port))
    for tool, backend in expected_tool_backends(case_name):
        metric_key = (tool, backend)
        require(
            after.get(metric_key, 0) > before.get(metric_key, 0),
            "{} did not record a successful {}/{} backend call".format(
                case_name, tool, backend
            ),
        )
    return text


def run_gateway_attempt(
    binary: Path,
    configuration: LiveConfiguration,
    selected_cases: Sequence[str],
    show_output: bool,
) -> None:
    ports, reservations = reserve_loopback_ports(3)
    gateway_port, readiness_port, stats_port = ports
    client_token = secrets.token_urlsafe(32)
    process: Optional[subprocess.Popen[Any]] = None
    try:
        with tempfile.TemporaryDirectory(prefix="agentgateway-live-functional-") as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                functional_config(gateway_port, readiness_port, stats_port), encoding="utf-8"
            )
            log_path = temp_path / "agentgateway.log"
            with log_path.open("wb") as child_log:
                close_reservations(reservations)
                reservations = []
                environment = configuration.child_environment(client_token)
                environment["AGENTGATEWAY_LIVE_MODEL"] = configuration.model
                environment[
                    "AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL"
                ] = configuration.upstream_base_url
                if show_output:
                    environment["LOG_FORMAT"] = "json"
                    environment["RUST_LOG"] = (
                        "info,agentgateway::llm::tool_runtime::runner=trace"
                    )
                process = subprocess.Popen(
                    [str(binary), "--file", str(config_path)],
                    cwd=temp_dir,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=child_log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    wait_for_readiness(process, readiness_port)
                    try:
                        fetch_metrics(stats_port)
                    except HarnessFailure as error:
                        raise GatewayStartupFailure(
                            "AgentGateway metrics endpoint was not ready"
                        ) from error
                    base_url = "http://127.0.0.1:{}".format(gateway_port)
                    payloads = case_payloads(
                        configuration.model,
                        configuration.weather_mcp_url,
                        configuration.weather_mcp_token,
                    )
                    log_offset = 0
                    for case_name in selected_cases:
                        text = run_case(
                            base_url,
                            stats_port,
                            client_token,
                            case_name,
                            payloads[case_name],
                        )
                        if show_output:
                            sources, log_offset = read_program_sources(log_path, log_offset)
                            print_case_output(case_name, text, sources)
                        print("PASS {}".format(case_name))
                    require(process.poll() is None, "AgentGateway exited during live cases")
                finally:
                    stop_process(process)
                    process = None
    finally:
        close_reservations(reservations)
        if process is not None:
            stop_process(process)


def run(
    binary: Path,
    dotenv: Path,
    selected_cases: Sequence[str],
    show_output: bool,
) -> None:
    binary = binary.expanduser().resolve()
    require(binary.is_file(), "release binary path is not a file")
    require(os.access(str(binary), os.X_OK), "release binary path is not executable")
    configuration = LiveConfiguration.load(dotenv.expanduser().resolve(), os.environ)
    run_with_startup_retries(
        lambda: run_gateway_attempt(binary, configuration, selected_cases, show_output)
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default="./target/release/agentgateway",
        help="path to the already-built AgentGateway executable",
    )
    parser.add_argument(
        "--env-file",
        default=str(repository_root() / ".env"),
        help="dotenv file used to fill known backend variables",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=CASE_NAMES,
        dest="cases",
        help="run one case; repeat to select several (default: all)",
    )
    parser.add_argument("--list-cases", action="store_true", help="list case names and exit")
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="print final model output and Weather PTC-generated Python code",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.list_cases:
        for case_name in CASE_NAMES:
            print(case_name)
        return 0
    selected_cases = args.cases if args.cases else list(CASE_NAMES)
    try:
        run(Path(args.binary), Path(args.env_file), selected_cases, args.show_output)
    except HarnessFailure as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        return 1
    except (OSError, subprocess.SubprocessError) as error:
        print(
            "FAIL: harness infrastructure error ({}); verify the binary, loopback, "
            "temporary-directory, and network access".format(type(error).__name__),
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("FAIL: interrupted; child cleanup completed", file=sys.stderr)
        return 130
    print("PASS {} live functional case(s)".format(len(selected_cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
