#!/usr/bin/env python3
"""Run AgentGateway's Anthropic Messages tool runtime against real backends."""

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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
    "programmatic-web-search",
    "streaming-tool-runtime",
)
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_UPSTREAM_BASE_URL = "https://api.anthropic.com/v1"
REQUIRED_KEYS = (
    "ANTHROPIC_API_KEY",
    "FC_WEB_SEARCH_URL",
    "FC_WEB_SEARCH_TOKEN",
    "E2B_API_KEY",
    "E2B_API_URL",
    "E2B_DOMAIN",
)
OPTIONAL_KEYS = (
    "AGENTGATEWAY_LIVE_MODEL",
    "AGENTGATEWAY_ANTHROPIC_BASE_URL",
    "AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL",
    "AGENTGATEWAY_UPSTREAM_MODEL",
    "AGENTGATEWAY_UPSTREAM_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL",
)
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
SANDBOX_METRIC_LINE_PATTERN = re.compile(
    r'^agentgateway_tool_runtime_sandbox_operations_total\{([^}]*)\}\s+([0-9]+(?:\.[0-9]+)?)$'
)
METRIC_LABEL_PATTERN = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def agentgateway_base_url(values: Mapping[str, str]) -> str:
    explicit = values.get("AGENTGATEWAY_ANTHROPIC_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    sdk_base_url = values.get("ANTHROPIC_BASE_URL")
    if sdk_base_url:
        sdk_base_url = sdk_base_url.rstrip("/")
        return sdk_base_url if sdk_base_url.endswith("/v1") else sdk_base_url + "/v1"
    return (
        values.get("AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL")
        or values.get("AGENTGATEWAY_UPSTREAM_BASE_URL")
        or DEFAULT_UPSTREAM_BASE_URL
    ).rstrip("/")


class LiveConfiguration:
    def __init__(self, values: Mapping[str, str]) -> None:
        self.anthropic_api_key = values["ANTHROPIC_API_KEY"]
        self.web_search_url = values["FC_WEB_SEARCH_URL"]
        self.web_search_token = values["FC_WEB_SEARCH_TOKEN"]
        self.e2b_api_key = values["E2B_API_KEY"]
        self.e2b_api_url = values["E2B_API_URL"]
        self.e2b_domain = values["E2B_DOMAIN"]
        self.model = (
            values.get("AGENTGATEWAY_LIVE_MODEL")
            or values.get("ANTHROPIC_MODEL")
            or values.get("AGENTGATEWAY_UPSTREAM_MODEL")
            or DEFAULT_MODEL
        )
        self.upstream_base_url = agentgateway_base_url(values)
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
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
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
  - name: anthropic-live
    provider: anthropic
    params:
      apiKey: $ANTHROPIC_API_KEY
      baseUrl: $AGENTGATEWAY_ANTHROPIC_BASE_URL
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
      reference: anthropic-live
"""


def case_payloads(model: str) -> Dict[str, Dict[str, Any]]:
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


def extract_message_text(response: Mapping[str, Any]) -> str:
    content = response.get("content")
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"]
    )


def contains_reserved_structured_name(value: Any) -> bool:
    if isinstance(value, dict):
        name = value.get("name")
        if isinstance(name, str) and name.startswith("_agentgateway_"):
            return True
        return any(contains_reserved_structured_name(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_reserved_structured_name(item) for item in value)
    return False


def sandbox_success_count(metrics: str) -> int:
    count = 0
    for line in metrics.splitlines():
        match = SANDBOX_METRIC_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        labels = {key: value for key, value in METRIC_LABEL_PATTERN.findall(match.group(1))}
        if labels.get("operation") == "execute" and labels.get("outcome") == "success":
            count += int(float(match.group(2)))
    return count


def expected_tool_backends(case_name: str) -> Tuple[Tuple[str, str], ...]:
    if case_name in ("web-search", "streaming-tool-runtime"):
        return (("web_search", "http"),)
    if case_name == "code-interpreter":
        return (("code_interpreter", "e2b"),)
    if case_name == "combined":
        return (("web_search", "http"), ("code_interpreter", "e2b"))
    if case_name == "programmatic-web-search":
        return (("web_search", "http"),)
    raise HarnessFailure("unknown Anthropic live functional case: {}".format(case_name))


def expected_marker(case_name: str) -> str:
    return {
        "web-search": "WEB_SEARCH_OK",
        "code-interpreter": "CODE_INTERPRETER_OK_385",
        "combined": "COMBINED_OK_391",
        "programmatic-web-search": "PROGRAMMATIC_WEB_SEARCH_OK",
        "streaming-tool-runtime": "STREAMING_TOOL_RUNTIME_OK",
    }[case_name]


def final_streaming_response(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    starts = [event for event in events if event.get("type") == "message_start"]
    stops = [event for event in events if event.get("type") == "message_stop"]
    require(len(starts) == 1, "streaming response did not contain one message_start")
    require(len(stops) == 1, "streaming response did not contain one message_stop")
    message = starts[0].get("message")
    require(isinstance(message, dict), "message_start omitted its message object")
    response: Dict[str, Any] = dict(message)
    blocks: Dict[int, Dict[str, Any]] = {}
    stopped_blocks = set()  # type: set[int]
    saw_text_delta = False
    for event in events:
        event_type = event.get("type")
        index = event.get("index")
        if event_type == "content_block_start":
            require(isinstance(index, int), "content_block_start omitted its index")
            require(index not in blocks, "content block started more than once")
            block = event.get("content_block")
            require(isinstance(block, dict), "content_block_start omitted its block")
            blocks[index] = dict(block)
        elif event_type == "content_block_delta":
            require(
                isinstance(index, int) and index in blocks and index not in stopped_blocks,
                "content delta occurred outside its block",
            )
            delta = event.get("delta")
            require(isinstance(delta, dict), "content delta omitted its payload")
            if delta.get("type") == "text_delta":
                text = delta.get("text")
                require(isinstance(text, str), "text delta omitted text")
                blocks[index]["text"] = str(blocks[index].get("text", "")) + text
                saw_text_delta = True
        elif event_type == "content_block_stop":
            require(
                isinstance(index, int) and index in blocks and index not in stopped_blocks,
                "content_block_stop did not match one open block",
            )
            stopped_blocks.add(index)
        elif event_type == "message_delta":
            delta = event.get("delta")
            require(isinstance(delta, dict), "message_delta omitted its payload")
            response.update(delta)
            usage = event.get("usage")
            if isinstance(usage, dict):
                merged_usage = dict(response.get("usage", {}))
                merged_usage.update(usage)
                response["usage"] = merged_usage
    require(saw_text_delta, "streaming response omitted a text delta")
    require(stopped_blocks == set(blocks), "streaming response left a content block open")
    response["content"] = [blocks[index] for index in sorted(blocks)]
    return response


def run_case(
    base_url: str,
    stats_port: int,
    token: str,
    case_name: str,
    payload: Mapping[str, Any],
) -> str:
    before_metrics = fetch_metrics(stats_port)
    before = tool_success_counts(before_metrics)
    before_sandbox = sandbox_success_count(before_metrics)
    if case_name == "streaming-tool-runtime":
        status, events = post_sse(base_url + "/v1/messages", token, payload)
        require(status == 200, "{} returned HTTP {}".format(case_name, status))
        response = final_streaming_response(events)
    else:
        status, response = post_json(base_url + "/v1/messages", token, payload)
    require(status == 200, "{} returned HTTP {}".format(case_name, status))
    require(response.get("type") == "message", "{} returned the wrong response type".format(case_name))
    require(response.get("stop_reason") == "end_turn", "{} did not complete".format(case_name))
    text = extract_message_text(response)
    require(bool(text), "{} returned no final text".format(case_name))
    require(
        expected_marker(case_name) in text,
        "{} final output omitted its verification marker".format(case_name),
    )
    if case_name == "programmatic-web-search":
        require(
            not contains_reserved_structured_name(response),
            "programmatic response leaked a reserved internal name",
        )
    after_metrics = fetch_metrics(stats_port)
    after = tool_success_counts(after_metrics)
    for tool, backend in expected_tool_backends(case_name):
        metric_key = (tool, backend)
        require(
            after.get(metric_key, 0) > before.get(metric_key, 0),
            "{} did not record a successful {}/{} backend call".format(case_name, tool, backend),
        )
    if case_name == "programmatic-web-search":
        require(
            sandbox_success_count(after_metrics) > before_sandbox,
            "programmatic-web-search did not record a successful Sandbox execution",
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
        with tempfile.TemporaryDirectory(prefix="agentgateway-anthropic-live-functional-") as temp_dir:
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
                environment["AGENTGATEWAY_ANTHROPIC_BASE_URL"] = configuration.upstream_base_url
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
                    payloads = case_payloads(configuration.model)
                    for case_name in selected_cases:
                        text = run_case(
                            base_url, stats_port, client_token, case_name, payloads[case_name]
                        )
                        if show_output:
                            print(text)
                        print("PASS {}".format(case_name))
                    require(process.poll() is None, "AgentGateway exited during live cases")
                finally:
                    stop_process(process)
                    process = None
    finally:
        close_reservations(reservations)
        if process is not None:
            stop_process(process)


def run(binary: Path, dotenv: Path, selected_cases: Sequence[str], show_output: bool) -> None:
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
        help="print final model output for each selected case",
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
    print("PASS {} Anthropic live functional case(s)".format(len(selected_cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
