#!/usr/bin/env python3
"""Run AgentGateway against real LLM, Web Search, E2B, and remote MCP backends."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CASE_NAMES = (
    "web-search",
    "code-interpreter",
    "combined",
    "streaming-tool-runtime",
    "remote-mcp-weather",
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
OPTIONAL_KEYS = ("AGENTGATEWAY_LIVE_MODEL", "AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL")
MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024
STARTUP_TIMEOUT_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 150.0
MAX_GATEWAY_START_ATTEMPTS = 3
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")
METRIC_LINE_PATTERN = re.compile(
    r'^agentgateway_tool_runtime_calls_total\{([^}]*)\}\s+([0-9]+(?:\.[0-9]+)?)$'
)
METRIC_LABEL_PATTERN = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class HarnessFailure(Exception):
    """A functional-test failure whose message contains no backend content."""


class GatewayStartupFailure(HarnessFailure):
    """A retryable AgentGateway startup failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessFailure(message)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    return value


def load_known_dotenv(path: Path, values: Dict[str, str]) -> None:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return
    allowed = set(REQUIRED_KEYS + OPTIONAL_KEYS)
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or key in values:
            continue
        value = _dotenv_value(raw_value)
        if value:
            values[key] = value


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
        self.model = values.get("AGENTGATEWAY_LIVE_MODEL", DEFAULT_MODEL)
        self.upstream_base_url = values.get(
            "AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL", DEFAULT_UPSTREAM_BASE_URL
        )
        require(bool(MODEL_NAME_PATTERN.fullmatch(self.model)), "live model name is invalid")

    @classmethod
    def load(cls, dotenv: Path, environment: Mapping[str, str]) -> "LiveConfiguration":
        values = {
            key: value
            for key in REQUIRED_KEYS + OPTIONAL_KEYS
            if (value := environment.get(key))
        }
        load_known_dotenv(dotenv, values)
        missing = sorted(key for key in REQUIRED_KEYS if not values.get(key))
        if missing:
            raise HarnessFailure("missing required configuration: " + ", ".join(missing))
        return cls(values)

    def child_environment(self, client_token: str) -> Dict[str, str]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "NO_COLOR": "1",
            "RUST_BACKTRACE": "0",
            "RUST_LOG": "error",
            "OPENAI_API_KEY": self.openai_api_key,
            "FC_WEB_SEARCH_URL": self.web_search_url,
            "FC_WEB_SEARCH_TOKEN": self.web_search_token,
            "E2B_API_KEY": self.e2b_api_key,
            "E2B_API_URL": self.e2b_api_url,
            "E2B_DOMAIN": self.e2b_domain,
            "AGENTGATEWAY_LIVE_CLIENT_TOKEN": client_token,
        }
        for key in ("TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"):
            if os.environ.get(key):
                environment[key] = os.environ[key]
        return environment


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


def tool_success_counts(metrics: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for line in metrics.splitlines():
        match = METRIC_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        labels = {key: value for key, value in METRIC_LABEL_PATTERN.findall(match.group(1))}
        if (
            labels.get("outcome") != "success"
            or "tool" not in labels
            or "backend" not in labels
        ):
            continue
        counts[(labels["tool"], labels["backend"])] = int(float(match.group(2)))
    return counts


def reserve_loopback_ports(count: int) -> Tuple[List[int], List[socket.socket]]:
    reservations: List[socket.socket] = []
    try:
        for _ in range(count):
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reservation.bind(("127.0.0.1", 0))
            reservation.listen(1)
            reservations.append(reservation)
        ports = [int(item.getsockname()[1]) for item in reservations]
        require(len(set(ports)) == count, "loopback port allocation returned duplicates")
        return ports, reservations
    except Exception:
        close_reservations(reservations)
        raise


def close_reservations(reservations: Iterable[socket.socket]) -> None:
    for reservation in reservations:
        reservation.close()


def wait_for_readiness(process: subprocess.Popen[Any], readiness_port: int) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    url = "http://127.0.0.1:{}/healthz/ready".format(readiness_port)
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise GatewayStartupFailure(
                "AgentGateway exited before readiness with status {} (logs withheld)".format(
                    exit_code
                )
            )
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.05)
    raise GatewayStartupFailure("AgentGateway readiness timed out")


def read_bounded(response: Any) -> bytes:
    body = response.read(MAX_HTTP_BODY_BYTES + 1)
    require(len(body) <= MAX_HTTP_BODY_BYTES, "HTTP response exceeded the body limit")
    return body


def post_json(url: str, token: str, payload: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "authorization": "Bearer " + token,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = read_bounded(response)
    except urllib.error.HTTPError as error:
        status = error.code
        body = read_bounded(error)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("Responses request failed at the HTTP transport") from error
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessFailure("Responses endpoint returned invalid JSON") from error
    require(isinstance(value, dict), "Responses endpoint returned a non-object JSON value")
    return status, value


def post_sse(url: str, token: str, payload: Mapping[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={
            "authorization": "Bearer " + token,
            "content-type": "application/json",
            "accept": "text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            content_type = response.headers.get_content_type()
            body = read_bounded(response)
    except urllib.error.HTTPError as error:
        status = error.code
        content_type = error.headers.get_content_type()
        body = read_bounded(error)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("Responses streaming request failed at the HTTP transport") from error
    require(content_type == "text/event-stream", "streaming response was not SSE")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HarnessFailure("Responses SSE endpoint returned non-UTF-8 content") from error
    events: List[Dict[str, Any]] = []
    for frame in text.split("\n\n"):
        event_name = next(
            (line[len("event: ") :] for line in frame.splitlines() if line.startswith("event: ")),
            None,
        )
        data = next(
            (line[len("data: ") :] for line in frame.splitlines() if line.startswith("data: ")),
            None,
        )
        if data is None:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise HarnessFailure("Responses endpoint returned invalid SSE JSON") from error
        require(isinstance(event, dict), "Responses SSE data was not an object")
        require(event.get("type") == event_name, "Responses SSE event name did not match its type")
        events.append(event)
    return status, events


def fetch_metrics(stats_port: int) -> str:
    url = "http://127.0.0.1:{}/metrics".format(stats_port)
    try:
        with urllib.request.urlopen(url, timeout=3.0) as response:
            body = read_bounded(response)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("metrics request failed at the HTTP transport") from error
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HarnessFailure("metrics endpoint returned non-UTF-8 content") from error


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_with_startup_retries(
    attempt: Callable[[], None], max_attempts: int = MAX_GATEWAY_START_ATTEMPTS
) -> None:
    require(max_attempts > 0, "maximum startup attempts must be greater than zero")
    for attempt_index in range(max_attempts):
        try:
            attempt()
            return
        except GatewayStartupFailure as error:
            if attempt_index + 1 == max_attempts:
                raise HarnessFailure(
                    "AgentGateway startup failed after {} attempts: {}".format(
                        max_attempts, error
                    )
                ) from error


def expected_tool_backends(case_name: str) -> Tuple[Tuple[str, str], ...]:
    if case_name in ("web-search", "streaming-tool-runtime"):
        return (("web_search", "http"),)
    if case_name == "code-interpreter":
        return (("code_interpreter", "e2b"),)
    if case_name == "remote-mcp-weather":
        return (("remote_mcp", "remote_mcp"),)
    return (
        ("web_search", "http"),
        ("code_interpreter", "e2b"),
    )


def expected_marker(case_name: str) -> str:
    return {
        "web-search": "WEB_SEARCH_OK",
        "code-interpreter": "CODE_INTERPRETER_OK_385",
        "combined": "COMBINED_OK_391",
        "streaming-tool-runtime": "STREAMING_TOOL_RUNTIME_OK",
        "remote-mcp-weather": "WEATHER_MCP_BEIJING_OK",
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


def run_case(
    base_url: str,
    stats_port: int,
    token: str,
    case_name: str,
    payload: Mapping[str, Any],
    show_output: bool,
) -> None:
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
    after = tool_success_counts(fetch_metrics(stats_port))
    for tool, backend in expected_tool_backends(case_name):
        metric_key = (tool, backend)
        require(
            after.get(metric_key, 0) > before.get(metric_key, 0),
            "{} did not record a successful {}/{} backend call".format(
                case_name, tool, backend
            ),
        )
    print("PASS {}".format(case_name))
    if show_output:
        print(text)


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
                    for case_name in selected_cases:
                        run_case(
                            base_url,
                            stats_port,
                            client_token,
                            case_name,
                            payloads[case_name],
                            show_output,
                        )
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
        help="print final model output after each passing case",
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
