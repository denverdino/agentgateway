#!/usr/bin/env python3
"""Hermetic functional tests for the AgentGateway Anthropic Messages tool runtime."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from functional_test_harness import (
    FUNCTIONAL_CLIENT_TOKEN,
    MAX_GATEWAY_START_ATTEMPTS,
    GatewayStartupFailure,
    HarnessFailure,
    MockServer,
    QuietHandler,
    close_reservations,
    post_json,
    post_status,
    require,
    reserve_distinct_loopback_ports,
    safe_child_environment,
    stop_process,
    wait_for_readiness,
)


CASE_NAMES = ("messages-programmatic",)


class MockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_requests: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.catalog_requests: List[Dict[str, Any]] = []
        self.sandbox_requests: List[Dict[str, Any]] = []
        self.pending_sandboxes = 0
        self.active_sandbox = False
        self.context_count = 0
        self.errors: List[str] = []

    def record_model(self, scenario: str, request: Dict[str, Any]) -> None:
        with self.lock:
            self.model_requests[scenario].append(request)
            if scenario == "messages-programmatic" and not has_tool_result(request):
                self.pending_sandboxes += 2

    def begin_sandbox(self) -> str:
        with self.lock:
            require(self.pending_sandboxes > 0, "E2B create had no pending Messages program")
            require(not self.active_sandbox, "E2B Sandbox lifecycles overlapped")
            self.pending_sandboxes -= 1
            self.active_sandbox = True
            return "functional-messages-programmatic"

    def next_context(self) -> str:
        with self.lock:
            require(self.active_sandbox, "E2B context had no active Sandbox")
            self.context_count += 1
            return "messages-programmatic-context-{}".format(self.context_count)

    def finish_sandbox(self) -> None:
        with self.lock:
            require(self.active_sandbox, "E2B kill had no active Sandbox")
            self.active_sandbox = False

    def record_catalog(self, request: Dict[str, Any]) -> None:
        with self.lock:
            self.catalog_requests.append(request)

    def record_sandbox(
        self, method: str, path: str, request: Optional[Dict[str, Any]] = None
    ) -> None:
        with self.lock:
            self.sandbox_requests.append({"method": method, "path": path, "body": request})

    def record_error(self, label: str) -> None:
        with self.lock:
            self.errors.append(label)


def has_tool_result(payload: Mapping[str, Any]) -> bool:
    messages = payload.get("messages")
    return isinstance(messages, list) and any(
        isinstance(message, dict)
        and isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in message["content"]
        )
        for message in messages
    )


def detect_messages_scenario(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload.get("messages"), separators=(",", ":"), sort_keys=True)
    if "functional-startup-auth-probe" in encoded:
        return "startup-auth-probe"
    if "functional-messages-programmatic" in encoded:
        return "messages-programmatic"
    raise HarnessFailure("Messages model mock received an unknown functional scenario")


def messages_response(scenario: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if scenario == "startup-auth-probe":
        return {
            "id": "msg_startup_auth",
            "type": "message",
            "role": "assistant",
            "model": "mock-claude",
            "content": [{"type": "text", "text": "startup authenticated"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    if scenario == "messages-programmatic":
        if has_tool_result(payload):
            return {
                "id": "msg_messages_programmatic_final",
                "type": "message",
                "role": "assistant",
                "model": "mock-claude",
                "content": [{"type": "text", "text": "messages programmatic final answer"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 3},
            }
        return {
            "id": "msg_messages_programmatic_call",
            "type": "message",
            "role": "assistant",
            "model": "mock-claude",
            "content": [
                {
                    "type": "tool_use",
                    "id": "messages-program-1",
                    "name": "_agentgateway_programmatic_tool_calling",
                    "input": {
                        "code": (
                            "result = tools.call('catalog_lookup', {'sku': 'AG-1'})\n"
                            "program_output({'stock': result})"
                        )
                    },
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
    raise HarnessFailure("Messages model mock could not select a response")


class AnthropicModelHandler(QuietHandler):
    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            self.require_contract("/v1/messages", "x-api-key", "functional-model-token")
            request = self.read_json()
            require(request.get("model") == "mock-claude", "Messages model mock received wrong model")
            require(request.get("stream") is not True, "internal Messages rounds must not stream")
            scenario = detect_messages_scenario(request)
            state.record_model(scenario, request)
            self.send_json(200, messages_response(scenario, request))
        except Exception:
            state.record_error("Messages model mock rejected a request")
            self.send_safe_failure()


class CatalogHandler(QuietHandler):
    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            self.require_contract("/lookup", "authorization", "Bearer functional-catalog-token")
            request = self.read_json()
            require(request.get("tool_name") == "catalog_lookup", "catalog mock received wrong tool")
            arguments = request.get("arguments")
            require(
                isinstance(arguments, dict) and arguments.get("sku") == "AG-1",
                "catalog mock received unexpected arguments",
            )
            state.record_catalog(request)
            self.send_json(200, {"sku": "AG-1", "in_stock": 3})
        except Exception:
            state.record_error("catalog mock rejected a request")
            self.send_safe_failure()


class SandboxHandler(QuietHandler):
    def require_control_auth(self) -> None:
        require(self.headers.get("x-api-key") == "functional-sandbox-token", "E2B control auth failed")

    def require_data_auth(self) -> None:
        require(self.headers.get("x-access-token") == "functional-envd-token", "E2B envd auth failed")
        require(
            self.headers.get("e2b-traffic-access-token") == "functional-traffic-token",
            "E2B traffic auth failed",
        )

    def send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("content-length", "0")
        self.end_headers()

    def send_ndjson(self, events: Sequence[Mapping[str, Any]]) -> None:
        encoded = b"".join(
            json.dumps(event, separators=(",", ":")).encode("utf-8") + b"\n"
            for event in events
        )
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            if self.path == "/sandboxes":
                self.require_control_auth()
                request = self.read_json()
                require(
                    request == {"templateID": "code-interpreter-v1", "timeout": 3, "secure": True},
                    "E2B create request shape changed",
                )
                sandbox_id = state.begin_sandbox()
                state.record_sandbox("POST", self.path, request)
                self.send_json(
                    201,
                    {
                        "clientID": "functional-client",
                        "envdVersion": "0.1.0",
                        "sandboxID": sandbox_id,
                        "templateID": "code-interpreter-v1",
                        "domain": "sandbox.example.com",
                        "envdAccessToken": "functional-envd-token",
                        "trafficAccessToken": "functional-traffic-token",
                    },
                )
                return
            self.require_data_auth()
            if self.path == "/contexts":
                request = self.read_json()
                require(
                    request == {"language": "python", "cwd": "/home/user"},
                    "E2B context request shape changed",
                )
                context_id = state.next_context()
                state.record_sandbox("POST", self.path, request)
                self.send_json(200, {"id": context_id, "language": "python", "cwd": "/home/user"})
                return
            if self.path == "/execute":
                request = self.read_json()
                env_vars = request.get("env_vars")
                require(isinstance(request.get("code"), str), "E2B execute omitted wrapped code")
                require(isinstance(env_vars, dict), "Messages program env_vars were absent")
                nonce = env_vars.get("AGENTGATEWAY_PTC_NONCE")
                replay_encoded = env_vars.get("AGENTGATEWAY_PTC_REPLAY")
                require(isinstance(nonce, str), "Messages program nonce was absent")
                require(isinstance(replay_encoded, str), "Messages program replay was absent")
                replay = json.loads(base64.b64decode(replay_encoded))
                require(isinstance(replay, list), "Messages program replay was not an array")
                if len(replay) == 0:
                    outcome = {
                        "version": 1,
                        "kind": "pending",
                        "sequence": 0,
                        "name": "catalog_lookup",
                        "arguments": {"sku": "AG-1"},
                    }
                elif len(replay) == 1:
                    outcome = {
                        "version": 1,
                        "kind": "completed",
                        "output": {"stock": {"sku": "AG-1", "in_stock": 3}},
                    }
                else:
                    raise HarnessFailure("Messages program replay length changed")
                payload = json.dumps(outcome, separators=(",", ":")).encode("utf-8")
                frame = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
                stdout = "__AGENTGATEWAY_PTC_V1__{}:{}\n".format(nonce, frame)
                state.record_sandbox("POST", self.path, request)
                self.send_ndjson(
                    [
                        {"type": "stdout", "text": stdout, "timestamp": 1},
                        {"type": "number_of_executions", "execution_count": 1},
                    ]
                )
                return
            raise HarnessFailure("E2B mock received an unknown POST path")
        except Exception:
            state.record_error("E2B mock rejected a POST request")
            self.send_safe_failure()

    def do_DELETE(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            if self.path.startswith("/contexts/"):
                self.require_data_auth()
                state.record_sandbox("DELETE", self.path)
                self.send_empty(204)
                return
            require(
                self.path == "/sandboxes/functional-messages-programmatic",
                "E2B kill targeted wrong Sandbox",
            )
            self.require_control_auth()
            state.record_sandbox("DELETE", self.path)
            state.finish_sandbox()
            self.send_empty(204)
        except Exception:
            state.record_error("E2B mock rejected a DELETE request")
            self.send_safe_failure()


def functional_config(
    gateway_port: int,
    readiness_port: int,
    model_port: int,
    sandbox_port: int,
    catalog_port: int,
) -> str:
    return f"""\
config:
  adminAddr: 127.0.0.1:0
  statsAddr: 127.0.0.1:0
  readinessAddr: 127.0.0.1:{readiness_port}
llm:
  port: {gateway_port}
  policies:
    apiKey:
      keys:
      - key: {FUNCTIONAL_CLIENT_TOKEN}
      mode: strict
      location:
        header:
          name: authorization
          prefix: 'Bearer '
  toolRuntime:
    maxRounds: 4
    maxToolCalls: 8
    maxParallelToolCalls: 4
    totalTimeout: 8s
    maxArgumentsBytes: 4096
    maxOutputBytes: 16384
    tools:
    - name: catalog_lookup
      backend:
        type: http
        url: http://127.0.0.1:{catalog_port}/lookup
        timeout: 3s
        bearerToken: functional-catalog-token
    - name: code_interpreter
      builtin: codeInterpreter
      backend:
        type: e2b
        apiKey: functional-sandbox-token
        apiUrl: http://127.0.0.1:{sandbox_port}
        domain: sandbox.example.com
        timeout: 3s
  models:
  - name: claude
    provider: anthropic
    params:
      model: mock-claude
      apiKey: functional-model-token
      baseUrl: http://127.0.0.1:{model_port}/v1
"""


def programmatic_messages_tools() -> List[Dict[str, Any]]:
    return [
        {"type": "code_execution_20260120", "name": "code_execution"},
        {
            "name": "catalog_lookup",
            "description": "Look up stock for a catalog SKU.",
            "allowed_callers": ["code_execution_20260120"],
            "input_schema": {
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    ]


def verify_ingress_auth(base_url: str, state: MockState) -> None:
    payload = {
        "model": "claude",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "functional-startup-auth-probe"}],
    }
    with state.lock:
        before = len(state.model_requests["startup-auth-probe"])
    for token in (None, "functional-invalid-token"):
        status = post_status(base_url + "/v1/messages", payload, token)
        if status != 401:
            raise GatewayStartupFailure("invalid client identity was not rejected with HTTP 401")
    with state.lock:
        if len(state.model_requests["startup-auth-probe"]) != before:
            raise GatewayStartupFailure("rejected client API key reached the Messages model")
    status, body = post_json(base_url + "/v1/messages", payload)
    if status != 200 or body.get("id") != "msg_startup_auth":
        raise GatewayStartupFailure("authenticated Messages identity probe failed")


class FunctionalCases:
    def __init__(self, base_url: str, state: MockState) -> None:
        self.base_url = base_url
        self.state = state

    def messages_programmatic(self) -> None:
        with self.state.lock:
            model_before = len(self.state.model_requests["messages-programmatic"])
            catalog_before = len(self.state.catalog_requests)
            sandbox_before = len(self.state.sandbox_requests)
        status, body = post_json(
            self.base_url + "/v1/messages",
            {
                "model": "claude",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "functional-messages-programmatic: check stock"}
                ],
                "tools": programmatic_messages_tools(),
            },
        )
        require(status == 200, "Messages programmatic case did not return HTTP 200")
        require(body.get("type") == "message", "Messages response had the wrong type")
        content = body.get("content")
        require(isinstance(content, list), "Messages response omitted final content")
        require(
            all(
                isinstance(block, dict)
                and block.get("type") not in {"tool_use", "server_tool_use"}
                for block in content
            ),
            "Messages response leaked an intermediate tool block",
        )
        usage = body.get("usage")
        require(
            isinstance(usage, dict) and usage.get("input_tokens") == 12,
            "Messages input usage was not aggregated across rounds",
        )
        require(
            "_agentgateway_" not in json.dumps(body, separators=(",", ":")),
            "Messages response leaked a reserved name",
        )
        with self.state.lock:
            model_count = len(self.state.model_requests["messages-programmatic"]) - model_before
            catalog_count = len(self.state.catalog_requests) - catalog_before
            sandbox_events = self.state.sandbox_requests[sandbox_before:]
        require(model_count > 1, "Messages case did not make multiple model rounds")
        require(catalog_count == 1, "Messages case did not make one managed tool call")
        require(len(sandbox_events) == 10, "Messages case did not complete two E2B lifecycles")


def run_gateway_attempt(
    binary: Path,
    selected_cases: Sequence[str],
    state: MockState,
    servers: Sequence[MockServer],
) -> None:
    ports, reservations = reserve_distinct_loopback_ports(2)
    gateway_port, readiness_port = ports
    try:
        with tempfile.TemporaryDirectory(prefix="agentgateway-anthropic-functional-") as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                functional_config(
                    gateway_port,
                    readiness_port,
                    servers[0].port,
                    servers[1].port,
                    servers[2].port,
                ),
                encoding="utf-8",
            )
            log_path = Path(temp_dir) / "agentgateway.log"
            with log_path.open("wb") as child_log:
                process: Optional[subprocess.Popen[Any]] = None
                try:
                    close_reservations(reservations)
                    reservations = []
                    process = subprocess.Popen(
                        [str(binary), "--file", str(config_path)],
                        cwd=temp_dir,
                        env=safe_child_environment(),
                        stdin=subprocess.DEVNULL,
                        stdout=child_log,
                        stderr=subprocess.STDOUT,
                    )
                    wait_for_readiness(process, readiness_port)
                    base_url = "http://127.0.0.1:{}".format(gateway_port)
                    verify_ingress_auth(base_url, state)
                    if process.poll() is not None:
                        raise GatewayStartupFailure("gateway exited during its identity probe")
                    cases = FunctionalCases(base_url, state)
                    methods = {"messages-programmatic": cases.messages_programmatic}
                    for case_name in selected_cases:
                        methods[case_name]()
                        print("PASS {}".format(case_name))
                    with state.lock:
                        require(not state.errors, "a local mock rejected a functional request")
                    require(process.poll() is None, "gateway exited during functional cases")
                finally:
                    if process is not None:
                        stop_process(process)
    finally:
        close_reservations(reservations)


def run(binary: Path, selected_cases: Sequence[str]) -> None:
    binary = binary.expanduser().resolve()
    require(binary.is_file(), "release binary path is not a file")
    require(os.access(str(binary), os.X_OK), "release binary path is not executable")
    state = MockState()
    servers: List[MockServer] = []
    try:
        for handler in (AnthropicModelHandler, SandboxHandler, CatalogHandler):
            server = MockServer(handler, state)
            servers.append(server)
            server.start()
        for attempt in range(MAX_GATEWAY_START_ATTEMPTS):
            try:
                run_gateway_attempt(binary, selected_cases, state, servers)
                return
            except GatewayStartupFailure as error:
                if attempt + 1 == MAX_GATEWAY_START_ATTEMPTS:
                    raise HarnessFailure(
                        "gateway startup and identity failed after bounded retries: {}".format(error)
                    ) from error
    finally:
        for server in reversed(servers):
            server.close()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default="./target/release/agentgateway",
        help="path to the already-built release agentgateway executable",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=CASE_NAMES,
        dest="cases",
        help="run one case; repeat to select several (default: all)",
    )
    parser.add_argument("--list-cases", action="store_true", help="list case names and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.list_cases:
        for case_name in CASE_NAMES:
            print(case_name)
        return 0
    selected_cases = args.cases if args.cases else list(CASE_NAMES)
    try:
        run(Path(args.binary), selected_cases)
    except HarnessFailure as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        return 1
    except (OSError, subprocess.SubprocessError) as error:
        print(
            "FAIL: harness infrastructure error ({}); verify executable, loopback, "
            "and temporary-directory permissions".format(type(error).__name__),
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("FAIL: interrupted; child cleanup completed", file=sys.stderr)
        return 130
    print("PASS {} Anthropic functional case(s)".format(len(selected_cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
