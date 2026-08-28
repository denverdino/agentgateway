#!/usr/bin/env python3
"""Hermetic functional tests for the AgentGateway Responses tool runtime.

The harness launches the requested release binary and exercises only its public
HTTP API. All provider and HTTP Tool backend dependencies are loopback mocks.
Failure diagnostics intentionally report structure and status only; request or
response content, code, tool output, URLs, and credentials are never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CASE_NAMES = (
    "dual-tool-overlap",
    "programmatic-server-tools",
    "web-search-single",
    "streaming-tool-runtime",
    "multi-code-reuse",
    "active-mode-rejections",
    "unmanaged-function-passthrough",
    "tool-search-deferred",
)
MAX_HTTP_BODY_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 5.0
STARTUP_TIMEOUT_SECONDS = 15.0
FUNCTIONAL_CLIENT_TOKEN = "functional-client-token"
BACKEND_BARRIER_TIMEOUT_SECONDS = 1.0
MAX_GATEWAY_START_ATTEMPTS = 3


class HarnessFailure(Exception):
    """A content-free functional assertion failure."""


class GatewayStartupFailure(HarnessFailure):
    """A retryable, content-free gateway startup or identity failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessFailure(message)


class TwoBackendBarrier:
    """Require both managed backends to arrive before either may complete."""

    def __init__(self, timeout_seconds: float) -> None:
        self.condition = threading.Condition()
        self.timeout_seconds = timeout_seconds
        self.arrived = set()  # type: set[str]
        self.confirmed = set()  # type: set[str]

    def arrive_and_wait(self, label: str) -> bool:
        deadline = time.monotonic() + self.timeout_seconds
        with self.condition:
            self.arrived.add(label)
            self.condition.notify_all()
            while len(self.arrived) < 2:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
            self.confirmed.add(label)
            return True

    def confirmed_labels(self) -> set[str]:
        with self.condition:
            return set(self.confirmed)


def verify_barrier_contract() -> None:
    serial = TwoBackendBarrier(0.04)
    started = time.monotonic()
    require(not serial.arrive_and_wait("web"), "serial backend passed the overlap barrier")
    require(time.monotonic() - started < 0.25, "serial overlap proof did not fail quickly")

    concurrent = TwoBackendBarrier(1.0)
    peer_result: List[bool] = []
    peer = threading.Thread(
        target=lambda: peer_result.append(concurrent.arrive_and_wait("web")), daemon=True
    )
    peer.start()
    sandbox_result = concurrent.arrive_and_wait("sandbox")
    peer.join(timeout=1.0)
    require(not peer.is_alive(), "concurrent overlap barrier did not release its peer")
    require(
        sandbox_result and peer_result == [True],
        "concurrent backends did not both pass the overlap barrier",
    )


def response_body(
    response_id: str,
    output: Sequence[Mapping[str, Any]],
    input_tokens: int,
    output_tokens: int,
) -> Dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "mock-model",
        "output": list(output),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def function_call(name: str, call_id: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_" + call_id,
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, separators=(",", ":")),
        "status": "completed",
    }


def message(item_id: str, text: str) -> Dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class MockState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_requests: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.web_requests: List[Dict[str, Any]] = []
        self.catalog_requests: List[Dict[str, Any]] = []
        self.sandbox_requests: List[Dict[str, Any]] = []
        self.pending_sandbox_scenarios: List[str] = []
        self.active_sandbox_scenario: Optional[str] = None
        self.context_counts: DefaultDict[str, int] = defaultdict(int)
        self.dual_tool_barrier = TwoBackendBarrier(BACKEND_BARRIER_TIMEOUT_SECONDS)
        self.errors: List[str] = []

    def record_model(self, scenario: str, request: Dict[str, Any]) -> None:
        with self.lock:
            self.model_requests[scenario].append(request)
            if scenario in {
                "dual-tool-overlap",
                "programmatic-server-tools",
                "multi-code-reuse",
            } and not has_function_output(request):
                count = 4 if scenario == "programmatic-server-tools" else 1
                self.pending_sandbox_scenarios.extend([scenario] * count)

    def record_web(self, scenario: str, request: Dict[str, Any]) -> None:
        with self.lock:
            self.web_requests.append(request)

    def record_catalog(self, request: Dict[str, Any]) -> None:
        with self.lock:
            self.catalog_requests.append(request)

    def begin_sandbox(self) -> Tuple[str, str]:
        with self.lock:
            require(bool(self.pending_sandbox_scenarios), "E2B create had no pending code scenario")
            scenario = self.pending_sandbox_scenarios.pop(0)
            require(self.active_sandbox_scenario is None, "E2B Sandbox lifecycles overlapped")
            self.active_sandbox_scenario = scenario
            sandbox_id = {
                "dual-tool-overlap": "functional-dual",
                "programmatic-server-tools": "functional-programmatic",
                "multi-code-reuse": "functional-multi",
            }[scenario]
            return scenario, sandbox_id

    def active_sandbox(self) -> str:
        with self.lock:
            require(
                self.active_sandbox_scenario is not None,
                "E2B data request had no active Sandbox",
            )
            return self.active_sandbox_scenario

    def next_context(self, scenario: str) -> str:
        with self.lock:
            self.context_counts[scenario] += 1
            prefix = {
                "dual-tool-overlap": "dual",
                "programmatic-server-tools": "programmatic",
                "multi-code-reuse": "multi",
            }[scenario]
            return "{}-context-{}".format(prefix, self.context_counts[scenario])

    def finish_sandbox(self, scenario: str) -> None:
        with self.lock:
            require(
                self.active_sandbox_scenario == scenario,
                "E2B kill targeted wrong active Sandbox",
            )
            self.active_sandbox_scenario = None

    def record_sandbox(
        self, scenario: str, method: str, path: str, request: Optional[Dict[str, Any]] = None
    ) -> None:
        with self.lock:
            self.sandbox_requests.append(
                {"scenario": scenario, "method": method, "path": path, "body": request}
            )

    def record_error(self, label: str) -> None:
        with self.lock:
            self.errors.append(label)


def input_contains(payload: Mapping[str, Any], marker: str) -> bool:
    encoded = json.dumps(payload.get("input"), separators=(",", ":"), sort_keys=True)
    return marker in encoded


def has_function_output(payload: Mapping[str, Any]) -> bool:
    input_items = payload.get("input")
    return isinstance(input_items, list) and any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in input_items
    )


def detect_model_scenario(payload: Mapping[str, Any]) -> str:
    tools = payload.get("tools")
    if isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("name") == "client_owned"
        for tool in tools
    ):
        return "unmanaged-function-passthrough"
    for scenario, marker in (
        ("startup-auth-probe", "functional-startup-auth-probe"),
        ("dual-tool-overlap", "functional-dual"),
        ("programmatic-server-tools", "functional-programmatic"),
        ("web-search-single", "functional-web-only"),
        ("multi-code-reuse", "functional-multi-code"),
        ("tool-search-deferred", "functional-tool-search"),
    ):
        if input_contains(payload, marker):
            return scenario
    raise HarnessFailure("model mock received an unknown functional scenario")


def model_response(scenario: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    continuation = has_function_output(payload)
    if scenario == "startup-auth-probe":
        return response_body(
            "resp_startup_auth", [message("msg_startup_auth", "startup authenticated")], 1, 1
        )
    if scenario == "dual-tool-overlap":
        if continuation:
            return response_body(
                "resp_dual_final", [message("msg_dual_final", "dual final answer")], 12, 6
            )
        output = [
            message("msg_dual_intermediate", "dual-intermediate-not-client-visible"),
            function_call(
                "_agentgateway_web_search", "dual-web-1", {"query": "functional dual search"}
            ),
            function_call(
                "_agentgateway_code_interpreter", "dual-code-2", {"code": "print(42)"}
            ),
        ]
        return response_body("resp_dual_intermediate", output, 10, 4)
    if scenario == "programmatic-server-tools":
        if continuation:
            return response_body(
                "resp_programmatic_final",
                [message("msg_programmatic_final", "programmatic final answer")],
                4,
                2,
            )
        return response_body(
            "resp_programmatic_calls",
            [
                function_call(
                    "_agentgateway_programmatic_tool_calling",
                    "programmatic-program-1",
                    {
                        "code": (
                            "fact = tools.call('web_search', "
                            "{'query': 'functional programmatic search'})\n"
                            "calculation = tools.call('code_interpreter', "
                            "{'code': 'print(29 * 31)'})\n"
                            "program_output({'fact': fact, 'calculation': calculation, "
                            "'product': 899})"
                        )
                    },
                ),
            ],
            10,
            5,
        )
    if scenario == "web-search-single":
        if continuation:
            return response_body(
                "resp_web_final", [message("msg_web_final", "web final answer")], 5, 3
            )
        return response_body(
            "resp_web_intermediate",
            [
                function_call(
                    "_agentgateway_web_search", "web-only-1", {"query": "functional web only"}
                )
            ],
            3,
            2,
        )
    if scenario == "multi-code-reuse":
        if continuation:
            return response_body(
                "resp_code_final", [message("msg_code_final", "code final answer")], 7, 4
            )
        return response_body(
            "resp_code_intermediate",
            [
                function_call(
                    "_agentgateway_code_interpreter",
                    "code-batch-1",
                    {"code": "private_value = 41; print('set')"},
                ),
                function_call(
                    "_agentgateway_code_interpreter",
                    "code-batch-2",
                    {
                        "code": "print('isolated' if 'private_value' not in globals() else 'leaked')"
                    },
                ),
            ],
            6,
            3,
        )
    if scenario == "tool-search-deferred":
        # The catalog call is the marker for the last round: its output only exists after the
        # searched declaration was injected and actually called.
        if input_contains(payload, "catalog-call-1"):
            return response_body(
                "resp_search_final",
                [message("msg_search_final", "deferred tool answer")],
                9,
                4,
            )
        if continuation:
            return response_body(
                "resp_search_call",
                [function_call("catalog_lookup", "catalog-call-1", {"sku": "AG-1"})],
                7,
                3,
            )
        return response_body(
            "resp_search_probe",
            [
                function_call(
                    "_agentgateway_tool_search",
                    "search-call-1",
                    {"query": "catalog lookup"},
                )
            ],
            5,
            2,
        )
    if scenario == "unmanaged-function-passthrough":
        return response_body(
            "resp_unmanaged",
            [function_call("client_owned", "client-call-1", {"value": 7})],
            4,
            2,
        )
    raise HarnessFailure("model mock could not select a response")


class QuietHandler(BaseHTTPRequestHandler):
    server_version = "AgentGatewayFunctionalMock/1"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError as error:
            raise HarnessFailure("mock received an invalid content length") from error
        require(0 < length <= MAX_HTTP_BODY_BYTES, "mock request body exceeded its safe bound")
        body = self.rfile.read(length)
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HarnessFailure("mock received invalid JSON") from error
        require(isinstance(value, dict), "mock request JSON must be an object")
        return value

    def require_contract(self, path: str, bearer_token: str) -> None:
        require(self.path == path, "mock request used an unexpected fixed path")
        require(
            self.headers.get("authorization") == "Bearer " + bearer_token,
            "mock request used unexpected bearer authentication",
        )
        require(
            self.headers.get("content-type") == "application/json",
            "mock request used an unexpected content type",
        )

    def send_json(
        self,
        status: int,
        body: Mapping[str, Any],
        extra_headers: Iterable[Tuple[str, str]] = (),
        delay: float = 0.0,
    ) -> None:
        if delay:
            time.sleep(delay)
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def send_safe_failure(self) -> None:
        encoded = b'{"error":"mock_failure"}'
        self.send_response(500)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ModelHandler(QuietHandler):
    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            self.require_contract("/v1/responses", "functional-model-token")
            request = self.read_json()
            require(request.get("model") == "mock-model", "model mock received wrong selected model")
            scenario = detect_model_scenario(request)
            state.record_model(scenario, request)
            self.send_json(200, model_response(scenario, request))
        except Exception:
            state.record_error("model mock rejected a request")
            self.send_safe_failure()


class WebSearchHandler(QuietHandler):
    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            self.require_contract("/invoke", "functional-web-token")
            request = self.read_json()
            query = request.get("query")
            if query == "functional dual search":
                scenario = "dual-tool-overlap"
                delay = 0.7
            elif query == "functional programmatic search":
                scenario = "programmatic-server-tools"
                delay = 0.01
            elif query == "functional web only":
                scenario = "web-search-single"
                delay = 0.01
            else:
                raise HarnessFailure("Web Search mock received an unknown query")
            state.record_web(scenario, request)
            if scenario == "dual-tool-overlap":
                require(
                    state.dual_tool_barrier.arrive_and_wait("web"),
                    "Web Search did not observe the Sandbox request before completion",
                )
            self.send_json(
                200,
                {
                    "results": [
                        {
                            "title": "Functional result",
                            "url": "https://agentgateway.dev/functional",
                            "snippet": "bounded functional snippet",
                            "published_at": None,
                        }
                    ]
                },
                delay=delay,
            )
        except Exception:
            state.record_error("Web Search mock rejected a request")
            self.send_safe_failure()


class CatalogHandler(QuietHandler):
    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        try:
            self.require_contract("/lookup", "functional-catalog-token")
            request = self.read_json()
            require(
                request.get("tool_name") == "catalog_lookup",
                "catalog mock received the wrong tool name",
            )
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
                    request
                    == {
                        "templateID": "code-interpreter-v1",
                        "timeout": 3,
                        "secure": True,
                    },
                    "E2B create request shape changed",
                )
                scenario, sandbox_id = state.begin_sandbox()
                state.record_sandbox(scenario, "POST", self.path, request)
                if scenario == "dual-tool-overlap":
                    require(
                        state.dual_tool_barrier.arrive_and_wait("sandbox"),
                        "E2B create did not observe Web Search before completion",
                    )
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
            scenario = state.active_sandbox()
            if self.path == "/contexts":
                request = self.read_json()
                require(
                    request == {"language": "python", "cwd": "/home/user"},
                    "E2B context request shape changed",
                )
                context_id = state.next_context(scenario)
                state.record_sandbox(scenario, "POST", self.path, request)
                self.send_json(
                    200,
                    {"id": context_id, "language": "python", "cwd": "/home/user"},
                )
                return
            if self.path == "/execute":
                request = self.read_json()
                context_id = request.get("context_id")
                require(isinstance(request.get("code"), str), "E2B execute omitted wrapped code")
                require(request.get("language") is None, "E2B execute selected an untrusted language")
                if scenario == "dual-tool-overlap":
                    require(request.get("env_vars") is None, "direct E2B received env vars")
                    require(context_id == "dual-context-1", "dual E2B context ID changed")
                    stdout = "42\n"
                elif scenario == "programmatic-server-tools":
                    env_vars = request.get("env_vars")
                    if env_vars is None:
                        require(
                            context_id == "programmatic-context-3",
                            "nested Code Interpreter context ID changed",
                        )
                        stdout = "899\n"
                    else:
                        require(isinstance(env_vars, dict), "program env_vars were not an object")
                        nonce = env_vars.get("AGENTGATEWAY_PTC_NONCE")
                        replay_encoded = env_vars.get("AGENTGATEWAY_PTC_REPLAY")
                        require(isinstance(nonce, str), "program nonce was absent")
                        require(isinstance(replay_encoded, str), "program replay was absent")
                        replay = json.loads(base64.b64decode(replay_encoded))
                        require(isinstance(replay, list), "program replay was not an array")
                        if len(replay) == 0:
                            outcome = {
                                "version": 1,
                                "kind": "pending",
                                "sequence": 0,
                                "name": "web_search",
                                "arguments": {"query": "functional programmatic search"},
                            }
                        elif len(replay) == 1:
                            outcome = {
                                "version": 1,
                                "kind": "pending",
                                "sequence": 1,
                                "name": "code_interpreter",
                                "arguments": {"code": "print(29 * 31)"},
                            }
                        elif len(replay) == 2:
                            outcome = {
                                "version": 1,
                                "kind": "completed",
                                "output": {
                                    "fact": "bounded functional snippet",
                                    "product": 899,
                                },
                            }
                        else:
                            raise HarnessFailure("program replay length changed")
                        payload = json.dumps(outcome, separators=(",", ":")).encode("utf-8")
                        frame = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
                        stdout = "__AGENTGATEWAY_PTC_V1__{}:{}\n".format(nonce, frame)
                elif context_id == "multi-context-1":
                    require(request.get("env_vars") is None, "direct E2B received env vars")
                    stdout = "set\n"
                elif context_id == "multi-context-2":
                    require(request.get("env_vars") is None, "direct E2B received env vars")
                    stdout = "isolated\n"
                else:
                    raise HarnessFailure("E2B execute received unknown context")
                state.record_sandbox(scenario, "POST", self.path, request)
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
            scenario = state.active_sandbox()
            if self.path.startswith("/contexts/"):
                self.require_data_auth()
                state.record_sandbox(scenario, "DELETE", self.path)
                self.send_empty(204)
                return
            expected_id = {
                "dual-tool-overlap": "functional-dual",
                "programmatic-server-tools": "functional-programmatic",
                "multi-code-reuse": "functional-multi",
            }[scenario]
            require(self.path == "/sandboxes/" + expected_id, "E2B kill targeted wrong Sandbox")
            self.require_control_auth()
            state.record_sandbox(scenario, "DELETE", self.path)
            state.finish_sandbox(scenario)
            self.send_empty(204)
        except Exception:
            state.record_error("E2B mock rejected a DELETE request")
            self.send_safe_failure()


class MockServer:
    def __init__(self, handler: Any, state: MockState) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.server.state = state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def reserve_distinct_loopback_ports(count: int) -> Tuple[List[int], List[socket.socket]]:
    reservations: List[socket.socket] = []
    try:
        for _ in range(count):
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            reservation.bind(("127.0.0.1", 0))
            reservation.listen(1)
            reservations.append(reservation)
        ports = [int(reservation.getsockname()[1]) for reservation in reservations]
        require(len(set(ports)) == count, "loopback port reservations were not distinct")
        return ports, reservations
    except Exception:
        for reservation in reservations:
            reservation.close()
        raise


def close_reservations(reservations: Iterable[socket.socket]) -> None:
    for reservation in reservations:
        reservation.close()


def functional_config(
    gateway_port: int,
    readiness_port: int,
    model_port: int,
    web_port: int,
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
    - name: web_search
      builtin: webSearch
      backend:
        type: http
        url: http://127.0.0.1:{web_port}/invoke
        timeout: 3s
        bearerToken: functional-web-token
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
  - name: smart
    provider: openAI
    params:
      model: mock-model
      apiKey: functional-model-token
      baseUrl: http://127.0.0.1:{model_port}/v1
"""


def safe_child_environment() -> Dict[str, str]:
    environment: Dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "NO_COLOR": "1",
        "RUST_BACKTRACE": "0",
        "RUST_LOG": "error",
    }
    if os.environ.get("TMPDIR"):
        environment["TMPDIR"] = os.environ["TMPDIR"]
    return environment


def wait_for_readiness(process: subprocess.Popen[Any], readiness_port: int) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    readiness_url = "http://127.0.0.1:{}/healthz/ready".format(readiness_port)
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise GatewayStartupFailure(
                "gateway exited before readiness with status {} (logs withheld)".format(exit_code)
            )
        try:
            with urllib.request.urlopen(readiness_url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.05)
    raise GatewayStartupFailure("gateway readiness timed out")


def post_json(
    url: str,
    payload: Mapping[str, Any],
    client_token: Optional[str] = FUNCTIONAL_CLIENT_TOKEN,
) -> Tuple[int, Dict[str, Any]]:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"content-type": "application/json"}
    if client_token is not None:
        headers["authorization"] = "Bearer " + client_token
    request = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read(MAX_HTTP_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read(MAX_HTTP_BODY_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("public Responses request failed at the HTTP transport") from error
    require(len(body) <= MAX_HTTP_BODY_BYTES, "public response exceeded the harness body bound")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessFailure("public response was not valid JSON") from error
    require(isinstance(value, dict), "public response JSON was not an object")
    return status, value


def post_sse(url: str, payload: Mapping[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={
            "authorization": "Bearer " + FUNCTIONAL_CLIENT_TOKEN,
            "content-type": "application/json",
            "accept": "text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            content_type = response.headers.get_content_type()
            body = response.read(MAX_HTTP_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        content_type = error.headers.get_content_type()
        body = error.read(MAX_HTTP_BODY_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("public Responses streaming request failed") from error
    require(len(body) <= MAX_HTTP_BODY_BYTES, "public SSE response exceeded the body bound")
    require(content_type == "text/event-stream", "public streaming response was not SSE")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HarnessFailure("public SSE response was not UTF-8") from error
    events: List[Dict[str, Any]] = []
    for frame in text.split("\n\n"):
        event_name = next(
            (line[7:] for line in frame.splitlines() if line.startswith("event: ")), None
        )
        data = next(
            (line[6:] for line in frame.splitlines() if line.startswith("data: ")), None
        )
        if data is None:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise HarnessFailure("public SSE event was not JSON") from error
        require(isinstance(event, dict), "public SSE event data was not an object")
        require(event.get("type") == event_name, "public SSE event name and type differed")
        events.append(event)
    return status, events


def post_status(
    url: str,
    payload: Mapping[str, Any],
    client_token: Optional[str],
) -> int:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"content-type": "application/json"}
    if client_token is not None:
        headers["authorization"] = "Bearer " + client_token
    request = urllib.request.Request(url, data=encoded, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read(MAX_HTTP_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read(MAX_HTTP_BODY_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("public auth probe failed at the HTTP transport") from error
    require(len(body) <= MAX_HTTP_BODY_BYTES, "public auth response exceeded the body bound")
    return status


def verify_ingress_auth(base_url: str, state: MockState) -> None:
    payload = {
        "model": "smart",
        "input": "functional-startup-auth-probe",
        "stream": False,
    }
    with state.lock:
        model_before = len(state.model_requests["startup-auth-probe"])
    try:
        status = post_status(base_url + "/v1/responses", payload, client_token=None)
        if status != 401:
            raise GatewayStartupFailure("missing client API key was not rejected with HTTP 401")
        status = post_status(
            base_url + "/v1/responses", payload, client_token="functional-invalid-token"
        )
        if status != 401:
            raise GatewayStartupFailure("invalid client API key was not rejected with HTTP 401")
    except HarnessFailure as error:
        if isinstance(error, GatewayStartupFailure):
            raise
        raise GatewayStartupFailure("gateway identity probe failed at the HTTP boundary") from error
    with state.lock:
        model_after_rejections = len(state.model_requests["startup-auth-probe"])
    if model_after_rejections != model_before:
        raise GatewayStartupFailure("rejected client API key reached the configured model")

    try:
        status, body = post_json(base_url + "/v1/responses", payload)
    except HarnessFailure as error:
        raise GatewayStartupFailure("authenticated gateway identity probe failed") from error
    if status != 200:
        raise GatewayStartupFailure("configured client API key was not accepted")
    if body.get("id") != "resp_startup_auth":
        raise GatewayStartupFailure("startup identity probe returned wrong response")
    with state.lock:
        model_after_valid = len(state.model_requests["startup-auth-probe"])
    if model_after_valid != model_before + 1:
        raise GatewayStartupFailure("startup identity probe missed the configured model")


def managed_tools() -> List[Dict[str, Any]]:
    return [
        {"type": "web_search"},
        {"type": "code_interpreter", "container": {"type": "auto"}},
    ]


def parse_tool_output(item: Mapping[str, Any]) -> Dict[str, Any]:
    output = item.get("output")
    require(isinstance(output, str), "canonical function output was not a string")
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise HarnessFailure("canonical function output was not JSON") from error
    require(isinstance(value, dict), "canonical function output JSON was not an object")
    return value


class FunctionalCases:
    def __init__(self, base_url: str, state: MockState) -> None:
        self.base_url = base_url
        self.state = state

    def request(self, payload: Mapping[str, Any]) -> Tuple[int, Dict[str, Any]]:
        return post_json(self.base_url + "/v1/responses", payload)

    def dual_tool_overlap(self) -> None:
        with self.state.lock:
            web_before = len(self.state.web_requests)
            sandbox_before = len(self.state.sandbox_requests)
        status, body = self.request(
            {
                "model": "smart",
                "input": "functional-dual: search and calculate",
                "tools": managed_tools(),
                "parallel_tool_calls": True,
                "stream": False,
            }
        )
        require(
            status == 200,
            "dual-tool case did not return HTTP 200: status={}, body={}".format(status, body),
        )
        require(body.get("id") == "resp_dual_final", "dual-tool case returned the wrong round")
        require(body.get("model") == "mock-model", "dual-tool case returned the wrong model")
        output = body.get("output")
        require(isinstance(output, list) and len(output) == 1, "dual-tool final output leaked items")
        require(output[0].get("id") == "msg_dual_final", "dual-tool final message was incorrect")
        require(
            body.get("usage")
            == {"input_tokens": 22, "output_tokens": 10, "total_tokens": 32},
            "dual-tool aggregate usage was incorrect",
        )
        encoded_final = json.dumps(body, separators=(",", ":"))
        for marker in (
            "dual-intermediate-not-client-visible",
            "dual-web-1",
            "dual-code-2",
            "bounded functional snippet",
        ):
            require(marker not in encoded_final, "dual-tool response leaked intermediate content")
        with self.state.lock:
            requests = list(self.state.model_requests["dual-tool-overlap"])
            web_count = len(self.state.web_requests) - web_before
            sandbox_events = self.state.sandbox_requests[sandbox_before:]
        require(len(requests) == 2, "dual-tool case did not make exactly two model rounds")
        first_tools = requests[0].get("tools")
        require(isinstance(first_tools, list) and len(first_tools) == 2, "dual-tool mapping count changed")
        require(
            [tool.get("name") for tool in first_tools]
            == ["_agentgateway_web_search", "_agentgateway_code_interpreter"],
            "dual-tool mapping order or names changed",
        )
        canonical = requests[1].get("input")
        require(isinstance(canonical, list) and len(canonical) == 6, "dual canonical history was incomplete")
        require(
            [item.get("type") for item in canonical[1:]]
            == [
                "message",
                "function_call",
                "function_call",
                "function_call_output",
                "function_call_output",
            ],
            "dual canonical history types or order changed",
        )
        require(
            [canonical[3].get("call_id"), canonical[4].get("call_id"), canonical[5].get("call_id")]
            == ["dual-code-2", "dual-web-1", "dual-code-2"],
            "dual canonical call IDs were not stable",
        )
        web_output = parse_tool_output(canonical[4])
        code_output = parse_tool_output(canonical[5])
        require(web_output.get("ok") is True, "dual Web Search output was not normalized")
        require(code_output.get("stdout") == "42\n", "dual code output value was unstable")
        require(web_count == 1, "dual-tool case made the wrong number of Web Search calls")
        require(len(sandbox_events) == 5, "dual-tool case did not complete one E2B lifecycle")
        require(
            [(event.get("method"), event.get("path")) for event in sandbox_events]
            == [
                ("POST", "/sandboxes"),
                ("POST", "/contexts"),
                ("POST", "/execute"),
                ("DELETE", "/contexts/dual-context-1"),
                ("DELETE", "/sandboxes/functional-dual"),
            ],
            "dual-tool E2B lifecycle order changed",
        )
        require(
            self.state.dual_tool_barrier.confirmed_labels() == {"web", "sandbox"},
            "each delayed backend did not observe its peer before completion",
        )

    def programmatic_server_tools(self) -> None:
        with self.state.lock:
            web_before = len(self.state.web_requests)
            sandbox_before = len(self.state.sandbox_requests)
        status, body = self.request(
            {
                "model": "smart",
                "input": "functional-programmatic: search and calculate in one program",
                "tools": [
                    {"type": "web_search", "allowed_callers": ["programmatic"]},
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto"},
                        "allowed_callers": ["programmatic"],
                    },
                    {"type": "programmatic_tool_calling"},
                ],
                "parallel_tool_calls": True,
                "stream": False,
            }
        )
        require(status == 200, "programmatic case did not return HTTP 200")
        require(
            body.get("id") == "resp_programmatic_final",
            "programmatic case returned the wrong round",
        )
        require(
            body.get("usage")
            == {"input_tokens": 14, "output_tokens": 7, "total_tokens": 21},
            "programmatic aggregate usage was incorrect",
        )
        output = body.get("output")
        require(
            isinstance(output, list)
            and len(output) == 1
            and output[0].get("id") == "msg_programmatic_final",
            "programmatic final response leaked intermediate output",
        )

        with self.state.lock:
            requests = list(self.state.model_requests["programmatic-server-tools"])
            web_count = len(self.state.web_requests) - web_before
            sandbox_events = self.state.sandbox_requests[sandbox_before:]
        require(len(requests) == 2, "programmatic case did not make exactly two model rounds")
        require(web_count == 1, "programmatic case did not invoke Web Search exactly once")
        require(
            len(sandbox_events) == 20,
            "programmatic case did not complete four E2B lifecycles",
        )

        mapped_tools = requests[0].get("tools")
        require(
            isinstance(mapped_tools, list) and len(mapped_tools) == 1,
            "programmatic tool mapping count changed",
        )
        require(
            [tool.get("type") for tool in mapped_tools] == ["function"],
            "programmatic tool mapping order or types changed",
        )
        require(
            mapped_tools[0].get("name") == "_agentgateway_programmatic_tool_calling",
            "gateway programmatic function name changed",
        )
        parameters = mapped_tools[0].get("parameters")
        require(
            parameters
            == {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            "gateway programmatic function schema changed",
        )
        description = mapped_tools[0].get("description")
        require(
            isinstance(description, str)
            and '"name":"web_search"' in description
            and '"name":"code_interpreter"' in description,
            "gateway programmatic catalog omitted a server tool",
        )

        second_input = requests[1].get("input")
        require(
            isinstance(second_input, list) and len(second_input) == 3,
            "programmatic function continuation history was incomplete",
        )
        require(
            [item.get("type") for item in second_input]
            == ["message", "function_call", "function_call_output"],
            "programmatic function continuation order changed",
        )
        program = parse_tool_output(second_input[2])
        require(program.get("ok") is True, "programmatic output was invalid")
        require(
            program.get("result")
            == {"fact": "bounded functional snippet", "product": 899},
            "programmatic final value was invalid",
        )
        expected_lifecycles = []
        for context_index in range(1, 5):
            expected_lifecycles.extend(
                [
                    ("POST", "/sandboxes"),
                    ("POST", "/contexts"),
                    ("POST", "/execute"),
                    ("DELETE", "/contexts/programmatic-context-{}".format(context_index)),
                    ("DELETE", "/sandboxes/functional-programmatic"),
                ]
            )
        require(
            [(event.get("method"), event.get("path")) for event in sandbox_events]
            == expected_lifecycles,
            "programmatic E2B lifecycle order changed",
        )

    def web_search_single(self) -> None:
        with self.state.lock:
            web_before = len(self.state.web_requests)
            sandbox_before = len(self.state.sandbox_requests)
        status, body = self.request(
            {
                "model": "smart",
                "input": "functional-web-only: search once",
                "tools": [{"type": "web_search"}],
                "stream": False,
            }
        )
        require(status == 200, "Web Search case did not return HTTP 200")
        require(body.get("id") == "resp_web_final", "Web Search case returned the wrong round")
        require(
            body.get("usage")
            == {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
            "Web Search aggregate usage was incorrect",
        )
        with self.state.lock:
            requests = list(self.state.model_requests["web-search-single"])
            web_count = len(self.state.web_requests) - web_before
            sandbox_count = len(self.state.sandbox_requests) - sandbox_before
        require(len(requests) == 2, "Web Search case did not make two model rounds")
        require(web_count == 1, "Web Search case did not invoke exactly once")
        require(sandbox_count == 0, "Web Search case unexpectedly invoked Sandbox")
        canonical = requests[1].get("input")
        require(isinstance(canonical, list) and len(canonical) == 3, "Web Search history was incomplete")
        require(canonical[2].get("call_id") == "web-only-1", "Web Search call ID changed")

    def streaming_tool_runtime(self) -> None:
        with self.state.lock:
            model_before = len(self.state.model_requests["web-search-single"])
            web_before = len(self.state.web_requests)
        status, events = post_sse(
            self.base_url + "/v1/responses",
            {
                "model": "smart",
                "input": "functional-web-only: stream after search",
                "tools": [{"type": "web_search"}],
                "stream": True,
            },
        )
        require(status == 200, "streaming Tool Runtime did not return HTTP 200")
        event_types = [event.get("type") for event in events]
        require("response.created" in event_types, "streaming output omitted response.created")
        require(
            "response.output_text.delta" in event_types,
            "streaming output omitted the final text delta",
        )
        completed = [event for event in events if event.get("type") == "response.completed"]
        require(len(completed) == 1, "streaming output omitted its single terminal event")
        final = completed[0].get("response")
        require(isinstance(final, dict), "streaming terminal event omitted its Response")
        require(
            final.get("usage")
            == {
                "input_tokens": 8,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 13,
            },
            "streaming aggregate usage was incorrect",
        )
        encoded_events = json.dumps(events, separators=(",", ":"))
        require("web final answer" in encoded_events, "streaming final answer was absent")
        require("_agentgateway_" not in encoded_events, "streaming output leaked a reserved name")
        require("web-only-1" not in encoded_events, "streaming output leaked an internal call")
        with self.state.lock:
            requests = self.state.model_requests["web-search-single"][model_before:]
            web_count = len(self.state.web_requests) - web_before
        require(len(requests) == 2, "streaming case did not make two model rounds")
        require(
            all(request.get("stream") is False for request in requests),
            "an internal streaming model round was not buffered",
        )
        require(web_count == 1, "streaming case did not invoke Web Search exactly once")

    def multi_code_reuse(self) -> None:
        with self.state.lock:
            web_before = len(self.state.web_requests)
            sandbox_before = len(self.state.sandbox_requests)
        status, body = self.request(
            {
                "model": "smart",
                "input": "functional-multi-code: prove fresh variable contexts",
                "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
                "stream": False,
            }
        )
        require(status == 200, "multi-code case did not return HTTP 200")
        require(body.get("id") == "resp_code_final", "multi-code case returned the wrong round")
        require(
            body.get("usage")
            == {"input_tokens": 13, "output_tokens": 7, "total_tokens": 20},
            "multi-code aggregate usage was incorrect",
        )
        with self.state.lock:
            requests = list(self.state.model_requests["multi-code-reuse"])
            web_count = len(self.state.web_requests) - web_before
            batches = self.state.sandbox_requests[sandbox_before:]
        require(len(requests) == 2, "multi-code case did not make two model rounds")
        require(web_count == 0, "multi-code case unexpectedly invoked Web Search")
        require(len(batches) == 8, "same-Response code calls did not use one E2B Sandbox")
        require(
            [(event.get("method"), event.get("path")) for event in batches]
            == [
                ("POST", "/sandboxes"),
                ("POST", "/contexts"),
                ("POST", "/execute"),
                ("DELETE", "/contexts/multi-context-1"),
                ("POST", "/contexts"),
                ("POST", "/execute"),
                ("DELETE", "/contexts/multi-context-2"),
                ("DELETE", "/sandboxes/functional-multi"),
            ],
            "multi-code E2B lifecycle did not reuse one Sandbox in stable order",
        )
        canonical = requests[1].get("input")
        require(isinstance(canonical, list) and len(canonical) == 5, "multi-code history was incomplete")
        require(
            [canonical[3].get("call_id"), canonical[4].get("call_id")]
            == ["code-batch-1", "code-batch-2"],
            "multi-code canonical call IDs were unstable",
        )
        first_output = parse_tool_output(canonical[3])
        second_output = parse_tool_output(canonical[4])
        require(first_output.get("stdout") == "set\n", "first code result was unstable")
        require(second_output.get("stdout") == "isolated\n", "fresh code context result was unstable")

    def active_mode_rejections(self) -> None:
        with self.state.lock:
            model_before = sum(len(requests) for requests in self.state.model_requests.values())
        base = {
            "model": "smart",
            "input": "functional rejection",
            "tools": [{"type": "web_search"}],
            "stream": False,
        }
        variants: List[Dict[str, Any]] = []
        background_request = dict(base)
        background_request["background"] = True
        variants.append(background_request)
        conversation_request = dict(base)
        conversation_request["conversation"] = "conv-functional"
        variants.append(conversation_request)
        mixed_request = dict(base)
        mixed_request["tools"] = [
            {"type": "web_search"},
            {"type": "function", "name": "client_owned", "parameters": {"type": "object"}},
        ]
        variants.append(mixed_request)
        untrusted_request = dict(base)
        untrusted_request["tools"] = [
            {
                "type": "function",
                "name": "_agentgateway_web_search",
                "parameters": {"type": "object"},
            }
        ]
        variants.append(untrusted_request)
        for request in variants:
            status, body = self.request(request)
            require(status == 400, "active-mode rejection did not return HTTP 400")
            error = body.get("error")
            require(isinstance(error, dict), "active-mode rejection omitted the error object")
            require(
                error.get("code") == "managed_tool_request_invalid",
                "active-mode rejection returned the wrong error code",
            )
        with self.state.lock:
            model_after = sum(len(requests) for requests in self.state.model_requests.values())
        require(model_after == model_before, "an active-mode rejection reached the model")

    def unmanaged_function_passthrough(self) -> None:
        with self.state.lock:
            web_before = len(self.state.web_requests)
            sandbox_before = len(self.state.sandbox_requests)
        status, body = self.request(
            {
                "model": "smart",
                "input": "client-owned function",
                "tools": [
                    {
                        "type": "function",
                        "name": "client_owned",
                        "description": "A client-managed function.",
                        "parameters": {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    }
                ],
                "stream": False,
            }
        )
        require(status == 200, "unmanaged function passthrough did not return HTTP 200")
        output = body.get("output")
        require(isinstance(output, list) and len(output) == 1, "unmanaged output shape changed")
        require(output[0].get("name") == "client_owned", "unmanaged function name changed")
        require(output[0].get("call_id") == "client-call-1", "unmanaged call ID changed")
        with self.state.lock:
            requests = list(self.state.model_requests["unmanaged-function-passthrough"])
            web_count = len(self.state.web_requests) - web_before
            sandbox_count = len(self.state.sandbox_requests) - sandbox_before
        require(len(requests) == 1, "unmanaged function made more than one model call")
        require(web_count == 0 and sandbox_count == 0, "unmanaged function reached a managed backend")
        forwarded_tools = requests[0].get("tools")
        require(
            isinstance(forwarded_tools, list)
            and len(forwarded_tools) == 1
            and forwarded_tools[0].get("name") == "client_owned",
            "unmanaged function declaration was not passed through",
        )


    def tool_search_deferred(self) -> None:
        with self.state.lock:
            catalog_before = len(self.state.catalog_requests)
        client_tools = [
            {"type": "tool_search"},
            {
                "type": "function",
                "name": "catalog_lookup",
                "description": "Look up stock for a catalog SKU.",
                "parameters": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
                "defer_loading": True,
            },
        ]
        status, body = self.request(
            {
                "model": "smart",
                "input": "functional-tool-search catalog stock",
                "tools": client_tools,
                "stream": False,
            }
        )
        require(status == 200, "deferred tool search did not return HTTP 200")
        output = body.get("output")
        require(
            isinstance(output, list) and len(output) == 1,
            "the client saw more than the final answer",
        )
        require(output[0].get("type") == "message", "the final output item was not a message")
        require(
            output[0]["content"][0]["text"] == "deferred tool answer",
            "the final answer text changed",
        )
        serialized = json.dumps(body, separators=(",", ":"), sort_keys=True)
        require(
            "_agentgateway" not in serialized,
            "a reserved internal name reached the client",
        )
        require(
            body.get("tools") == client_tools,
            "the client tool declarations did not round-trip verbatim",
        )
        usage = body.get("usage")
        require(
            isinstance(usage, dict) and usage.get("total_tokens") == 30,
            "usage was not aggregated across the three model rounds",
        )

        with self.state.lock:
            requests = list(self.state.model_requests["tool-search-deferred"])
            catalog_count = len(self.state.catalog_requests) - catalog_before
        require(len(requests) == 3, "the deferred tool loop did not take three model rounds")

        first_tools = requests[0].get("tools")
        require(
            isinstance(first_tools, list) and len(first_tools) == 1,
            "the deferred tool was not withheld from the first round",
        )
        require(
            first_tools[0].get("name") == "_agentgateway_tool_search",
            "the first round did not declare only the search function",
        )
        index = first_tools[0].get("description")
        require(
            isinstance(index, str) and "catalog_lookup" in index,
            "the search index did not advertise the deferred tool",
        )

        second_tools = requests[1].get("tools")
        require(
            isinstance(second_tools, list) and len(second_tools) == 2,
            "the searched declaration was not appended for the second round",
        )
        require(
            second_tools[0].get("name") == "_agentgateway_tool_search"
            and second_tools[1].get("name") == "catalog_lookup",
            "the loaded declaration was not appended after the cached prefix",
        )
        search_output = next(
            (
                item
                for item in requests[1].get("input", [])
                if isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and item.get("call_id") == "search-call-1"
            ),
            None,
        )
        require(search_output is not None, "the search result was not replayed to the model")
        reported = parse_tool_output(search_output)
        require(reported.get("ok") is True, "the search result was not a success envelope")
        require(
            any(
                isinstance(tool, dict) and tool.get("name") == "catalog_lookup"
                for tool in reported.get("tools", [])
            ),
            "the search result did not report the matching tool",
        )
        require(catalog_count == 1, "the loaded tool was not executed exactly once")


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_gateway_attempt(
    binary: Path,
    selected_cases: Sequence[str],
    state: MockState,
    servers: Sequence[MockServer],
) -> None:
    ports, reservations = reserve_distinct_loopback_ports(2)
    gateway_port, readiness_port = ports
    try:
        with tempfile.TemporaryDirectory(prefix="agentgateway-functional-") as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                functional_config(
                    gateway_port,
                    readiness_port,
                    servers[0].port,
                    servers[1].port,
                    servers[2].port,
                    servers[3].port,
                ),
                encoding="utf-8",
            )
            log_path = Path(temp_dir) / "agentgateway.log"
            with log_path.open("wb") as child_log:
                process = None
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
                    methods = {
                        "dual-tool-overlap": cases.dual_tool_overlap,
                        "programmatic-server-tools": cases.programmatic_server_tools,
                        "web-search-single": cases.web_search_single,
                        "streaming-tool-runtime": cases.streaming_tool_runtime,
                        "multi-code-reuse": cases.multi_code_reuse,
                        "active-mode-rejections": cases.active_mode_rejections,
                        "unmanaged-function-passthrough": cases.unmanaged_function_passthrough,
                        "tool-search-deferred": cases.tool_search_deferred,
                    }
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
    verify_barrier_contract()

    state = MockState()
    servers: List[MockServer] = []
    try:
        for handler in (ModelHandler, WebSearchHandler, SandboxHandler, CatalogHandler):
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
    parser.add_argument(
        "--list-cases", action="store_true", help="list selectable case names and exit"
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
    print("PASS {} functional case(s)".format(len(selected_cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
