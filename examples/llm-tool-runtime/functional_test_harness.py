#!/usr/bin/env python3
"""Shared process, loopback HTTP, and lifecycle helpers for functional harnesses."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MAX_HTTP_BODY_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 5.0
STARTUP_TIMEOUT_SECONDS = 15.0
FUNCTIONAL_CLIENT_TOKEN = "functional-client-token"
MAX_GATEWAY_START_ATTEMPTS = 3
METRIC_LINE_PATTERN = re.compile(
    r'^agentgateway_tool_runtime_calls_total\{([^}]*)\}\s+([0-9]+(?:\.[0-9]+)?)$'
)
METRIC_LABEL_PATTERN = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class HarnessFailure(Exception):
    """A content-free functional assertion failure."""


class GatewayStartupFailure(HarnessFailure):
    """A retryable, content-free gateway startup or identity failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessFailure(message)


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

    def require_contract(self, path: str, auth_header: str, auth_value: str) -> None:
        require(self.path == path, "mock request used an unexpected fixed path")
        require(self.headers.get(auth_header) == auth_value, "mock request used unexpected authentication")
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


class MockServer:
    def __init__(self, handler: Any, state: Any) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.daemon_threads = True
        self.server.state = state  # type: ignore[attr-defined]
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )

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
        close_reservations(reservations)
        raise


def close_reservations(reservations: Iterable[socket.socket]) -> None:
    for reservation in reservations:
        reservation.close()


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
    request = urllib.request.Request(url, data=encoded, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read(MAX_HTTP_BODY_BYTES + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read(MAX_HTTP_BODY_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("public functional request failed at the HTTP transport") from error
    require(len(body) <= MAX_HTTP_BODY_BYTES, "public response exceeded the harness body bound")
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessFailure("public response was not valid JSON") from error
    require(isinstance(value, dict), "public response JSON was not an object")
    return status, value


def post_status(url: str, payload: Mapping[str, Any], client_token: Optional[str]) -> int:
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


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)
