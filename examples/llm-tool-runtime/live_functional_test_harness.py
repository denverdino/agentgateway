#!/usr/bin/env python3
"""Shared infrastructure for real-backend functional test applications."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


MAX_HTTP_BODY_BYTES = 2 * 1024 * 1024
STARTUP_TIMEOUT_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 150.0
MAX_GATEWAY_START_ATTEMPTS = 3
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


def dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    return value


def load_known_dotenv(
    path: Path,
    values: Dict[str, str],
    allowed_keys: Sequence[str],
) -> None:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return
    allowed = set(allowed_keys)
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
        value = dotenv_value(raw_value)
        if value:
            values[key] = value


def live_child_environment(overrides: Mapping[str, str]) -> Dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "NO_COLOR": "1",
        "RUST_BACKTRACE": "0",
        "RUST_LOG": "error",
    }
    environment.update(overrides)
    for key in ("TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        if os.environ.get(key):
            environment[key] = os.environ[key]
    return environment


def tool_success_counts(metrics: str) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for line in metrics.splitlines():
        match = METRIC_LINE_PATTERN.match(line.strip())
        if not match:
            continue
        labels = {key: value for key, value in METRIC_LABEL_PATTERN.findall(match.group(1))}
        if labels.get("outcome") != "success" or "tool" not in labels or "backend" not in labels:
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
        headers={"authorization": "Bearer " + token, "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = read_bounded(response)
    except urllib.error.HTTPError as error:
        status = error.code
        body = read_bounded(error)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise HarnessFailure("live request failed at the HTTP transport") from error
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessFailure("live endpoint returned invalid JSON") from error
    require(isinstance(value, dict), "live endpoint returned a non-object JSON value")
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
        raise HarnessFailure("live streaming request failed at the HTTP transport") from error
    require(content_type == "text/event-stream", "streaming response was not SSE")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HarnessFailure("live SSE endpoint returned non-UTF-8 content") from error
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
            raise HarnessFailure("live endpoint returned invalid SSE JSON") from error
        require(isinstance(event, dict), "live SSE data was not an object")
        require(event.get("type") == event_name, "live SSE event name did not match its type")
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
    attempt: Callable[[], None],
    max_attempts: int = MAX_GATEWAY_START_ATTEMPTS,
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
