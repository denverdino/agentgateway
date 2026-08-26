#!/usr/bin/env python3
"""Build and idempotently deploy the WeatherAPI Remote MCP Web Function."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


REQUIRED_KEYS = (
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "ALIBABA_CLOUD_REGION",
    "WEATHERAPI_KEY",
)
OPTIONAL_KEYS = (
    "ALIBABA_CLOUD_SECURITY_TOKEN",
    "ALIBABA_CLOUD_FC_ENDPOINT",
    "FC_WEATHER_MCP_FUNCTION_NAME",
    "FC_WEATHER_MCP_TRIGGER_NAME",
    "FC_WEATHER_MCP_TOKEN",
)
MANAGED_DOTENV_KEYS = ("FC_WEATHER_MCP_URL", "FC_WEATHER_MCP_TOKEN")
FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
TRIGGER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
ARCHIVE_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
ARCHIVE_FILES = frozenset(("bootstrap", "server.mjs", "package.json", "package-lock.json"))
FC_CONNECT_TIMEOUT_MS = 10_000
FC_READ_TIMEOUT_MS = 120_000


class DeploymentError(Exception):
    """A deployment failure whose message never contains credentials."""


def _dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    return value


def load_allowlisted_dotenv(path: Path, values: dict[str, str]) -> None:
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


@dataclass(frozen=True)
class DeploymentSettings:
    access_key_id: str = field(repr=False)
    access_key_secret: str = field(repr=False)
    region: str
    weatherapi_key: str = field(repr=False)
    ingress_token: str = field(repr=False)
    function_name: str
    trigger_name: str
    security_token: str | None = field(default=None, repr=False)
    endpoint: str | None = None

    @classmethod
    def load(cls, dotenv: Path, environment: Mapping[str, str]) -> "DeploymentSettings":
        values = {
            key: value
            for key in REQUIRED_KEYS + OPTIONAL_KEYS
            if (value := environment.get(key))
        }
        load_allowlisted_dotenv(dotenv, values)
        missing = sorted(key for key in REQUIRED_KEYS if not values.get(key))
        if missing:
            raise DeploymentError("missing required configuration: " + ", ".join(missing))
        function_name = values.get(
            "FC_WEATHER_MCP_FUNCTION_NAME", "agentgateway-weatherapi-mcp"
        )
        trigger_name = values.get(
            "FC_WEATHER_MCP_TRIGGER_NAME", "weatherapi-mcp-http"
        )
        if not FUNCTION_NAME_PATTERN.fullmatch(function_name):
            raise DeploymentError("FC_WEATHER_MCP_FUNCTION_NAME is invalid")
        if not TRIGGER_NAME_PATTERN.fullmatch(trigger_name):
            raise DeploymentError("FC_WEATHER_MCP_TRIGGER_NAME is invalid")
        return cls(
            access_key_id=values["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=values["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            region=values["ALIBABA_CLOUD_REGION"],
            weatherapi_key=values["WEATHERAPI_KEY"],
            ingress_token=values.get("FC_WEATHER_MCP_TOKEN")
            or secrets.token_urlsafe(32),
            function_name=function_name,
            trigger_name=trigger_name,
            security_token=values.get("ALIBABA_CLOUD_SECURITY_TOKEN"),
            endpoint=values.get("ALIBABA_CLOUD_FC_ENDPOINT"),
        )


def function_input(settings: DeploymentSettings, encoded_zip: str) -> dict[str, Any]:
    return {
        "code": {"zipFile": encoded_zip},
        "customRuntimeConfig": {"command": ["./bootstrap"], "port": 9000},
        "description": "Authenticated WeatherAPI Remote MCP server for AgentGateway",
        "environmentVariables": {
            "WEATHERAPI_KEY": settings.weatherapi_key,
            "INGRESS_BEARER_TOKEN": settings.ingress_token,
        },
        "handler": "index.handler",
        "instanceConcurrency": 8,
        "internetAccess": True,
        "memorySize": 512,
        "runtime": "custom.debian10",
        "timeout": 30,
    }


def trigger_input(settings: DeploymentSettings) -> dict[str, Any]:
    return {
        "description": "Remote MCP Streamable HTTP endpoint",
        "triggerName": settings.trigger_name,
        "triggerType": "http",
        "triggerConfig": {
            "authType": "anonymous",
            "disableURLInternet": False,
            "methods": ["GET", "POST"],
        },
    }


def _archive_paths(source_directory: Path) -> list[Path]:
    paths = [source_directory / name for name in ARCHIVE_FILES]
    node_modules = source_directory / "node_modules"
    if node_modules.is_dir():
        paths.extend(path for path in node_modules.rglob("*") if path.is_file())
    missing = [path.name for path in paths[: len(ARCHIVE_FILES)] if not path.is_file()]
    if missing:
        raise DeploymentError("deployment source is incomplete: " + ", ".join(sorted(missing)))
    return sorted(paths, key=lambda path: path.relative_to(source_directory).as_posix())


def build_archive(source_directory: Path, destination: Path) -> None:
    source_directory = source_directory.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _archive_paths(source_directory):
            relative = path.relative_to(source_directory).as_posix()
            info = zipfile.ZipInfo(relative, ARCHIVE_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if relative == "bootstrap" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes())


def normalize_public_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise DeploymentError("Function Compute returned an invalid public URL")
    path = parsed.path.rstrip("/") + "/mcp"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def update_dotenv(path: Path, public_url: str, ingress_token: str) -> None:
    try:
        original = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        original = ""
    except OSError as error:
        raise DeploymentError("could not read the dotenv file") from error
    replacements = {
        "FC_WEATHER_MCP_URL": public_url,
        "FC_WEATHER_MCP_TOKEN": ingress_token,
    }
    output: list[str] = []
    found: set[str] = set()
    for line in original.splitlines():
        candidate = line.strip()
        if candidate.startswith("export "):
            candidate = candidate[len("export ") :].lstrip()
        key = candidate.split("=", 1)[0].strip() if "=" in candidate else ""
        if key in replacements:
            if key not in found:
                output.append(f"{key}={replacements[key]}")
                found.add(key)
        else:
            output.append(line)
    for key in MANAGED_DOTENV_KEYS:
        if key not in found:
            output.append(f"{key}={replacements[key]}")
    contents = "\n".join(output).rstrip("\n") + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".env.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(contents)
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sdk_models(data: Mapping[str, Any], update: bool = False) -> Any:
    from alibabacloud_fc20230330 import models

    code = models.InputCodeLocation(zip_file=data["code"]["zipFile"])
    runtime = models.CustomRuntimeConfig(
        command=data["customRuntimeConfig"]["command"],
        port=data["customRuntimeConfig"]["port"],
    )
    input_type = models.UpdateFunctionInput if update else models.CreateFunctionInput
    arguments = {
        "code": code,
        "custom_runtime_config": runtime,
        "description": data["description"],
        "environment_variables": data["environmentVariables"],
        "handler": data["handler"],
        "instance_concurrency": data["instanceConcurrency"],
        "internet_access": data["internetAccess"],
        "memory_size": data["memorySize"],
        "runtime": data["runtime"],
        "timeout": data["timeout"],
    }
    return input_type(**arguments)


def create_fc_client(settings: DeploymentSettings) -> Any:
    from alibabacloud_fc20230330.client import Client
    from alibabacloud_tea_openapi import models as open_api_models

    configuration = fc_client_configuration(settings, open_api_models.Config)
    return Client(configuration)


def fc_client_configuration(settings: DeploymentSettings, config_type: Any) -> Any:
    configuration = config_type(
        access_key_id=settings.access_key_id,
        access_key_secret=settings.access_key_secret,
        security_token=settings.security_token,
        connect_timeout=FC_CONNECT_TIMEOUT_MS,
        read_timeout=FC_READ_TIMEOUT_MS,
    )
    configuration.region_id = settings.region
    if (endpoint := configured_fc_endpoint(settings)) is not None:
        configuration.endpoint = endpoint
    return configuration


def configured_fc_endpoint(settings: DeploymentSettings) -> str | None:
    """Return only an explicit override; otherwise use the SDK regional map."""
    return settings.endpoint


def _is_not_found(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    data = getattr(error, "data", None)
    return status_code == 404 or (
        isinstance(data, dict) and data.get("statusCode") == 404
    )


def deploy(settings: DeploymentSettings, encoded_zip: str) -> str:
    from alibabacloud_fc20230330 import models

    client = create_fc_client(settings)
    desired_function = function_input(settings, encoded_zip)
    try:
        client.get_function(settings.function_name, models.GetFunctionRequest())
    except Exception as error:
        if not _is_not_found(error):
            raise DeploymentError("could not inspect the Function Compute function") from error
        create_body = _sdk_models(desired_function)
        create_body.function_name = settings.function_name
        client.create_function(models.CreateFunctionRequest(body=create_body))
    else:
        client.update_function(
            settings.function_name,
            models.UpdateFunctionRequest(body=_sdk_models(desired_function, update=True)),
        )

    desired_trigger = trigger_input(settings)
    trigger_config_json = json.dumps(
        desired_trigger["triggerConfig"], separators=(",", ":"), sort_keys=True
    )
    try:
        trigger_response = client.get_trigger(settings.function_name, settings.trigger_name)
    except Exception as error:
        if not _is_not_found(error):
            raise DeploymentError("could not inspect the Function Compute trigger") from error
        trigger_response = client.create_trigger(
            settings.function_name,
            models.CreateTriggerRequest(
                body=models.CreateTriggerInput(
                    description=desired_trigger["description"],
                    trigger_config=trigger_config_json,
                    trigger_name=settings.trigger_name,
                    trigger_type="http",
                )
            ),
        )
    else:
        trigger_response = client.update_trigger(
            settings.function_name,
            settings.trigger_name,
            models.UpdateTriggerRequest(
                body=models.UpdateTriggerInput(
                    description=desired_trigger["description"],
                    trigger_config=trigger_config_json,
                )
            ),
        )
    http_trigger = getattr(getattr(trigger_response, "body", None), "http_trigger", None)
    public_url = getattr(http_trigger, "url_internet", None)
    if not isinstance(public_url, str):
        refreshed = client.get_trigger(settings.function_name, settings.trigger_name)
        http_trigger = getattr(getattr(refreshed, "body", None), "http_trigger", None)
        public_url = getattr(http_trigger, "url_internet", None)
    if not isinstance(public_url, str):
        raise DeploymentError("Function Compute did not return a public trigger URL")
    return normalize_public_url(public_url)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotenv", type=Path, default=repository_root() / ".env")
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).with_name("weather-mcp-fc.zip"),
    )
    parser.add_argument("--skip-npm-ci", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    source_directory = Path(__file__).resolve().parent
    settings = DeploymentSettings.load(arguments.dotenv, os.environ)
    if not arguments.skip_npm_ci:
        try:
            subprocess.run(
                ["npm", "ci", "--omit=dev", "--ignore-scripts"],
                cwd=source_directory,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise DeploymentError("npm production dependency installation failed") from error
    build_archive(source_directory, arguments.archive)
    encoded_zip = base64.b64encode(arguments.archive.read_bytes()).decode("ascii")
    public_url = deploy(settings, encoded_zip)
    update_dotenv(arguments.dotenv, public_url, settings.ingress_token)
    print(f"Deployed {settings.function_name} in {settings.region}")
    print(f"MCP URL: {public_url}")
    print("Updated FC_WEATHER_MCP_URL and FC_WEATHER_MCP_TOKEN in the dotenv file")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DeploymentError as error:
        raise SystemExit(f"deployment failed: {error}") from None
