#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("openai_sdk_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("openai_sdk_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load OpenAI SDK test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, text, tools=None, status="completed", extra=None):
        self.status = status
        self.output_text = text
        self.tools = tools
        self._extra = extra or {}

    def model_dump(self, **_kwargs):
        value = {
            "status": self.status,
            "output_text": self.output_text,
            "tools": self.tools,
        }
        value.update(self._extra)
        return value


class FakeResponses:
    def __init__(self, response, stream_events=None):
        self.response = response
        self.stream_events = stream_events
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if request.get("stream"):
            if self.stream_events is not None:
                return iter(self.stream_events)
            text = self.response.output_text
            return iter(
                (
                    SimpleNamespace(type="response.created"),
                    SimpleNamespace(type="response.output_text.delta", delta=text),
                    SimpleNamespace(type="response.completed", response=self.response),
                )
            )
        return self.response


class OpenAISDKTestUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def payloads(self):
        return self.module.case_payloads(
            "qwen3.8-flash",
            "https://weather.example.test/mcp",
            "weather-secret",
        )

    def response_for(self, case_name, payload):
        marker = self.module.expected_marker(case_name)
        text = marker + " result"
        tools = None
        if case_name == "programmatic-mcp-weather":
            text = marker + " highest 2026-09-01 31 C lowest 2026-09-02 19 C"
        elif case_name == "tool-search-weather-mcp":
            text = marker + " observed 25 C"
            tools = []
            for tool in payload["tools"]:
                sanitized = dict(tool)
                sanitized.pop("authorization", None)
                tools.append(sanitized)
        return FakeResponse(text, tools=tools)

    def test_defines_every_live_functional_case(self) -> None:
        self.assertEqual(
            self.module.CASE_NAMES,
            (
                "web-search",
                "code-interpreter",
                "combined",
                "programmatic-server-tools",
                "programmatic-mcp-weather",
                "streaming-tool-runtime",
                "remote-mcp-weather",
                "tool-search-weather-mcp",
            ),
        )

    def test_builds_specialized_tool_payloads(self) -> None:
        payloads = self.payloads()

        self.assertEqual(set(payloads), set(self.module.CASE_NAMES))
        self.assertTrue(payloads["combined"]["parallel_tool_calls"])
        self.assertEqual(
            payloads["programmatic-server-tools"]["tools"][-1],
            {"type": "programmatic_tool_calling"},
        )
        programmatic_mcp = payloads["programmatic-mcp-weather"]["tools"][0]
        self.assertEqual(programmatic_mcp["allowed_callers"], ["programmatic"])
        self.assertEqual(programmatic_mcp["allowed_tools"], ["get_forecast"])
        self.assertEqual(programmatic_mcp["authorization"], "weather-secret")
        self.assertFalse(payloads["remote-mcp-weather"]["enable_thinking"])
        deferred_mcp = payloads["tool-search-weather-mcp"]["tools"][1]
        self.assertTrue(deferred_mcp["defer_loading"])
        self.assertNotIn("tool_choice", payloads["tool-search-weather-mcp"])
        self.assertTrue(payloads["streaming-tool-runtime"]["stream"])

    def test_runs_all_cases_through_the_sdk(self) -> None:
        payloads = self.payloads()
        for case_name in self.module.CASE_NAMES:
            with self.subTest(case=case_name):
                response = self.response_for(case_name, payloads[case_name])
                responses = FakeResponses(response)
                client = SimpleNamespace(responses=responses)

                text = self.module.run_sdk_case(
                    client, case_name, payloads[case_name]
                )

                expected_request = dict(payloads[case_name])
                if "enable_thinking" in expected_request:
                    expected_request["extra_body"] = {
                        "enable_thinking": expected_request.pop("enable_thinking")
                    }
                self.assertEqual(text, response.output_text)
                self.assertEqual(responses.requests, [expected_request])

    def test_rejects_a_stream_without_a_terminal_response(self) -> None:
        payload = self.payloads()["streaming-tool-runtime"]
        responses = FakeResponses(
            FakeResponse("unused"),
            stream_events=(
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="partial"),
            ),
        )
        client = SimpleNamespace(responses=responses)

        with self.assertRaisesRegex(self.module.SDKTestFailure, "response.completed"):
            self.module.run_sdk_case(client, "streaming-tool-runtime", payload)

    def test_rejects_programmatic_weather_without_extrema(self) -> None:
        payload = self.payloads()["programmatic-mcp-weather"]
        response = FakeResponse("PROGRAMMATIC_MCP_WEATHER_3DAY_OK result")
        client = SimpleNamespace(responses=FakeResponses(response))

        with self.assertRaisesRegex(self.module.SDKTestFailure, "highest or lowest"):
            self.module.run_sdk_case(client, "programmatic-mcp-weather", payload)

    def test_rejects_deferred_tool_declaration_mismatch(self) -> None:
        payload = self.payloads()["tool-search-weather-mcp"]
        response = FakeResponse(
            "TOOL_SEARCH_WEATHER_MCP_OK observed 25 C",
            tools=[{"type": "tool_search"}],
        )
        client = SimpleNamespace(responses=FakeResponses(response))

        with self.assertRaisesRegex(self.module.SDKTestFailure, "did not round-trip"):
            self.module.run_sdk_case(client, "tool-search-weather-mcp", payload)

    def test_gateway_options_do_not_reuse_upstream_openai_values(self) -> None:
        with patch.dict(
            self.module.os.environ,
            {
                "OPENAI_API_KEY": "upstream-secret",
                "OPENAI_BASE_URL": "https://upstream.example.test/v1",
                "AGENTGATEWAY_API_KEY": "client-secret",
                "AGENTGATEWAY_BASE_URL": "http://127.0.0.1:4321/v1",
                "FC_WEATHER_MCP_URL": "https://weather.example.test/mcp",
                "FC_WEATHER_MCP_TOKEN": "weather-secret",
            },
            clear=True,
        ):
            args = self.module.parse_args([])

        self.assertEqual(args.api_key, "client-secret")
        self.assertEqual(args.base_url, "http://127.0.0.1:4321/v1")
        self.assertEqual(args.weather_mcp_url, "https://weather.example.test/mcp")
        self.assertEqual(args.weather_mcp_token, "weather-secret")


if __name__ == "__main__":
    unittest.main()
