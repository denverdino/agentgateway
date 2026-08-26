#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("live_functional_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("live_functional_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load live functional test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveFunctionalTestUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_load_configuration_prefers_environment_and_reads_allowlisted_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv = Path(temp_dir) / ".env"
            dotenv.write_text(
                "OPENAI_API_KEY=dotenv-openai\n"
                "export FC_WEB_SEARCH_URL='https://search.example.test/invoke'\n"
                'FC_WEB_SEARCH_TOKEN="search-token"\n'
                "E2B_API_KEY=e2b-key\n"
                "E2B_API_URL=https://api.e2b.example.test\n"
                "E2B_DOMAIN=sandbox.e2b.example.test\n"
                "FC_WEATHER_MCP_URL=https://weather.example.test/mcp\n"
                "FC_WEATHER_MCP_TOKEN=weather-mcp-token\n"
                "UNRELATED_SECRET=must-not-load\n",
                encoding="utf-8",
            )
            environment = {"OPENAI_API_KEY": "environment-openai"}

            config = self.module.LiveConfiguration.load(dotenv, environment)

            self.assertEqual(config.openai_api_key, "environment-openai")
            self.assertEqual(config.web_search_url, "https://search.example.test/invoke")
            self.assertEqual(config.weather_mcp_url, "https://weather.example.test/mcp")
            self.assertEqual(config.weather_mcp_token, "weather-mcp-token")
            self.assertEqual(config.model, "qwen3.6-flash")
            child_environment = config.child_environment("client-token")
            self.assertEqual(child_environment["OPENAI_API_KEY"], "environment-openai")
            self.assertNotIn("UNRELATED_SECRET", child_environment)

    def test_load_configuration_reports_missing_names_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv = Path(temp_dir) / ".env"
            dotenv.write_text("OPENAI_API_KEY=secret-value\n", encoding="utf-8")

            with self.assertRaisesRegex(
                self.module.HarnessFailure,
                "missing required configuration: E2B_API_KEY, E2B_API_URL, E2B_DOMAIN, "
                "FC_WEATHER_MCP_TOKEN, FC_WEATHER_MCP_URL, FC_WEB_SEARCH_TOKEN, "
                "FC_WEB_SEARCH_URL",
            ) as raised:
                self.module.LiveConfiguration.load(dotenv, {})

            self.assertNotIn("secret-value", str(raised.exception))

    def test_functional_config_uses_named_gateway_and_provider_reference(self) -> None:
        rendered = self.module.functional_config(41234, 41235)

        self.assertIn("gateways:\n  default:\n    port: 41234", rendered)
        self.assertIn("llm:\n  gateways: default", rendered)
        self.assertIn("- name: bailian\n    provider: openAI", rendered)
        self.assertIn("apiKey: $OPENAI_API_KEY", rendered)
        self.assertIn("reference: bailian", rendered)
        self.assertIn("apiKey: $E2B_API_KEY", rendered)
        self.assertIn("url: $FC_WEB_SEARCH_URL", rendered)
        self.assertNotIn("database:", rendered)
        self.assertNotIn("ui:", rendered)

    def test_extract_output_text_reads_response_message_content(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": "second"},
                    ],
                }
            ]
        }

        self.assertEqual(self.module.extract_output_text(response), "first\nsecond")

    def test_case_payloads_expose_both_real_backends(self) -> None:
        cases = self.module.case_payloads(
            "qwen3.6-flash",
            "https://weather.example.test/mcp",
            "weather-mcp-token",
        )

        self.assertEqual(
            set(cases),
            {
                "web-search",
                "code-interpreter",
                "combined",
                "streaming-tool-runtime",
                "remote-mcp-weather",
            },
        )
        self.assertEqual(cases["web-search"]["tools"], [{"type": "web_search"}])
        self.assertEqual(
            cases["code-interpreter"]["tools"],
            [{"type": "code_interpreter", "container": {"type": "auto"}}],
        )
        self.assertTrue(cases["combined"]["parallel_tool_calls"])
        self.assertEqual(len(cases["combined"]["tools"]), 2)
        self.assertEqual(
            cases["streaming-tool-runtime"]["tools"], [{"type": "web_search"}]
        )
        self.assertTrue(cases["streaming-tool-runtime"]["stream"])
        self.assertEqual(
            cases["remote-mcp-weather"]["tools"],
            [
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://weather.example.test/mcp",
                    "authorization": "weather-mcp-token",
                    "allowed_tools": ["get_forecast", "get_current_weather"],
                    "require_approval": "auto",
                }
            ],
        )
        self.assertFalse(cases["remote-mcp-weather"]["stream"])
        self.assertEqual(
            cases["remote-mcp-weather"]["tool_choice"],
            {
                "type": "mcp",
                "server_label": "weather",
                "name": "get_current_weather",
            },
        )
        self.assertIs(cases["remote-mcp-weather"]["enable_thinking"], False)
        for payload in cases.values():
            self.assertEqual(payload["model"], "qwen3.6-flash")
        for case_name in (
            "web-search",
            "code-interpreter",
            "combined",
            "remote-mcp-weather",
        ):
            self.assertFalse(cases[case_name]["stream"])

    def test_final_streaming_response_requires_lifecycle_delta_and_completion(self) -> None:
        response = {"status": "completed", "output": []}
        events = [
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": response},
        ]

        self.assertIs(self.module.final_streaming_response(events), response)

        for invalid_events in (
            events[1:],
            [events[0], events[2]],
            [events[0], events[1]],
            [events[0], events[1], events[2], events[2]],
        ):
            with self.assertRaises(self.module.HarnessFailure):
                self.module.final_streaming_response(invalid_events)

    def test_tool_success_counts_parses_content_free_metrics(self) -> None:
        metrics = "\n".join(
            (
                "# HELP agentgateway_tool_runtime_calls Total calls",
                'agentgateway_tool_runtime_calls_total{backend="e2b",outcome="success",'
                'tool="code_interpreter"} 2',
                'agentgateway_tool_runtime_calls_total{tool="web_search",'
                'backend="http",outcome="success"} 1',
                'agentgateway_tool_runtime_calls_total{backend="e2b",outcome="executing",'
                'tool="code_interpreter"} 2',
            )
        )

        self.assertEqual(
            self.module.tool_success_counts(metrics),
            {
                ("code_interpreter", "e2b"): 2,
                ("web_search", "http"): 1,
            },
        )

    def test_gateway_startup_retries_are_bounded(self) -> None:
        attempts = []

        def eventually_starts() -> None:
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                raise self.module.GatewayStartupFailure("port race")

        self.module.run_with_startup_retries(eventually_starts, max_attempts=3)
        self.assertEqual(attempts, [1, 2, 3])

        attempts.clear()

        def never_starts() -> None:
            attempts.append(len(attempts) + 1)
            raise self.module.GatewayStartupFailure("port race")

        with self.assertRaisesRegex(
            self.module.HarnessFailure, "startup failed after 3 attempts"
        ):
            self.module.run_with_startup_retries(never_starts, max_attempts=3)
        self.assertEqual(attempts, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
