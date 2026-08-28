#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, List, Optional


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
                "OPENAI_MODEL=environment-compatible-model\n"
                "OPENAI_BASE_URL=https://responses.example.test/v1\n"
                "UNRELATED_SECRET=must-not-load\n",
                encoding="utf-8",
            )
            environment = {"OPENAI_API_KEY": "environment-openai"}

            config = self.module.LiveConfiguration.load(dotenv, environment)

            self.assertEqual(config.openai_api_key, "environment-openai")
            self.assertEqual(config.web_search_url, "https://search.example.test/invoke")
            self.assertEqual(config.weather_mcp_url, "https://weather.example.test/mcp")
            self.assertEqual(config.weather_mcp_token, "weather-mcp-token")
            self.assertEqual(config.model, "environment-compatible-model")
            self.assertEqual(
                config.upstream_base_url, "https://responses.example.test/v1"
            )
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

    def _source_event(self, code: str) -> str:
        return json.dumps(
            {
                "level": "trace",
                "time": "2026-08-28T04:00:00.000000Z",
                "scope": "agentgateway::llm::tool_runtime::runner",
                "message": self.module.PROGRAM_SOURCE_MESSAGE,
                "code": code,
            }
        )

    def test_extract_program_sources_reads_only_source_events(self) -> None:
        logs = "\n".join(
            (
                self._source_event(
                    'result = tools.call("weather.get_forecast", '
                    '{"q": "Shanghai", "days": 3})\nprogram_output(result)'
                ),
                json.dumps(
                    {
                        "level": "debug",
                        "scope": "agentgateway::llm::tool_runtime::runner",
                        "message": "generated programmatic tool call code",
                        "code_bytes": 40,
                    }
                ),
                self._source_event(""),
                "not json at all",
            )
        )

        self.assertEqual(
            self.module.extract_program_sources(logs),
            [
                'result = tools.call("weather.get_forecast", '
                '{"q": "Shanghai", "days": 3})\nprogram_output(result)'
            ],
        )

    def test_read_program_sources_advances_past_only_complete_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "agentgateway.log"
            first = self._source_event("program_output(1)")
            second = self._source_event("program_output(2)")
            log_path.write_text(first + "\n" + second, encoding="utf-8")

            sources, offset = self.module.read_program_sources(log_path, 0)

            self.assertEqual(sources, ["program_output(1)"])
            self.assertEqual(offset, len(first) + 1)

            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

            self.assertEqual(
                self.module.read_program_sources(log_path, offset),
                (["program_output(2)"], len(first) + len(second) + 2),
            )

    def test_print_case_output_requires_a_program_for_the_weather_case(self) -> None:
        with self.assertRaisesRegex(
            self.module.HarnessFailure,
            "did not capture generated program source",
        ):
            self.module.print_case_output("programmatic-mcp-weather", "text", ())

        output = io.StringIO()
        with redirect_stdout(output):
            self.module.print_case_output(
                "programmatic-mcp-weather", "answer", ["program_output(1)"]
            )
            # A case that never runs a program has nothing to report.
            self.module.print_case_output("web-search", "plain", ())

        self.assertEqual(
            output.getvalue(),
            "PROGRAM SOURCE 1 (Python)\n"
            "program_output(1)\n"
            "END PROGRAM SOURCE 1\n"
            "answer\n"
            "plain\n",
        )

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
                "programmatic-server-tools",
                "programmatic-mcp-weather",
                "streaming-tool-runtime",
                "remote-mcp-weather",
                "tool-search-weather-mcp",
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
            cases["programmatic-server-tools"]["tools"],
            [
                {"type": "web_search", "allowed_callers": ["programmatic"]},
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto"},
                    "allowed_callers": ["programmatic"],
                },
                {"type": "programmatic_tool_calling"},
            ],
        )
        weather_program = cases["programmatic-mcp-weather"]
        self.assertEqual(
            weather_program["tools"],
            [
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://weather.example.test/mcp",
                    "authorization": "weather-mcp-token",
                    "allowed_tools": ["get_forecast"],
                    "allowed_callers": ["programmatic"],
                    "require_approval": "never",
                },
                {"type": "programmatic_tool_calling"},
            ],
        )
        self.assertIn("3-day", weather_program["input"])
        self.assertIn("highest daily maximum temperature", weather_program["input"])
        self.assertIn("lowest daily minimum temperature", weather_program["input"])
        self.assertIn("earliest date", weather_program["input"])
        self.assertNotIn("report each date", weather_program["input"])
        self.assertNotIn("future 5 days", weather_program["input"])
        self.assertIn("PROGRAMMATIC_MCP_WEATHER_3DAY_OK", weather_program["input"])
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
        tool_search_weather = cases["tool-search-weather-mcp"]
        self.assertEqual(
            tool_search_weather["tools"],
            [
                {"type": "tool_search"},
                {
                    "type": "mcp",
                    "server_label": "weather",
                    "server_url": "https://weather.example.test/mcp",
                    "authorization": "weather-mcp-token",
                    "allowed_tools": ["get_forecast", "get_current_weather"],
                    "require_approval": "never",
                    "defer_loading": True,
                },
            ],
        )
        # A forced tool on a deferred server is injected eagerly, which would let the case pass
        # without the model ever searching.
        self.assertNotIn("tool_choice", tool_search_weather)
        self.assertIn("TOOL_SEARCH_WEATHER_MCP_OK", tool_search_weather["input"])
        for payload in cases.values():
            self.assertEqual(payload["model"], "qwen3.6-flash")
        for case_name in (
            "web-search",
            "code-interpreter",
            "combined",
            "programmatic-server-tools",
            "programmatic-mcp-weather",
            "remote-mcp-weather",
            "tool-search-weather-mcp",
        ):
            self.assertFalse(cases[case_name]["stream"])

    def _run_programmatic_weather_case(self, output_text: str) -> str:
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": output_text,
                        }
                    ],
                }
            ],
        }
        metrics = iter(
            (
                "",
                'agentgateway_tool_runtime_calls_total{tool="remote_mcp",'
                'backend="remote_mcp",outcome="success"} 1',
            )
        )
        original_post_json = self.module.post_json
        original_fetch_metrics = self.module.fetch_metrics
        self.module.post_json = lambda *_args: (200, response)
        self.module.fetch_metrics = lambda _port: next(metrics)
        try:
            return self.module.run_case(
                "http://127.0.0.1:8080",
                15020,
                "client-token",
                "programmatic-mcp-weather",
                {"model": "test"},
            )
        finally:
            self.module.post_json = original_post_json
            self.module.fetch_metrics = original_fetch_metrics

    def test_run_case_accepts_programmatic_weather_extreme_days(self) -> None:
        text = self._run_programmatic_weather_case(
            "Highest daily maximum temperature: 2026-08-28, 35 C\n"
            "Lowest daily minimum temperature: 2026-08-27, 24 C\n"
            "PROGRAMMATIC_MCP_WEATHER_3DAY_OK"
        )

        self.assertIn("2026-08-28", text)
        self.assertIn("2026-08-27", text)

    def test_run_case_accepts_programmatic_weather_listing_with_extremes(self) -> None:
        text = self._run_programmatic_weather_case(
            "3-day forecast for Shanghai:\n"
            "- 2026-08-27: max 33 C, min 26.2 C\n"
            "- 2026-08-28: max 29.9 C, min 26 C\n"
            "- 2026-08-29: max 32.5 C, min 27.1 C\n"
            "Highest daily maximum temperature: 33 C on 2026-08-27\n"
            "Lowest daily minimum temperature: 26 C on 2026-08-28\n"
            "PROGRAMMATIC_MCP_WEATHER_3DAY_OK"
        )

        self.assertIn("2026-08-29", text)

    def test_run_case_rejects_programmatic_weather_daily_listing(self) -> None:
        with self.assertRaisesRegex(
            self.module.HarnessFailure,
            "highest or lowest temperature day",
        ):
            self._run_programmatic_weather_case(
                "2026-08-27: min 24 C, max 33 C\n"
                "2026-08-28: min 25 C, max 35 C\n"
                "2026-08-29: min 26 C, max 34 C\n"
                "PROGRAMMATIC_MCP_WEATHER_3DAY_OK"
            )

    def _run_tool_search_weather_case(
        self, output_text: str, echoed_tools: Optional[List[Any]] = None
    ) -> str:
        deferred = {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "https://weather.example.test/mcp",
            "authorization": "weather-mcp-token",
            "allowed_tools": ["get_current_weather"],
            "require_approval": "never",
            "defer_loading": True,
        }
        payload_tools = [{"type": "tool_search"}, deferred]
        if echoed_tools is None:
            sanitized = {
                key: value for key, value in deferred.items() if key != "authorization"
            }
            echoed_tools = [{"type": "tool_search"}, sanitized]
        response = {
            "status": "completed",
            "tools": echoed_tools,
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ],
        }
        metrics = iter(
            (
                "",
                'agentgateway_tool_runtime_calls_total{tool="remote_mcp",'
                'backend="remote_mcp",outcome="success"} 1',
            )
        )
        original_post_json = self.module.post_json
        original_fetch_metrics = self.module.fetch_metrics
        self.module.post_json = lambda *_args: (200, response)
        self.module.fetch_metrics = lambda _port: next(metrics)
        try:
            return self.module.run_case(
                "http://127.0.0.1:8080",
                15020,
                "client-token",
                "tool-search-weather-mcp",
                {"model": "test", "tools": payload_tools},
            )
        finally:
            self.module.post_json = original_post_json
            self.module.fetch_metrics = original_fetch_metrics

    def test_run_case_accepts_tool_search_weather_summary(self) -> None:
        text = self._run_tool_search_weather_case(
            "Shanghai is 30.2°C with patchy rain nearby.\nTOOL_SEARCH_WEATHER_MCP_OK"
        )

        self.assertIn("30.2", text)

    def test_run_case_rejects_tool_search_weather_without_observed_value(self) -> None:
        with self.assertRaisesRegex(
            self.module.HarnessFailure, "omitted an observed temperature"
        ):
            self._run_tool_search_weather_case(
                "I found a weather tool.\nTOOL_SEARCH_WEATHER_MCP_OK"
            )

    def test_run_case_rejects_tool_search_weather_losing_defer_loading(self) -> None:
        with self.assertRaisesRegex(self.module.HarnessFailure, "did not round-trip"):
            self._run_tool_search_weather_case(
                "Shanghai is 30.2°C.\nTOOL_SEARCH_WEATHER_MCP_OK",
                echoed_tools=[
                    {"type": "tool_search"},
                    {
                        "type": "mcp",
                        "server_label": "weather",
                        "server_url": "https://weather.example.test/mcp",
                        "allowed_tools": ["get_current_weather"],
                        "require_approval": "never",
                    },
                ],
            )

    def test_run_case_rejects_tool_search_weather_leaking_reserved_name(self) -> None:
        with self.assertRaisesRegex(
            self.module.HarnessFailure, "reserved internal name"
        ):
            self._run_tool_search_weather_case(
                "Shanghai is 30.2°C via _agentgateway_tool_search.\n"
                "TOOL_SEARCH_WEATHER_MCP_OK"
            )

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
