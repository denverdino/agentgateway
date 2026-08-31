#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("anthropic_live_functional_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("anthropic_live_functional_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Anthropic live functional test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnthropicLiveFunctionalTestUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_load_configuration_prefers_environment_and_allowlists_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv = Path(temp_dir) / ".env"
            dotenv.write_text(
                "ANTHROPIC_API_KEY=dotenv-anthropic\n"
                "FC_WEB_SEARCH_URL=https://search.example.test/invoke\n"
                "FC_WEB_SEARCH_TOKEN=search-token\n"
                "E2B_API_KEY=e2b-key\n"
                "E2B_API_URL=https://api.e2b.example.test\n"
                "E2B_DOMAIN=sandbox.e2b.example.test\n"
                "ANTHROPIC_MODEL=dotenv-model\n"
                "ANTHROPIC_BASE_URL=https://anthropic.example.test/v1\n"
                "UNRELATED_SECRET=must-not-load\n",
                encoding="utf-8",
            )
            environment = {
                "ANTHROPIC_API_KEY": "environment-anthropic",
                "AGENTGATEWAY_LIVE_MODEL": "environment-model",
                "AGENTGATEWAY_ANTHROPIC_BASE_URL": "https://override.example.test/v1",
            }

            config = self.module.LiveConfiguration.load(dotenv, environment)

            self.assertEqual(config.anthropic_api_key, "environment-anthropic")
            self.assertEqual(config.model, "environment-model")
            self.assertEqual(config.upstream_base_url, "https://override.example.test/v1")
            child = config.child_environment("client-token")
            self.assertEqual(child["ANTHROPIC_API_KEY"], "environment-anthropic")
            self.assertNotIn("UNRELATED_SECRET", child)

    def test_sdk_base_url_adds_v1_for_agentgateway_path_prefix(self) -> None:
        values = {
            "ANTHROPIC_API_KEY": "key",
            "ANTHROPIC_BASE_URL": "https://dashscope.example.test/apps/anthropic/",
            "FC_WEB_SEARCH_URL": "https://search.example.test/invoke",
            "FC_WEB_SEARCH_TOKEN": "token",
            "E2B_API_KEY": "e2b-key",
            "E2B_API_URL": "https://api.e2b.example.test",
            "E2B_DOMAIN": "sandbox.e2b.example.test",
        }

        config = self.module.LiveConfiguration(values)

        self.assertEqual(
            config.upstream_base_url,
            "https://dashscope.example.test/apps/anthropic/v1",
        )

    def test_default_model_and_base_url_match_native_anthropic(self) -> None:
        values = {
            "ANTHROPIC_API_KEY": "key",
            "FC_WEB_SEARCH_URL": "https://search.example.test/invoke",
            "FC_WEB_SEARCH_TOKEN": "token",
            "E2B_API_KEY": "e2b-key",
            "E2B_API_URL": "https://api.e2b.example.test",
            "E2B_DOMAIN": "sandbox.e2b.example.test",
        }

        config = self.module.LiveConfiguration(values)

        self.assertEqual(config.model, "claude-haiku-4-5-20251001")
        self.assertEqual(config.upstream_base_url, "https://api.anthropic.com/v1")

    def test_missing_configuration_reports_names_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv = Path(temp_dir) / ".env"
            dotenv.write_text("ANTHROPIC_API_KEY=secret-value\n", encoding="utf-8")

            with self.assertRaises(self.module.HarnessFailure) as raised:
                self.module.LiveConfiguration.load(dotenv, {})

            message = str(raised.exception)
            self.assertIn("E2B_API_KEY", message)
            self.assertIn("FC_WEB_SEARCH_URL", message)
            self.assertNotIn("secret-value", message)

    def test_functional_config_is_native_anthropic_messages(self) -> None:
        rendered = self.module.functional_config(41234, 41235, 41236)

        self.assertIn("gateways:\n  default:\n    port: 41234", rendered)
        self.assertIn("provider: anthropic", rendered)
        self.assertIn("apiKey: $ANTHROPIC_API_KEY", rendered)
        self.assertIn("baseUrl: $AGENTGATEWAY_ANTHROPIC_BASE_URL", rendered)
        self.assertNotIn("provider: openAI", rendered)
        self.assertNotIn("database:", rendered)
        self.assertNotIn("ui:", rendered)

    def test_case_payloads_use_only_messages_supported_tools(self) -> None:
        cases = self.module.case_payloads("claude-test")

        self.assertEqual(
            set(cases),
            {
                "web-search",
                "code-interpreter",
                "combined",
                "programmatic-web-search",
                "streaming-tool-runtime",
            },
        )
        self.assertEqual(
            cases["web-search"]["tools"],
            [{"type": "web_search_20250305", "name": "web_search"}],
        )
        self.assertEqual(
            cases["code-interpreter"]["tools"],
            [{"type": "code_execution_20250825", "name": "code_execution"}],
        )
        self.assertEqual(len(cases["combined"]["tools"]), 2)
        self.assertEqual(
            cases["programmatic-web-search"]["tools"],
            [
                {"type": "code_execution_20260120", "name": "code_execution"},
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "allowed_callers": ["code_execution_20260120"],
                },
            ],
        )
        self.assertTrue(cases["streaming-tool-runtime"]["stream"])
        for name, payload in cases.items():
            self.assertEqual(payload["model"], "claude-test", name)
            self.assertIn("max_tokens", payload, name)
            self.assertIsInstance(payload["messages"], list, name)
            self.assertNotIn("input", payload, name)

    def test_extract_message_text_reads_text_blocks(self) -> None:
        response = {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "server_tool_use", "name": "hidden"},
                {"type": "text", "text": "second"},
            ]
        }

        self.assertEqual(self.module.extract_message_text(response), "first\nsecond")

    def test_final_streaming_response_validates_messages_sse_lifecycle(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "STREAMING_"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "TOOL_RUNTIME_OK"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
            {"type": "message_stop"},
        ]

        response = self.module.final_streaming_response(events)

        self.assertEqual(self.module.extract_message_text(response), "STREAMING_TOOL_RUNTIME_OK")
        self.assertEqual(response["stop_reason"], "end_turn")
        invalid_sequences = (
            events[1:],
            events[:-1],
            events + [events[-1]],
            [event for event in events if event.get("type") != "content_block_stop"],
        )
        for invalid in invalid_sequences:
            with self.assertRaises(self.module.HarnessFailure):
                self.module.final_streaming_response(invalid)

    def test_sandbox_success_count_reads_execute_metric(self) -> None:
        metrics = "\n".join(
            (
                'agentgateway_tool_runtime_sandbox_operations_total{operation="execute",outcome="success"} 2',
                'agentgateway_tool_runtime_sandbox_operations_total{operation="cleanup",outcome="success"} 3',
                'agentgateway_tool_runtime_sandbox_operations_total{operation="execute",outcome="failure"} 4',
            )
        )

        self.assertEqual(self.module.sandbox_success_count(metrics), 2)

    def test_expected_markers_and_backends_cover_all_cases(self) -> None:
        expected = {
            "web-search": (("web_search", "http"),),
            "code-interpreter": (("code_interpreter", "e2b"),),
            "combined": (("web_search", "http"), ("code_interpreter", "e2b")),
            "programmatic-web-search": (("web_search", "http"),),
            "streaming-tool-runtime": (("web_search", "http"),),
        }
        for name in self.module.CASE_NAMES:
            self.assertTrue(self.module.expected_marker(name))
            self.assertEqual(self.module.expected_tool_backends(name), expected[name])

    def test_programmatic_response_rejects_structured_reserved_names(self) -> None:
        response = {
            "type": "message",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "PROGRAMMATIC_WEB_SEARCH_OK model mentioned "
                        "_agentgateway_programmatic_tool_calling"
                    ),
                }
            ],
            "stop_reason": "end_turn",
        }
        metrics = iter(
            (
                "",
                "\n".join((
                    'agentgateway_tool_runtime_calls_total{tool="web_search",backend="http",outcome="success"} 1',
                    'agentgateway_tool_runtime_sandbox_operations_total{operation="execute",outcome="success"} 1',
                )),
            )
        )
        original_post_json = self.module.post_json
        original_fetch_metrics = self.module.fetch_metrics
        self.module.post_json = lambda *_args: (200, response)
        self.module.fetch_metrics = lambda _port: next(metrics)
        try:
            text = self.module.run_case(
                "http://127.0.0.1:8080",
                15020,
                "client-token",
                "programmatic-web-search",
                {"model": "test"},
            )
        finally:
            self.module.post_json = original_post_json
            self.module.fetch_metrics = original_fetch_metrics
        self.assertIn("PROGRAMMATIC_WEB_SEARCH_OK", text)

        leaked = dict(response)
        leaked["content"] = [
            {"type": "text", "text": "PROGRAMMATIC_WEB_SEARCH_OK"},
            {
                "type": "tool_use",
                "id": "toolu_leaked",
                "name": "_agentgateway_hidden",
                "input": {},
            },
        ]
        metrics = iter(("", ""))
        self.module.post_json = lambda *_args: (200, leaked)
        self.module.fetch_metrics = lambda _port: next(metrics)
        try:
            with self.assertRaisesRegex(self.module.HarnessFailure, "reserved"):
                self.module.run_case(
                    "http://127.0.0.1:8080",
                    15020,
                    "client-token",
                    "programmatic-web-search",
                    {"model": "test"},
                )
        finally:
            self.module.post_json = original_post_json
            self.module.fetch_metrics = original_fetch_metrics

    def test_http_failure_diagnostic_does_not_include_response_content(self) -> None:
        secret = "backend-secret-response"
        original_post_json = self.module.post_json
        original_fetch_metrics = self.module.fetch_metrics
        self.module.post_json = lambda *_args: (502, {"error": secret})
        self.module.fetch_metrics = lambda _port: ""
        try:
            with self.assertRaises(self.module.HarnessFailure) as raised:
                self.module.run_case(
                    "http://127.0.0.1:8080",
                    15020,
                    "client-token",
                    "web-search",
                    {"model": "test"},
                )
        finally:
            self.module.post_json = original_post_json
            self.module.fetch_metrics = original_fetch_metrics
        self.assertNotIn(secret, str(raised.exception))

    def test_tool_success_counts_parses_content_free_metrics(self) -> None:
        metrics = "\n".join(
            (
                'agentgateway_tool_runtime_calls_total{backend="e2b",outcome="success",tool="code_interpreter"} 2',
                'agentgateway_tool_runtime_calls_total{tool="web_search",backend="http",outcome="success"} 1',
            )
        )
        self.assertEqual(
            self.module.tool_success_counts(metrics),
            {("code_interpreter", "e2b"): 2, ("web_search", "http"): 1},
        )


if __name__ == "__main__":
    unittest.main()
