#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("anthropic_sdk_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("anthropic_sdk_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Anthropic SDK test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMessage:
    def __init__(self, text, stop_reason="end_turn", extra=None):
        self.type = "message"
        self.stop_reason = stop_reason
        self.content = [SimpleNamespace(type="text", text=text)]
        self._extra = extra or {}

    def model_dump(self, **_kwargs):
        value = {
            "type": self.type,
            "stop_reason": self.stop_reason,
            "content": [{"type": "text", "text": self.content[0].text}],
        }
        value.update(self._extra)
        return value


class FakeMessageStream:
    def __init__(self, message, deltas=None):
        self.message = message
        self.text_stream = iter(deltas if deltas is not None else [message.content[0].text])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, message, deltas=None):
        self.message = message
        self.deltas = deltas
        self.create_requests = []
        self.stream_requests = []

    def create(self, **request):
        self.create_requests.append(request)
        return self.message

    def stream(self, **request):
        self.stream_requests.append(request)
        return FakeMessageStream(self.message, self.deltas)


class AnthropicSDKTestUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def payloads(self):
        return self.module.case_payloads("claude-haiku-4-5-20251001")

    def message_for(self, case_name):
        return FakeMessage(self.module.expected_marker(case_name) + " result")

    def test_defines_every_supported_anthropic_case(self) -> None:
        self.assertEqual(
            self.module.CASE_NAMES,
            (
                "web-search",
                "code-interpreter",
                "combined",
                "programmatic-web-search",
                "streaming-tool-runtime",
            ),
        )

    def test_builds_provider_specific_tool_payloads(self) -> None:
        payloads = self.payloads()

        self.assertEqual(set(payloads), set(self.module.CASE_NAMES))
        self.assertEqual(
            payloads["web-search"]["tools"],
            [{"type": "web_search_20250305", "name": "web_search"}],
        )
        self.assertEqual(
            payloads["code-interpreter"]["tools"],
            [{"type": "code_execution_20250825", "name": "code_execution"}],
        )
        programmatic_tools = payloads["programmatic-web-search"]["tools"]
        self.assertEqual(programmatic_tools[0]["type"], "code_execution_20260120")
        self.assertEqual(
            programmatic_tools[1]["allowed_callers"],
            ["code_execution_20260120"],
        )
        self.assertTrue(payloads["streaming-tool-runtime"]["stream"])

    def test_runs_all_cases_through_the_sdk(self) -> None:
        payloads = self.payloads()
        for case_name in self.module.CASE_NAMES:
            with self.subTest(case=case_name):
                message = self.message_for(case_name)
                messages = FakeMessages(message)
                client = SimpleNamespace(messages=messages)

                text = self.module.run_sdk_case(client, case_name, payloads[case_name])

                self.assertEqual(text, message.content[0].text)
                expected_request = dict(payloads[case_name])
                if expected_request.pop("stream"):
                    self.assertEqual(messages.create_requests, [])
                    self.assertEqual(messages.stream_requests, [expected_request])
                else:
                    self.assertEqual(messages.create_requests, [expected_request])
                    self.assertEqual(messages.stream_requests, [])

    def test_streams_multiple_text_blocks_without_inserting_separators(self) -> None:
        payload = self.payloads()["streaming-tool-runtime"]
        message = self.message_for("streaming-tool-runtime")
        message.content = [
            SimpleNamespace(type="text", text="STREAMING_TOOL_"),
            SimpleNamespace(type="text", text="RUNTIME_OK result"),
        ]
        client = SimpleNamespace(
            messages=FakeMessages(
                message,
                deltas=["STREAMING_TOOL_", "RUNTIME_OK result"],
            )
        )

        text = self.module.run_sdk_case(client, "streaming-tool-runtime", payload)

        self.assertEqual(text, "STREAMING_TOOL_RUNTIME_OK result")

    def test_rejects_stream_without_text_deltas(self) -> None:
        payload = self.payloads()["streaming-tool-runtime"]
        message = self.message_for("streaming-tool-runtime")
        client = SimpleNamespace(messages=FakeMessages(message, deltas=[]))

        with self.assertRaisesRegex(self.module.SDKTestFailure, "text delta"):
            self.module.run_sdk_case(client, "streaming-tool-runtime", payload)

    def test_rejects_programmatic_reserved_name_leak(self) -> None:
        payload = self.payloads()["programmatic-web-search"]
        message = FakeMessage(
            "PROGRAMMATIC_WEB_SEARCH_OK result",
            extra={
                "content": [
                    {"type": "server_tool_use", "name": "_agentgateway_web_search"}
                ]
            },
        )
        client = SimpleNamespace(messages=FakeMessages(message))

        with self.assertRaisesRegex(self.module.SDKTestFailure, "reserved internal name"):
            self.module.run_sdk_case(client, "programmatic-web-search", payload)

    def test_allows_reserved_name_in_plain_assistant_text(self) -> None:
        payload = self.payloads()["programmatic-web-search"]
        message = FakeMessage(
            "PROGRAMMATIC_WEB_SEARCH_OK mentions _agentgateway as plain text"
        )
        client = SimpleNamespace(messages=FakeMessages(message))

        text = self.module.run_sdk_case(client, "programmatic-web-search", payload)

        self.assertIn("_agentgateway", text)

    def test_uses_gateway_root_and_bearer_authentication(self) -> None:
        with patch.dict(
            self.module.os.environ,
            {
                "ANTHROPIC_API_KEY": "upstream-secret",
                "ANTHROPIC_BASE_URL": "https://upstream.example.test/v1",
                "AGENTGATEWAY_ANTHROPIC_BASE_URL": "https://provider.example.test/v1",
                "AGENTGATEWAY_API_KEY": "client-secret",
                "AGENTGATEWAY_ANTHROPIC_CLIENT_BASE_URL": "http://127.0.0.1:4321",
            },
            clear=True,
        ):
            args = self.module.parse_args([])
            options = self.module.client_options(args)

        self.assertEqual(args.base_url, "http://127.0.0.1:4321")
        self.assertEqual(
            options,
            {
                "auth_token": "client-secret",
                "base_url": "http://127.0.0.1:4321",
                "timeout": 180.0,
                "max_retries": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
