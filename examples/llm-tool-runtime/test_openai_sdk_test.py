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


class FakeResponses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **request):
        self.request = request
        response = SimpleNamespace(
            status="completed",
            output_text="OPENAI_SDK_TOOLS_OK_391 searched fact",
        )
        return iter(
            (
                SimpleNamespace(type="response.created"),
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta="OPENAI_SDK_TOOLS_OK_",
                ),
                SimpleNamespace(type="response.output_text.delta", delta="391 searched fact"),
                SimpleNamespace(type="response.completed", response=response),
            )
        )


class OpenAISDKTestUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_runs_parallel_builtin_tools_through_typed_stream(self) -> None:
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)

        text = self.module.run_sdk_case(client, "qwen3.6-flash")

        self.assertEqual(text, "OPENAI_SDK_TOOLS_OK_391 searched fact")
        self.assertEqual(responses.request["model"], "qwen3.6-flash")
        self.assertEqual(
            responses.request["tools"],
            [
                {"type": "web_search"},
                {"type": "code_interpreter", "container": {"type": "auto"}},
            ],
        )
        self.assertTrue(responses.request["parallel_tool_calls"])
        self.assertTrue(responses.request["stream"])

    def test_rejects_a_stream_without_a_terminal_response(self) -> None:
        responses = FakeResponses()
        responses.create = lambda **request: iter(
            (
                SimpleNamespace(type="response.created"),
                SimpleNamespace(type="response.output_text.delta", delta="partial"),
            )
        )
        client = SimpleNamespace(responses=responses)

        with self.assertRaisesRegex(
            self.module.SDKTestFailure, "response.completed"
        ):
            self.module.run_sdk_case(client, "qwen3.6-flash")

    def test_gateway_base_url_does_not_reuse_upstream_openai_base_url(self) -> None:
        with patch.dict(
            self.module.os.environ,
            {
                "OPENAI_BASE_URL": "https://upstream.example.test/v1",
                "AGENTGATEWAY_BASE_URL": "http://127.0.0.1:4321/v1",
            },
        ):
            args = self.module.parse_args([])

        self.assertEqual(args.base_url, "http://127.0.0.1:4321/v1")


if __name__ == "__main__":
    unittest.main()
