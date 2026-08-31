#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
OPENAI_CASES = (
    "dual-tool-overlap",
    "programmatic-server-tools",
    "web-search-single",
    "streaming-tool-runtime",
    "multi-code-reuse",
    "active-mode-rejections",
    "unmanaged-function-passthrough",
    "tool-search-deferred",
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load functional test module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FunctionalHarnessSplitTests(unittest.TestCase):
    def test_openai_entrypoint_owns_only_responses_cases(self) -> None:
        module = load_module("openai_functional_test", "functional_test.py")

        self.assertEqual(module.CASE_NAMES, OPENAI_CASES)
        self.assertFalse(hasattr(module, "AnthropicModelHandler"))
        self.assertFalse(hasattr(module, "messages_response"))
        self.assertNotIn("provider: anthropic", module.functional_config(1, 2, 3, 4, 5, 6))

    def test_anthropic_entrypoint_owns_messages_programmatic(self) -> None:
        module = load_module("anthropic_functional_test", "anthropic_functional_test.py")

        self.assertEqual(module.CASE_NAMES, ("messages-programmatic",))
        rendered = module.functional_config(1, 2, 3, 4, 5)
        self.assertIn("provider: anthropic", rendered)
        self.assertIn("baseUrl: http://127.0.0.1:3/v1", rendered)
        self.assertNotIn("provider: openAI", rendered)
        self.assertTrue(hasattr(module, "AnthropicModelHandler"))
        self.assertFalse(hasattr(module, "ModelHandler"))


if __name__ == "__main__":
    unittest.main()
