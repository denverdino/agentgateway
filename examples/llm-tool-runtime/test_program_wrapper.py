#!/usr/bin/env python3
"""Execute the checked-in Programmatic Tool Calling Python wrapper locally."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List


WRAPPER = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "agentgateway"
    / "src"
    / "llm"
    / "tool_runtime"
    / "program_wrapper.py"
)
PREFIX = "__AGENTGATEWAY_PTC_V1__test-nonce:"


def encoded(value: Any) -> str:
    return base64.b64encode(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


def run_wrapper(code: str, replay: List[Dict[str, Any]]) -> Dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "AGENTGATEWAY_PTC_CODE": base64.b64encode(code.encode("utf-8")).decode(
                "ascii"
            ),
            "AGENTGATEWAY_PTC_REPLAY": encoded(replay),
            "AGENTGATEWAY_PTC_NONCE": "test-nonce",
        }
    )
    result = subprocess.run(
        [sys.executable, str(WRAPPER)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    lines = result.stdout.splitlines()
    if len(lines) != 1 or not lines[0].startswith(PREFIX):
        raise AssertionError("wrapper emitted an invalid protocol frame")
    frame = lines[0][len(PREFIX) :]
    frame += "=" * (-len(frame) % 4)
    return json.loads(base64.urlsafe_b64decode(frame))


class ProgramWrapperTests(unittest.TestCase):
    def test_pending_tool_call_is_not_swallowed_by_except_exception(self) -> None:
        outcome = run_wrapper(
            "try:\n"
            "    value = tools.call('web_search', {'query':'AgentGateway'})\n"
            "except Exception:\n"
            "    program_output({'caught':True})\n"
            "program_output(value)\n",
            [],
        )
        self.assertEqual(outcome["kind"], "pending")
        self.assertEqual(outcome["name"], "web_search")
        self.assertEqual(outcome["arguments"], {"query": "AgentGateway"})

    def test_program_output_is_not_swallowed_by_except_exception(self) -> None:
        outcome = run_wrapper(
            "try:\n"
            "    program_output({'value':1})\n"
            "except Exception:\n"
            "    program_output({'value':2})\n",
            [],
        )
        self.assertEqual(outcome, {"version": 1, "kind": "completed", "output": {"value": 1}})

    def test_json_type_change_is_a_contract_error(self) -> None:
        outcome = run_wrapper(
            "value = tools.call('typed', {'value':True})\nprogram_output(value)\n",
            [
                {
                    "sequence": 0,
                    "name": "typed",
                    "arguments": {"value": 1},
                    "output": {"ok": True},
                }
            ],
        )
        self.assertEqual(outcome["kind"], "contract_error")
        self.assertIn("diverged", outcome["message"])

    def test_suppressed_completion_signal_is_a_contract_error(self) -> None:
        outcome = run_wrapper(
            "try:\n    program_output(1)\nexcept BaseException:\n    pass\n",
            [],
        )
        self.assertEqual(outcome["kind"], "contract_error")
        self.assertIn("suppressed", outcome["message"])

    def test_non_json_output_is_model_visible_error(self) -> None:
        outcome = run_wrapper("program_output(float('nan'))\n", [])
        self.assertEqual(outcome["kind"], "error")
        self.assertEqual(outcome["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
