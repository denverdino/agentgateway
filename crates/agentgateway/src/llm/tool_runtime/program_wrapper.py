import base64 as _ag_b64
import json as _ag_json
import os as _ag_os
import sys as _ag_sys

_ag_code = _ag_b64.b64decode(_ag_os.environ["AGENTGATEWAY_PTC_CODE"]).decode("utf-8")
_ag_replay = _ag_json.loads(
    _ag_b64.b64decode(_ag_os.environ["AGENTGATEWAY_PTC_REPLAY"]).decode("utf-8")
)
_ag_nonce = _ag_os.environ["AGENTGATEWAY_PTC_NONCE"]


def _ag_canonical(value):
    return _ag_json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class _PendingToolCall(BaseException):
    def __init__(self, sequence, name, arguments):
        self.sequence = sequence
        self.name = name
        self.arguments = arguments


class _ProgramCompleted(BaseException):
    def __init__(self, value):
        self.value = value


class _ProgramContractError(BaseException):
    pass


class _NullWriter:
    def write(self, value):
        return len(value)

    def flush(self):
        return None


class _Tools:
    def __init__(self):
        self.sequence = 0
        self.completed = False

    def call(self, name, arguments):
        if self.completed:
            raise _ProgramContractError("tool call occurred after program_output")
        if not isinstance(name, str) or not name:
            raise TypeError("tools.call name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise TypeError("tools.call arguments must be an object")
        sequence = self.sequence
        self.sequence += 1
        if sequence < len(_ag_replay):
            entry = _ag_replay[sequence]
            try:
                arguments_match = _ag_canonical(entry.get("arguments")) == _ag_canonical(
                    arguments
                )
            except BaseException as error:
                raise _ProgramContractError(
                    "program replay arguments are not valid JSON"
                ) from error
            if (
                entry.get("sequence") != sequence
                or entry.get("name") != name
                or not arguments_match
            ):
                raise _ProgramContractError(
                    "program replay diverged from the authorized transcript"
                )
            return entry.get("output")
        raise _PendingToolCall(sequence, name, arguments)


tools = _Tools()


def program_output(value):
    if tools.completed:
        raise _ProgramContractError("program_output was called more than once")
    if tools.sequence != len(_ag_replay):
        raise _ProgramContractError(
            "program completed before consuming the authorized transcript"
        )
    _ag_canonical(value)
    tools.completed = True
    raise _ProgramCompleted(value)


def _ag_bound(value):
    encoded = str(value).encode("utf-8")[:1024]
    return encoded.decode("utf-8", errors="ignore")


_ag_stdout, _ag_stderr = _ag_sys.stdout, _ag_sys.stderr
try:
    _ag_sys.stdout = _NullWriter()
    _ag_sys.stderr = _NullWriter()
    try:
        exec(
            compile(_ag_code, "<agentgateway-program>", "exec"),
            {"tools": tools, "program_output": program_output},
        )
        if tools.completed:
            _ag_outcome = {
                "version": 1,
                "kind": "contract_error",
                "message": "program suppressed the program_output completion signal",
            }
        else:
            _ag_outcome = {
                "version": 1,
                "kind": "error",
                "error_type": "missing_program_output",
                "message": "program returned without calling program_output",
            }
    except _PendingToolCall as pending:
        _ag_outcome = {
            "version": 1,
            "kind": "pending",
            "sequence": pending.sequence,
            "name": pending.name,
            "arguments": pending.arguments,
        }
    except _ProgramCompleted as completed:
        _ag_outcome = {"version": 1, "kind": "completed", "output": completed.value}
    except _ProgramContractError as error:
        _ag_outcome = {
            "version": 1,
            "kind": "contract_error",
            "message": _ag_bound(error),
        }
    except BaseException as error:
        _ag_outcome = {
            "version": 1,
            "kind": "error",
            "error_type": _ag_bound(type(error).__name__),
            "message": _ag_bound(error),
        }
finally:
    _ag_sys.stdout, _ag_sys.stderr = _ag_stdout, _ag_stderr

try:
    _ag_payload = _ag_json.dumps(
        _ag_outcome,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
except BaseException as error:
    _ag_payload = _ag_json.dumps(
        {
            "version": 1,
            "kind": "error",
            "error_type": "serialization_error",
            "message": _ag_bound(error),
        },
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
_ag_frame = _ag_b64.urlsafe_b64encode(_ag_payload).decode("ascii").rstrip("=")
print("__AGENTGATEWAY_PTC_V1__" + _ag_nonce + ":" + _ag_frame)
