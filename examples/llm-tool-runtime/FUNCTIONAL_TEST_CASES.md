# Responses tool runtime functional test cases

The release-binary harness is
[`functional_test.py`](functional_test.py). It launches the requested
`agentgateway` executable with ephemeral loopback ports and a temporary config,
waits for readiness, and sends requests only through public
`/v1/responses`. Its model, Web Search, and Sandbox endpoints are local mocks;
no cloud credential is required.

Run all harness cases or select repeatable cases by name:

```sh
python3 examples/llm-tool-runtime/functional_test.py \
  --binary ./target/release/agentgateway
python3 examples/llm-tool-runtime/functional_test.py \
  --binary ./target/release/agentgateway \
  --case web-search-single \
  --case unmanaged-function-passthrough
```

## Coverage matrix

| Scenario | Release harness | Other automation | Exact expected assertions |
|---|---|---|---|
| Strict listener API key | Startup identity check before every selected case | Config validation | Missing and incorrect bearer keys return HTTP 401 and make zero model requests. The configured client key is accepted, reaches the exact local model endpoint as `mock-model`, and returns the fixed identity response. |
| Web Search single call | `web-search-single` | Live smoke is opt-in | HTTP 200 final round; exactly two model rounds, one Web Search call, zero E2B lifecycle calls; aggregate usage `8/5/13`; complete canonical history; stable `web-only-1` call ID. |
| Same-Response multi-code reuse | `multi-code-reuse` | Direct E2B Rust protocol tests | HTTP 200 final round; exactly two model rounds; two stable call IDs in request order; exactly one Sandbox create/kill pair, two sequential context lifecycles, and no Web Search; canonical outputs remain `set\n` then `isolated\n`; aggregate usage `13/7/20`. OS filesystem/process state remains shared. |
| Web Search and Code overlap | `dual-tool-overlap` | Hermetic Rust acceptance | Round one emits both reserved calls. Web Search and E2B create wait at a bounded two-party barrier and must observe each other before either can complete. Exactly one Web Search call and one E2B Sandbox lifecycle run; canonical output order follows model order rather than completion order. |
| Multi-round usage and final-only response | `dual-tool-overlap` | Hermetic Rust acceptance | Exactly two model rounds; public model is `mock-model`; aggregate usage is `22/10/32`; only `msg_dual_final` is returned. Intermediate message, call IDs, Web Search snippet, code output, and function outputs do not leak into the client body. |
| Python Programmatic Tool Calling replay | `programmatic-server-tools` | Protocol, E2B, budget, and runner Rust tests | The model emits one `{code}` synthetic call. AgentGateway performs two pending replays, executes one Web Search and one nested Code Interpreter call sequentially, performs a completed replay, and returns one normalized synthetic output. Four E2B lifecycles occur without charging Sandbox runs to `maxToolCalls`; only the two nested tools consume call budget. |
| Managed streaming final response | `streaming-tool-runtime` (also live) | `responses_managed_tool_runtime_streams_only_the_final_answer` | HTTP 200 `text/event-stream`; internal model requests use `stream: false`; SSE contains the lifecycle and final text delta but no reserved tool name or intermediate result; `response.completed` carries aggregate usage; Web Search executes once. |
| Live Beijing weather over remote MCP | `remote-mcp-weather` in the opt-in live harness | Remote MCP Rust protocol tests and `weather-mcp-fc` Node tests | HTTP 200 completed response; final output contains `WEATHER_MCP_BEIJING_OK`; the request exposes only `get_forecast` and `get_current_weather` from the authenticated FC MCP URL; local metrics prove a successful `remote_mcp` execution. |
| Live three-day weather through a Python program | `programmatic-mcp-weather` in the opt-in live harness | Programmatic MCP catalog and replay Rust tests | The MCP declaration is programmatic-only and exposes `weather.get_forecast` as the public Python name. The generated Python requests `days=3` and selects the highest daily maximum plus the lowest daily minimum, choosing the earliest date for ties; the final answer contains both extreme days and `PROGRAMMATIC_MCP_WEATHER_3DAY_OK`; local metrics prove one successful `remote_mcp` call. |
| HTTP 400 active-mode rejection | `active-mode-rejections` | Rust unit coverage | Each of background, conversation, managed+unmanaged mix, and untrusted client declaration of `_agentgateway_web_search` returns HTTP 400 with code `managed_tool_request_invalid`; zero requests reach the model or a tool backend. |
| HTTP 502 tool transport/protocol failure | Not in the release harness; hermetic Rust | `responses_managed_tool_runtime_maps_tool_infrastructure_failure_to_502` | Public status 502 with a fixed sanitized infrastructure error; no backend body/URL/credential or model/tool content in the response; no next model round; started work is terminally accounted once. |
| HTTP 502 Sandbox create failure | Not in the release harness; direct E2B Rust | E2B protocol unit tests | A non-201 or malformed create response is sanitized; no execution or kill occurs because no Sandbox ID was accepted; AgentGateway returns HTTP 502 and does not start another model round. |
| HTTP 502 Sandbox context/cleanup failure | Not in the release harness; direct E2B Rust | E2B protocol and lifecycle tests | Context failure aborts later calls and still kills the Sandbox. A failed first kill gets exactly one bounded retry. Definitive cleanup failure discards completed results and returns sanitized HTTP 502. |
| HTTP 504 absolute model-header timeout | Not in the release harness; hermetic Rust | `responses_managed_tool_runtime_absolute_deadline_includes_model_time` | HTTP 504 at the one request deadline; no tool invocation, no later model round, sanitized body, and one terminal timeout accounting path. |
| HTTP 504 absolute model-body timeout | Not in the release harness; hermetic Rust | `responses_managed_tool_runtime_deadline_covers_body_after_headers` | Headers do not reset the deadline. A stalled/periodic body cannot extend it; response is sanitized 504 and no tool starts. |
| HTTP 504 absolute tool timeout | Not in the release harness; hermetic Rust | `responses_managed_tool_runtime_absolute_deadline_includes_tool_time` | Tool time consumes the same absolute request budget; expiry is sanitized 504, later model rounds do not start, sibling work is cancelled, and cleanup/terminal telemetry is bounded once. |
| Unmanaged function single-call passthrough | `unmanaged-function-passthrough` | Rust regression | HTTP 200 contains the upstream `client_owned` function call and stable `client-call-1`; exactly one model request receives the declaration unchanged; zero Web Search or Sandbox requests. |

## Detailed acceptance invariants

### Dual built-ins and canonical history

The first mock model response emits an intermediate assistant message followed
by `_agentgateway_web_search` and `_agentgateway_code_interpreter`. The second
model request must contain the original user input, every round-one output item,
and exactly two `function_call_output` items. Tool output order is the original
model call order even though the faster code backend completes first. Each
output keeps its original `call_id` and contains normalized typed JSON.

The delayed local endpoints prove concurrency with a bounded handshake: each
backend records its arrival, waits until the peer has arrived, and only then
may produce its response. The focused barrier tests also prove a lone serial
arrival fails quickly instead of hanging. The public client response is the
untouched final model Response with usage summed across both rounds and none of
the intermediate history.

### Same-Response code isolation

The release harness proves that AgentGateway groups two ordered code calls into
one direct E2B lifecycle and preserves independent result values and call IDs.
It asserts one create, two sequential create/execute/remove context sequences,
and one final kill. Rust protocol tests independently verify the wire contract.

Variable isolation is not OS isolation: code contexts in one Sandbox share its
filesystem and process state. This is an explicit operational caveat, not an
automated expectation of filesystem isolation.

### Failure hygiene and telemetry

All failure assertions are structural. Diagnostics may name a case, status,
count, or missing environment variable, but must not dump request/response
bodies, query, code, stdout/stderr, URL, bearer token, upstream key, call ID,
Sandbox ID, or credential. Runtime metric labels are restricted to configured
tool names and closed backend/outcome/operation values; spans have no
content-bearing fields.

## Live/manual boundary

The documented release harness, Rust acceptance/unit tests, and Web Search
Function suite are reproducible hermetic checks; this example does not wire
them into repository CI. [`live_functional_test.py`](live_functional_test.py)
is the opt-in end-to-end application for a real LLM model, FC Web Search
Function, E2B-compatible Sandbox, and the deployed FC WeatherAPI MCP server. It
starts a release AgentGateway with a temporary named-gateway config and verifies
successful tool-call metric deltas for `web-search`, `code-interpreter`,
`combined`, `programmatic-server-tools`, `programmatic-mcp-weather`,
`streaming-tool-runtime`, and `remote-mcp-weather`. The streaming
case validates final-only Responses SSE output and a successful Web Search
invocation; the MCP case queries Beijing weather through an allowlisted remote
tool and validates a successful `remote_mcp` metric. The programmatic MCP case
uses the Free-plan three-day forecast window through `tools.call`. When invoked
with `--show-output`, it prints all model-generated Python programs in execution
order before the final forecast. It never modifies the operator's system config.

The two ignored Rust live tests are automated backend smokes only when
`AGENTGATEWAY_LIVE_TOOLS=1` and the Web Search plus three E2B variables are present.
They make one bounded Web Search call and one direct E2B call, use
30 seconds per call, and rely on AgentGateway's cleanup. Missing
configuration prints variable names only and records the smokes as skipped.

Operational properties outside the public API and local metrics remain manual:
Tavily account/quota, FC ingress policy, network isolation, platform
memory/lifetime settings, external metrics export, content-free Function logs,
and confirmation that no residual Sandbox remains after cleanup.
