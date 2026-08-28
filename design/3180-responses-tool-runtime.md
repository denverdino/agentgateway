# EP-3180: Managed Tool Runtime for OpenAI-Compatible Responses

- Issue: [#3180](https://github.com/agentgateway/agentgateway/issues/3180)
- Related: N/A
- Status: implemented
- Date: 8/24/2026
- Implementation review: 8/26/2026
- Programmatic Tool Calling extension: approved design, implementation pending
- Programmatic Tool Calling design review: 8/27/2026
- Tool Search extension: implemented
- Tool Search design review: 8/27/2026

> **Note:** The original managed Tool Runtime was reconciled with the current implementation on the implementation-review
> date above. The Programmatic Tool Calling sections describe the approved follow-up implementation.

## Summary

AgentGateway's OpenAI-compatible `/v1/responses` endpoint previously proxied one model request and returned one model
response. It could forward tool declarations and tool calls, but it did not execute model-generated calls.

This design adds a Tool Runtime around the existing Responses request path. When a request declares a configured
managed function, `web_search`, `code_interpreter`, or a request-scoped remote MCP server, AgentGateway can:

1. map compatible built-in tools to reserved function definitions;
2. call the selected model and inspect its translated Responses output;
3. authorize and execute model-generated calls through configured backends;
4. append the complete intermediate model output and corresponding `function_call_output` items to the canonical input;
5. continue inference until the model produces no more managed calls; and
6. return only the final model response, with usage aggregated across all model rounds.

Ordinary functions and Web Search use operator-configured HTTP functions, including Alibaba Cloud Function Compute HTTP
triggers. The example Web Search function calls Tavily with a function-side `TAVILY_API_KEY`. Code Interpreter uses the
E2B-compatible control-plane and code-execution protocols directly from AgentGateway. Multiple code calls in one model
Response share one short-lived Sandbox but use separate Python contexts.

Remote MCP declarations are discovered and executed through AgentGateway's existing outbound MCP stack. Tool Runtime
adds only a request-scoped orchestration facade; `PolicyClient`, `McpHttpClient`, Streamable HTTP parsing, MCP session
IDs, response bounds, outbound policies, and telemetry remain owned by the existing MCP implementation. All calls to
one declared server in a Response reuse the same MCP session.

The example application also includes an Alibaba Cloud Function Compute Custom Runtime Web Function derived from the
public `weatherapicom/weatherapi-mcp` tool contract. It exposes only current-weather and forecast operations over
stateless Streamable HTTP, keeps `WEATHERAPI_KEY` inside the Function, and protects `/mcp` with a separate application
bearer token. Stateless transport lets FC distribute successive protocol requests across instances without affinity.

The implementation is request-local and configured through static local YAML. It accepts `stream: true`, buffers the
internal model/tool loop, and emits only the final canonical Response as OpenAI-compatible SSE lifecycle events.

The follow-up Programmatic Tool Calling implementation accepts OpenAI-compatible `allowed_callers` and
`programmatic_tool_calling` declarations even when the selected upstream model does not implement the native hosted
runtime. AgentGateway asks that model for a Python program, executes it through the existing E2B Python `runCode`
protocol, and resolves program-owned calls to Web Search, Code Interpreter, and discovered Remote
MCP tools through the same authorized backends used for direct calls. This Python runtime is an AgentGateway extension:
OpenAI's native Programmatic Tool Calling runtime executes JavaScript in V8.

## Background

AgentGateway already had the protocol machinery needed for a model/tool loop:

- `process_responses_request` parses Responses requests, applies LLM request policies, selects the upstream provider
  format, and serializes the first upstream request.
- Provider conversion code translates function declarations, `function_call` output items, and
  `function_call_output` input items across supported provider formats.
- `httpproxy` owns target selection, transport, TLS, authentication, policy ordering, and the upstream call.
- `process_response` translates the final provider response, applies response policies, records usage, and serializes
  the client response.

The missing layer was orchestration between the first model call and final response processing. Putting execution in
the gateway also requires an explicit trust boundary: a model may choose only a declared tool and its arguments; it must
not choose a destination, credential, Sandbox template, execution timeout, or network policy.

The implementation therefore compiles a static tool registry at configuration load time and activates the loop only
for requests that select a managed tool. The existing single-call proxy path remains intact for all other traffic.

## Goals

- Execute configured model-generated function calls for OpenAI-compatible Responses requests.
- Present OpenAI-style `web_search` and `code_interpreter` declarations to clients while using portable function calls
  in model-facing requests.
- Keep tool destinations, credentials, backend selection, and hard limits under operator control.
- Reuse AgentGateway's existing provider conversion, model routing, authentication, policy, and transport layers for
  every model round.
- Reuse AgentGateway's existing outbound MCP client and proxy policy chain for remote MCP discovery and execution.
- Accept OpenAI-compatible Programmatic Tool Calling declarations and execute model-generated Python in an E2B
  Sandbox for upstream models without native Programmatic Tool Calling support.
- Let generated Python invoke request-authorized Web Search, Code Interpreter, and Remote MCP tools
  through one stable `tools.call(name, arguments)` interface.
- Reuse the existing E2B Python context, execution, result parsing, deadline, and cleanup implementation rather than
  adding a JavaScript runtime or a second Sandbox protocol.
- Execute independent tool operations concurrently with a server-side bound while preserving model output order.
- Reuse one E2B Sandbox for all Code Interpreter calls in a single model Response, while using a new Python context for
  each call and terminating the Sandbox after the batch.
- Enforce one absolute request deadline and bounded argument, output, round, call, parallelism, and Sandbox limits.
- Aggregate token usage across internal model rounds and run final response policies once.
- Expose bounded, content-free metrics for runtime, tool, and Sandbox outcomes.
- Preserve the previous request path when the managed runtime is not active.

## Non-Goals

- Streaming managed execution or exposing intermediate tool events to clients.
- OpenAI background mode or conversation-managed state while the runtime is active.
- Exact native `web_search_call` or `code_interpreter_call` output-item compatibility.
- Persistent Sandboxes, variables, or files across model Responses or client requests.
- Operating-system isolation between multiple Python calls that share one Sandbox.
- Client-selected endpoints or credentials for FC, E2B, or ordinary managed functions; client-provided remote MCP
  endpoints and OAuth tokens are the explicit exception and are subject to a strict request-scoped contract.
- Code Interpreter artifacts, input files, package installation, Shell, Node.js, or languages other than Python.
- Automatically retrying model-generated tool execution.
- A new Kubernetes API, controller translation, or xDS resource in the initial implementation.
- A generic web crawler in AgentGateway; the provided Function Compute implementation is a bounded Tavily adapter.
- OpenAI-managed `connector_id` MCP connectors, interactive MCP approval/resume flows, server-initiated sampling or
  elicitation, and persistent MCP sessions across Responses requests.
- Legacy MCP HTTP/SSE transport in the initial remote MCP implementation; request-scoped MCP uses Streamable HTTP.
- Native OpenAI JavaScript/V8 runtime equivalence. AgentGateway-managed programs are Python and intermediate `program`
  and `program_output` items remain internal to the managed loop.
- Parallel tool dispatch from inside one Python program in the initial Programmatic Tool Calling implementation.
- Disabling or filtering network access from the Programmatic Tool Calling Sandbox. The E2B default network policy is
  retained.
- Programmatic invocation of ordinary managed `function` or `custom` tools in the initial extension. Their existing
  direct execution path is unchanged.
- Forwarding a client `tool_search` declaration to the upstream provider, or relying on any provider-native tool-search
  implementation. AgentGateway executes the search itself so the feature works against every fronted model.
- Embedding-based, model-based, or remotely served tool matching. Tool Search ranks candidates with deterministic local
  lexical scoring so a search costs no extra network hop and no additional dependency.
- Deferring `web_search` or `code_interpreter`. Each contributes exactly one declaration, so withholding it saves
  nothing.
- Exposing native `tool_search_call` and `tool_search_output` items to clients. Search calls and their results remain
  internal to the managed loop, like `program` and `program_output`.
- Unloading an injected tool definition later in the same request, or preserving the cached prefix across the injection
  itself. Definitions are appended after the existing declarations and stay for the remainder of the request.

## API

### Client request

Clients continue to use the OpenAI Responses API shape. A request can declare either or both supported built-ins:

```json
{
  "model": "smart",
  "input": "Search today's market data and calculate the average.",
  "tools": [
    { "type": "web_search" },
    { "type": "code_interpreter", "container": { "type": "auto" } }
  ],
  "parallel_tool_calls": true,
  "stream": false
}
```

An ordinary managed function retains its client-provided name, description, JSON Schema, and `strict` setting. Its name
must also exist in the operator registry.

A remote MCP server uses the OpenAI-compatible declaration below:

```json
{
  "type": "mcp",
  "server_label": "calendar",
  "server_description": "Calendar tools for the current user",
  "server_url": "https://mcp.example.com/mcp",
  "authorization": "oauth-access-token",
  "allowed_tools": ["list_events", "create_event"],
  "require_approval": "never"
}
```

`server_label` must be unique within the request. The initial implementation supports URL-based remote MCP servers,
not `connector_id`. `require_approval` accepts `auto` or `never`; both execute inside AgentGateway without an approval
round trip. Values such as `always` and per-tool policies are rejected because Tool Runtime has no client
approval/resume state. `authorization` is kept in redacted secret storage, is sent only to the selected MCP server as
a bearer token, and is removed from the final response's echoed `tools` field.

Programmatic Tool Calling uses the OpenAI-compatible request declarations below. `allowed_callers` is accepted on
supported built-ins and Remote MCP declarations. Omitted `allowed_callers` is equivalent to
`["direct"]`; `["programmatic"]` hides that tool from direct model calls; and `["direct", "programmatic"]` enables
both routes.

```json
{
  "model": "smart",
  "input": "Use one program to search for AgentGateway and calculate 29 * 31.",
  "tools": [
    { "type": "web_search", "allowed_callers": ["programmatic"] },
    {
      "type": "code_interpreter",
      "container": { "type": "auto" },
      "allowed_callers": ["programmatic"]
    },
    { "type": "programmatic_tool_calling" }
  ],
  "stream": false
}
```

Programmatic execution requires an operator-configured `codeInterpreter` E2B backend even when the client program
declares only Web Search or MCP tools. The E2B backend supplies the Python runtime; declaring `code_interpreter` in the
client request separately determines whether the generated program may invoke Code Interpreter as a nested managed
tool.

A request may declare `programmatic_tool_calling` at most once and must expose at least one eligible programmatic tool.
One upstream response may contain either one synthetic program call or direct managed calls, but not both. These rules
keep authorization, replay history, and synthetic `function_call_output` association unambiguous.

### Local configuration

The Tool Runtime is configured under `llm.toolRuntime`:

```yaml
llm:
  toolRuntime:
    maxRounds: 8
    maxToolCalls: 16
    maxParallelToolCalls: 4
    totalTimeout: 120s
    maxArgumentsBytes: 65536
    maxOutputBytes: 1048576
    tools:
    - name: web_search
      builtin: webSearch
      backend:
        type: http
        url: $FC_WEB_SEARCH_URL
        timeout: 15s
        bearerToken: $FC_WEB_SEARCH_TOKEN
    - name: code_interpreter
      builtin: codeInterpreter
      backend:
        type: e2b
        apiKey: $E2B_API_KEY
        apiUrl: $E2B_API_URL
        domain: $E2B_DOMAIN
        timeout: 120s
    - name: get_weather
      backend:
        type: http
        url: $FC_WEATHER_URL
        timeout: 5s
        bearerToken: $FC_WEATHER_TOKEN
```

The limit fields use these defaults when omitted:

| Field | Default | Meaning |
| --- | ---: | --- |
| `maxRounds` | `8` | Maximum model rounds, including the final no-call round |
| `maxToolCalls` | `16` | Maximum tool calls across the request |
| `maxParallelToolCalls` | `4` | Maximum concurrent backend operations |
| `totalTimeout` | `120s` | Absolute deadline for all model and tool work |
| `maxArgumentsBytes` | `65536` | Maximum serialized arguments for one call |
| `maxOutputBytes` | `1048576` | Runtime and backend output bound; minimum `128` with code interpreter |

Every limit must be greater than zero, and `maxParallelToolCalls` cannot exceed `maxToolCalls`. At least one tool is
required. Names and built-in kinds must be unique, and configured names cannot start with `_agentgateway_`.

Backend compatibility is fixed:

- `webSearch` requires `http`.
- `codeInterpreter` requires `e2b`.
- ordinary functions require `http`.

`http` is a provider-neutral JSON-over-HTTPS backend and can target Function Compute, Lambda, a container service, or
any service implementing the documented contract. The legacy discriminator `functionComputeHttp` remains accepted as
an input alias and is normalized to `http`; new configuration and serialized output use `http`.

Backend URLs must use HTTPS, except loopback HTTP used by tests. An E2B `apiUrl` must be an origin without a path or
query. Its `domain` must be a valid hostname and is the data-plane fallback when the Sandbox create response does not
supply a domain. The E2B control plane is trusted to select the data-plane hostname by returning an optional valid
`domain`; when present, that value takes precedence over the configured fallback. The E2B timeout must not exceed 24
hours. Secrets use AgentGateway's redacted secret types and may be supplied through environment expansion or
file-backed secret configuration.

### Built-in mapping

`web_search` is rewritten before the model request to a strict function named `_agentgateway_web_search` with one
required string argument, `query`. Supported client options are validated and retained in request-scoped trusted state:

- `filters.allowed_domains`;
- `search_context_size` (`low`, `medium`, or `high`); and
- approximate `user_location` fields.

AgentGateway preserves these values in the HTTP backend contract, so the client, gateway, and function consistently
use `low`, `medium`, and `high` without an internal alias mapping.

The model can generate only `query`. AgentGateway merges the trusted options after argument validation and sends a flat
request to the Web Search function:

```json
{
  "query": "today's market data",
  "allowed_domains": ["example.com"],
  "search_context_size": "medium",
  "user_location": { "type": "approximate", "country": "CN" }
}
```

The function returns a bounded object containing `results` and an optional `truncated` flag. The checked-in Function
Compute example invokes the fixed Tavily endpoint and obtains its credential only from the function environment.

`code_interpreter` is rewritten to `_agentgateway_code_interpreter`, a strict function with one required string
argument, `code`. Only `container: { type: "auto" }` is accepted. Explicit container identifiers and extra options such
as files or language selection are rejected.

The `_agentgateway_` prefix is reserved. Clients cannot declare a function in that namespace, and model calls are
executable only when their rewritten name is both declared in the canonical request and present in the compiled
registry.

### Remote MCP mapping

Before the first model round, AgentGateway initializes each declared MCP server and retrieves `tools/list`, including
pagination. `allowed_tools` filters discovery and is fail-closed: every allowed name must be advertised. Each accepted
MCP tool is compiled into a strict request-local function named `_agentgateway_mcp_{serverIndex}_{toolIndex}`. The
model never receives the MCP URL or authorization value. A forced MCP `tool_choice` is translated to the corresponding
internal function choice after discovery. That forced choice applies only to the first model round; after any managed
tool outputs are appended, Tool Runtime resets `tool_choice` to `auto` so the model can produce a final answer instead
of being forced into an unbounded sequence of tool calls.

When `allowed_callers` includes `programmatic`, discovery also adds the tool to a request-local programmatic catalog
under the public name `{server_label}.{tool_name}`. A programmatic-only MCP tool is omitted from the model's direct
function declarations. A dual-mode MCP tool appears both as its reserved internal function and in the programmatic
catalog. After all MCP servers finish discovery, AgentGateway regenerates the synthetic program function description so
the model sees the final tool names, descriptions, input schemas, generic MCP result shape, and failure behavior before
it writes Python. The aggregate serialized programmatic catalog is capped at 2 MiB across all discovered servers. A
forced MCP `tool_choice` is rejected when the selected tool is programmatic-only because that
choice requests a direct MCP call rather than a program.

When the model selects an imported function, Tool Runtime validates its arguments against the server-provided input
schema and invokes `tools/call` with the original trusted MCP tool name. The MCP result is serialized as the matching
`function_call_output`, and the normal model/tool loop continues. Multiple calls to the same server share one
request-scoped initialized session and participate in the existing `parallel_tool_calls` scheduler.

The checked-in WeatherAPI example is deployed as an FC Web Function rather than an event-function HTTP adapter so MCP
status codes, headers, and JSON response semantics pass through unchanged. It listens on `CAPort`, uses the MCP SDK's
stateless JSON response mode, and accepts only `POST /mcp`. The FC trigger uses anonymous platform authentication while
the application validates `INGRESS_BEARER_TOKEN` before parsing JSON. Its deployment utility creates or updates the
Function and trigger through the FC 20230330 SDK and writes only the resulting URL and ingress token to the ignored
local `.env`.

### AgentGateway-managed Programmatic Tool Calling

OpenAI's native Programmatic Tool Calling runtime executes generated JavaScript in V8. AgentGateway instead implements
an intentionally provider-independent compatibility route for upstream models without native support: the model calls
one reserved function with generated Python, AgentGateway executes that Python through E2B, and only the final managed
response is returned to the client. Native `program`, nested caller, fingerprint, and `program_output` items do not
cross the client boundary in this mode.

The initial AgentGateway extension supports `web_search`, `code_interpreter`, and discovered Remote MCP tools. Native
OpenAI PTC documents MCP and Code Interpreter as programmatic-capable; AgentGateway additionally enables its managed
Web Search mapping. Ordinary managed functions and custom tools remain direct-only in this phase.

After built-in mapping and Remote MCP discovery, AgentGateway adds this strict model-facing function:

```json
{
  "type": "function",
  "name": "_agentgateway_programmatic_tool_calling",
  "description": "Run one Python program in E2B. Use tools.call(name, arguments) for declared tools and program_output(value) exactly once for the final result.",
  "strict": true,
  "parameters": {
    "type": "object",
    "properties": {
      "code": {
        "type": "string",
        "description": "Python source code"
      }
    },
    "required": ["code"],
    "additionalProperties": false
  }
}
```

The generated program receives one stable API:

```python
value = tools.call("tool-name", {"argument": "value"})
program_output(value)
```

Built-ins use `web_search` and `code_interpreter`, and Remote MCP tools use `{server_label}.{tool_name}`. One
string-based API avoids Python-identifier restrictions, provides an explicit namespace for MCP name collisions, and
keeps generated wrapper code independent of the number and kinds of discovered tools. Every call is resolved through
the request-local catalog and existing registry, caller authorization, compiled argument schema, trusted options,
budget, backend, and telemetry path. No destination or credential is included in the Python source or replay
transcript.

#### Record/replay protocol

The program executor uses sequential record/replay because an E2B `runCode` execution cannot synchronously call back
into the in-flight AgentGateway process:

1. AgentGateway wraps the generated source with `tools.call`, `program_output`, a bounded replay transcript, and a
   per-execution random protocol nonce.
2. The existing E2B Python context and `/execute` path runs the wrapper in a fresh short-lived Sandbox.
3. `tools.call` assigns a zero-based sequence number. If the transcript contains the same sequence, public tool name,
   and canonical JSON arguments, it returns the recorded JSON output as a Python value. Internal pending/completed
   signals inherit from `BaseException`, so ordinary generated `except Exception` handlers cannot swallow them.
4. The first call without a replay entry raises an internal `PendingToolCall`. The wrapper emits one nonce-bound,
   size-bounded protocol result and exits normally.
5. AgentGateway parses the pending call, resolves and validates it, executes it through the existing backend, appends
   its normalized result to the transcript, and runs the program again from the beginning in a new Sandbox.
6. `program_output(value)` emits the completed program result. The executor requires exactly one final output and no
   unresolved call.

```rust
struct ProgramReplayEntry {
    sequence: usize,
    public_name: Strng,
    arguments: Value,
    output: Value,
}
```

Replay mismatch is a deterministic-program contract violation and terminates the managed request; it is not returned
to the model as a retryable program error after earlier tool side effects. Each unresolved call consumes the normal
`maxToolCalls` budget. Each replay consumes the same absolute `totalTimeout`; one successful program
uses one Sandbox per unresolved call plus one completion pass. The request-wide upper bound is the sum of
`maxToolCalls` and `maxRounds` Sandbox executions. Python source, wrapper input, protocol result, tool arguments, tool
outputs, stdout, and stderr remain subject to the existing source and output bounds. The initial implementation runs
program-owned calls sequentially and does not add `call_many` or program-internal concurrency.

Python syntax/runtime failures and missing `program_output` are returned to the upstream model as one structured
synthetic-function error so it can generate a corrected program within the remaining model-round budget. E2B creation,
transport, deadline, or cleanup failures remain request-terminating infrastructure errors. Tool application errors are
recorded as ordinary replay values so the Python program can inspect them; tool infrastructure failures terminate the
request. Nested results are truncated again when necessary to fit the remaining aggregate replay budget. Before each
nested call, the runner reserves enough space for a bounded truncation result; if no such space remains, it returns a
`program_replay_limit` application result to the model without executing another tool or turning earlier side effects
into a client-visible request failure.

The program Sandbox uses E2B's default network behavior. AgentGateway does not configure egress rules, patch Python
network libraries, or claim equivalence with OpenAI's no-network V8 runtime. AgentGateway still withholds all backend
URLs, authorization values, E2B credentials, and MCP session data. Because generated Python can access the network and
receives replayed tool results, operators must treat this mode as capable of transmitting user or tool data and must
not treat it as a Zero Data Retention equivalent.

#### Weather Remote MCP example

The checked-in WeatherAPI Remote MCP server provides `get_forecast`. The live example uses only that discovered tool
from a program and requests three days because WeatherAPI's current Free plan includes a three-day forecast according
to its [pricing matrix](https://www.weatherapi.com/pricing.aspx):

```json
{
  "model": "smart",
  "input": "Use Programmatic Tool Calling to return Beijing's daily high and low temperatures for the next three forecast days, plus the highest and lowest temperature across the period. Include PROGRAMMATIC_MCP_WEATHER_3DAY_OK in the final answer.",
  "tools": [
    {
      "type": "mcp",
      "server_label": "weather",
      "server_url": "$FC_WEATHER_MCP_URL",
      "authorization": "$FC_WEATHER_MCP_TOKEN",
      "allowed_tools": ["get_forecast"],
      "require_approval": "auto",
      "allowed_callers": ["programmatic"]
    },
    { "type": "programmatic_tool_calling" }
  ],
  "stream": false
}
```

A valid model-generated program is:

```python
import json

response = tools.call("weather.get_forecast", {
    "q": "Beijing, China",
    "days": 3,
    "alerts": "no",
    "aqi": "no",
})

text_item = next(item for item in response["content"] if item.get("type") == "text")
weather = json.loads(text_item["text"])
forecast_days = weather["forecast"]["forecastday"]
if len(forecast_days) != 3:
    raise ValueError("weather server did not return exactly three forecast days")

daily = [
    {
        "date": item["date"],
        "max_c": item["day"]["maxtemp_c"],
        "min_c": item["day"]["mintemp_c"],
    }
    for item in forecast_days
]

program_output({
    "location": weather["location"]["name"],
    "daily": daily,
    "period_high_c": max(item["max_c"] for item in daily),
    "period_low_c": min(item["min_c"] for item in daily),
})
```

The final program value contains exactly three daily records plus the period extrema. The following model round converts
that reduced JSON value into the client-visible answer and preserves the exact test marker.

### AgentGateway-executed Tool Search

A large tool catalog is expensive before the model has chosen anything: one Remote MCP server alone can contribute 128
declarations, and every one of them occupies the cached prompt prefix on every round. OpenAI's hosted Tool Search
addresses this by letting a client declare `{"type": "tool_search"}` and mark individual tools `"defer_loading": true`;
deferred definitions are withheld until the model searches for them.

AgentGateway executes that search itself instead of forwarding the declaration upstream, so the behavior is identical
across every fronted model and provider rather than being limited to providers that implement it natively. The mechanism
reuses the existing managed loop: the search is one more reserved function, and loading a tool is one more mutation of
the canonical request between rounds.

A client declares Tool Search alongside its tools:

```json
{
  "tools": [
    {"type": "tool_search"},
    {"type": "mcp", "server_label": "dice", "server_url": "https://example.test/mcp", "defer_loading": true},
    {"type": "function", "name": "lookup", "parameters": {"type": "object"}, "defer_loading": true},
    {
      "type": "namespace",
      "name": "weather",
      "description": "Weather utilities",
      "defer_loading": true,
      "tools": [{"type": "function", "name": "forecast", "parameters": {"type": "object"}}]
    }
  ]
}
```

`defer_loading` is accepted on managed `function` declarations, on `mcp` servers, and on the `namespace` type, where it
is inherited by every member. Deferrable tools are exactly the ones that can contribute many declarations. `web_search`
and `code_interpreter` contribute one each and are not deferrable.

A `namespace` is flattened rather than forwarded: `collect_model_calls` rejects any model call carrying a `namespace`
field, so a namespace declaration must never reach the model. Its members keep their original names as both public and
internal name, and the namespace name and description are composed onto each member description exactly as a Remote MCP
server label and description are composed onto its tools. The namespace name also becomes the member's search label.

A deferred tool is withheld from `tools` and recorded in a request-local deferred catalog. In its place, AgentGateway
declares one strict reserved function, `_agentgateway_tool_search`, taking a single required `query` string. Its
description carries a bounded `label (names)` index of the deferred catalog grouped by MCP server label or namespace
name, so the model can tell what is searchable without paying for full schemas. The index is regenerated after every
Remote MCP server finishes discovery, because a deferred server's tool count is unknown until then.

Matching is deterministic local lexical scoring. The query and each candidate's text are case-folded and split on
whitespace, `_` and `.`. Each matched token scores 8 against the public name, 4 against the label, and 2 against the
description; a prefix-only match scores half, and tokens shorter than three bytes are not eligible to prefix-match at
all, since they would match most of a catalog and displace better candidates from the bounded result set. The tier
weights are even so that halving stays meaningful at every tier. Candidates are sorted by score descending and tie-broken
on internal name so results are stable, zero-scoring candidates stay deferred, and the top `MAX_TOOL_SEARCH_RESULTS` are
returned. This costs no network hop and adds no dependency, which is why it is preferred to embedding or model-based
matching.

A group that would exceed the index budget is truncated within itself and marked `+N more` rather than dropped whole. One
MCP server contributes one label and therefore one group, so dropping the group would leave the model with no searchable
listing at all and silently disable the feature for exactly the large catalogs it exists to serve.

A search is executed in process. There is no Tool backend and no new execution result variant: the winning declarations
are appended to the end of `tools` and the model sees an ordinary `function_call_output` containing
`{"tools": [...], "ok": true}`. Appending rather than inserting keeps the previously cached declarations byte-identical.
Injection is idempotent — a repeated search still reports the definitions but never declares the same tool name twice,
which would otherwise be an upstream 400.

Search internals stay private. No `tool_search_call` or `tool_search_output` item is synthesized, and because only the
terminating round's output is returned to the client, no `_agentgateway_tool_search` call or result crosses the client
boundary. The client's original `tools` array, including `tool_search`, `namespace` and `defer_loading`, is echoed back
verbatim in the final response.

Bounds are request-local constants, sized against the existing Remote MCP limits:

| Bound | Value | Rationale |
| --- | --- | --- |
| `MAX_DEFERRED_TOOLS` | `4 × MAX_DISCOVERED_TOOLS` | The deferred catalog is the point of the feature, so it is allowed to exceed one server's discovery limit. |
| `MAX_TOOL_SEARCH_RESULTS` | 8 | One search reports a reviewable set rather than a second catalog. |
| `MAX_INJECTED_TOOLS` | `MAX_TOOL_SEARCH_RESULTS × MAX_TOOL_SEARCH_CALLS - 1` | Caps total context growth across all searches in one request. Set one below what the permitted searches can inject, so the cap binds rather than being unreachable behind the other two limits. |
| `MAX_TOOL_SEARCH_CALLS` | 8 | Matches the default round limit. |
| `MAX_TOOL_SEARCH_INDEX_BYTES` | 32 KiB | The index lives in the cached prefix, so it is bounded far below the 2 MiB programmatic catalog. |

Neither the search-call nor the injected-tool limit fails the request, so the model can still finish with the tools it
already has. Exceeding the search-call limit returns an application error, because nothing was injected. Reaching the
injected-tool limit is instead a partial success: the declarations appended before the cap are live for the next round,
so the search reports them together with `truncated: true`. Search calls are counted against `max_tool_calls` even
though they bypass the Tool backend batch.

The following declarations are rejected as `invalid_request`:

- `defer_loading: true` with no `tool_search` declaration, which would make the tool unreachable.
- `tool_search` declared but nothing deferred, checked after Remote MCP discovery.
- A duplicate or optioned `tool_search` declaration, matching `web_search` handling.
- `defer_loading` combined with programmatic `allowed_callers`, because the programmatic catalog is embedded in the
  initial-context description and gating it later would invalidate the cached prefix anyway.
- `tool_choice` forcing a deferred function by name, which would omit that declaration from the very round that requires
  it.
- `defer_loading` on any other tool type, including passthrough types the gateway forwards untouched. The provider would
  receive a `defer_loading` the gateway consumed the `tool_search` declaration for, and the gateway holds no catalog
  entry to search the tool back into context.

A `tool_choice` of type `mcp` naming a tool on a deferred server is not rejected. That one declaration is injected
eagerly instead, consistent with the existing forced-MCP translation path.

## Runtime Design

### Components

- `config.rs` defines the deserialized local configuration, defaults, runtime limits, and backend-specific settings.
- `validation.rs` enforces startup-time configuration invariants before the registry is made available to requests.
- `mapper.rs` validates tool declarations, rewrites built-ins, retains trusted Web Search options, compiles per-request
  function schemas, and decides whether the runtime is active.
- `registry.rs` compiles immutable operator configuration and resolves public and internal tool names.
- `schema.rs` compiles and validates the supported JSON Schema subset before any backend invocation.
- `runner.rs` owns the bounded model/tool loop, round history, usage aggregation, and terminal runtime summary.
- `program.rs` owns Python wrapper construction, nonce-bound protocol parsing, replay transcript validation, and the
  sequential Programmatic Tool Calling state machine.
- `mod.rs` owns request budgets, call authorization, backend grouping, concurrency, stable output ordering,
  `function_call_output` construction, and final managed-response/SSE reconstruction.
- `backend.rs` defines normalized calls, results, application errors, infrastructure errors, and the `ToolBackend`
  interface.
- `http_backend.rs` executes ordinary functions and Web Search over authenticated HTTP.
- `e2b.rs` owns the complete direct E2B Sandbox lifecycle and Python protocol normalization.
- `remote_mcp/mod.rs` adapts request-local MCP tools to the common `ToolBackend` interface.
- `remote_mcp/client.rs` is the request-scoped orchestration facade over the existing MCP outbound client and protocol
  stack.
- `transport.rs` contains shared transport-status classification and UTF-8-safe output truncation helpers.
- `telemetry.rs` exposes closed-label metrics and spans without user content.
- `proxy/httpproxy.rs` implements the pinned model round trip, rebuilding and authenticating each round against the
  initially selected provider/backend, invokes `runner.rs`, and sends the final response through the existing response
  policy and serialization pipeline.

### Request flow

```text
client POST /v1/responses
  -> parse request and apply LLM request defaults/policies
  -> validate tool declarations and map configured built-ins
  -> inactive: use the existing single upstream call path
  -> active:
       create one absolute RuntimeBudget
       pin the initially selected provider, backend target, and request template
       repeat:
         render and authenticate the next model request
         call the pinned model backend within the remaining deadline
         buffer and translate the response into canonical Responses items
         aggregate this round's token usage
         authorize and validate all function_call items
         no calls: pass the reconstructed final response to process_response
         direct calls:
           reserve call budget
           group and execute backends with bounded concurrency
           sort results back into original model-call order
           append every raw intermediate output item
           append matching function_call_output items
         program call:
           validate and extract generated Python
           repeat E2B runCode with the bounded replay transcript
           execute each pending authorized tool through its existing backend
           stop at program_output or a bounded error
           append the raw synthetic function call and one synthetic function_call_output
  -> replace final-round usage with aggregate usage
  -> run existing final response policies and serialization once
  -> return only the final response
```

All intermediate response output is preserved, including reasoning items, because the next stateless Responses request
must carry the complete prior model output. `call_id` is the authoritative association between a function call and its
output.

Remote MCP discovery occurs before the first model call so only successfully discovered, schema-validated tools are
shown to the model. The facade deliberately does not reuse the inbound MCP `Router`, `Relay`, or `SessionManager`:
those components serve downstream MCP clients and implement multi-target proxy semantics that do not belong in the
Responses loop. It does reuse `PolicyClient`, `McpHttpClient`, and the Streamable HTTP client, which preserves the
normal outbound policy, TLS, tracing, body-limit, compression, JSON/SSE response parsing, and session-ID behavior.
Discovery also completes the programmatic catalog before the synthetic Python function is rendered, so a generated
program cannot name an MCP tool that was filtered, missing, invalid, or not authorized for programmatic use.

Every model round uses the provider/backend selected for the first round. AgentGateway rebuilds the request, reapplies
provider setup and backend authentication, and uses the same transport policies. When the managed runtime is active,
the proxy marks its response as handled so the general retry path cannot replay the managed request, including the
zero-tool-call case and any request that may already have performed side-effecting tool work.

### Parallel execution and Sandbox reuse

`parallel_tool_calls` is optional and must be a JSON boolean when present; other JSON types are rejected with a
sanitized OpenAI-compatible `400`. The field remains in every upstream model round.

If `parallel_tool_calls` is absent or `true`, ordinary HTTP calls are separate operations and may run concurrently up to
`maxParallelToolCalls`. Remote MCP calls are also independent operations and may run concurrently while reusing their
server's initialized MCP session. All E2B calls in the same model Response form one operation, so they share one Sandbox. The E2B
backend executes those calls sequentially and creates/removes a fresh Python context for each call. The E2B batch can
overlap with independent HTTP operations.

If `parallel_tool_calls` is `false`, operations execute serially in model order. Adjacent E2B calls are still combined
into a shared-Sandbox batch, but an intervening non-E2B call creates a sequencing boundary.

Results are always sorted back into original response order before `function_call_output` items are appended. Each E2B
batch can contain at most eight calls. With parallel execution, all E2B calls in one model Response share that single
batch; in serial mode, intervening non-E2B calls can split them into multiple adjacent-call batches. Each non-empty
`call_id` is limited to 256 UTF-8 bytes and each Python source string to 32 KiB.

Fresh contexts isolate Python interpreter variables, but calls in one Sandbox still share filesystem and process state.
A later model Response or client request creates a new Sandbox.

`parallel_tool_calls` continues to govern direct model-generated calls only. A Python program pauses at its first
unresolved `tools.call`, so program-owned calls are executed sequentially even when `parallel_tool_calls` is true. Each
record/replay pass uses one new Sandbox through the same E2B lifecycle as a single Code Interpreter execution. Reusing
the existing lifecycle avoids a second session abstraction and ensures cancellation and cleanup follow the established
path; the tradeoff is one Sandbox creation per replayed program call plus one final completion pass.

### HTTP Tool backend

Ordinary function requests use this envelope:

```json
{
  "tool_name": "get_weather",
  "call_id": "call_123",
  "arguments": { "city": "Hangzhou" },
  "context": {
    "request_id": "gateway-request-id",
    "deadline_ms": 5000
  }
}
```

The backend timeout is the smaller of its configured timeout and the request's remaining absolute budget. An optional
bearer token is added by AgentGateway. Responses are body-limited and normalized into a success value or an
application-level error. AgentGateway does not retry the request because the function may already have executed.

Web Search uses the flattened contract shown above instead of the ordinary function envelope. The example FC function
also enforces its own request, query, domain, provider-response, result-count, field-size, and serialized-output bounds;
it rechecks allowed domains after Tavily responds.

### E2B Code Interpreter backend

The direct E2B flow is:

```text
POST {apiUrl}/sandboxes
  templateID=code-interpreter-v1, secure=true
  -> choose effectiveDomain from the create response, falling back to configured domain
  -> for each call in model order:
       POST https://49999-{sandboxID}.{effectiveDomain}/contexts
       POST https://49999-{sandboxID}.{effectiveDomain}/execute
       DELETE https://49999-{sandboxID}.{effectiveDomain}/contexts/{contextID}
  -> DELETE {apiUrl}/sandboxes/{sandboxID}
```

Control-plane requests use `X-API-Key`. Data-plane access headers are forwarded only when the create response supplies
the corresponding E2B tokens. A valid optional domain from the trusted create response overrides the configured
fallback domain. The template, Python language, `/home/user` working directory, secure mode, and Jupyter port are fixed
in AgentGateway and cannot be overridden by the model or client.

The backend parses bounded NDJSON execution output into normalized `stdout`, `stderr`, execution errors, and
truncation state. It accepts but ignores E2B `result` and `number_of_executions` events. Successful Code Interpreter
output contains `exit_code`, `stdout`, `stderr`, `timed_out`, `truncated`, and an empty `artifacts` array. Rich
expression results and artifacts are not exposed. The backend reserves part of the absolute deadline for termination.
Context and execution operations are never retried. Sandbox termination accepts `204` and `404`, retries exactly once
after failure, and converts a definitive cleanup failure to a request-terminating infrastructure error even if code
execution completed.

Programmatic Tool Calling reuses this control-plane, Python-context, `/execute`, NDJSON, bound, deadline, and cleanup
implementation. Its generated wrapper is executed as Python source and interprets the normalized stdout/stderr result
as an internal nonce-bound program protocol. It does not add a JavaScript context, custom E2B template, persistent
Sandbox session, or network-policy control call. Program replays and direct Code Interpreter calls remain separate E2B
operations so existing direct-call batching semantics do not change.

### Error handling

Errors are divided into three classes:

- Request and limit errors return a sanitized OpenAI-compatible `400`. Examples include invalid declarations,
  unregistered or undeclared model calls, malformed arguments, schema mismatch, reserved names, unsupported active-mode
  fields, invalid caller modes, replay mismatch, and exhausted round/call/argument/Sandbox limits. No rejected call
  reaches a backend.
- Application errors are structured tool outputs returned to the model. Examples include Python exceptions and
  backend-declared per-call failures. A generated-program syntax/runtime failure or missing `program_output` becomes a
  structured synthetic-function error; a nested tool application error becomes a replay value. They consume round and
  call budget, allowing the model or program to repair or explain the result.
- Authentication, transport, protocol, configuration, and cleanup failures terminate the request with a sanitized
  `502`. Expiration of the one absolute deadline returns `504`.

Raw backend bodies, endpoint URLs, credentials, Sandbox IDs, source code, query text, stdout, and stderr are not placed
in client-visible infrastructure errors.

Client-selected MCP URLs are HTTPS-only in production, cannot contain URL userinfo or fragments, must resolve entirely
to public addresses, and are connected through a validated resolved address while TLS verifies the original hostname.
This prevents a second DNS lookup from rebinding the destination to loopback, link-local, private, documentation,
multicast, or other special-use ranges. The request's gateway authorization and unrelated downstream headers are not
copied to MCP; only the explicit MCP bearer authorization and tracing extensions enter the outbound context. MCP
transport/protocol failures are sanitized to the same request-terminating infrastructure error class as other managed
backends.

### Telemetry

The runtime records request outcomes, model-round counts and duration, calls and duration by configured tool and backend
kind, Sandbox operation counts and duration, limit exhaustion, and output truncation. Labels use configured tool names
and closed enums such as `http`, `e2b`, operation, and outcome.

User-controlled or high-cardinality content—including remote MCP server/tool names, `call_id`, query, code, output,
result URLs, endpoint, request ID, and Sandbox ID—is excluded from metric labels and runtime spans. Remote MCP calls use
the bounded telemetry tool label `remote_mcp`.

Debug logs are content-free for the same reason: the programmatic path logs program sizes, replay entry counts and
durations, never the program itself. Debugging a generated program does need the source, so it is emitted on a single
`trace` event on the `agentgateway::llm::tool_runtime::runner` target, off by default and enabled explicitly with
`RUST_LOG=info,agentgateway::llm::tool_runtime::runner=trace`. The split is deliberate: the source inlines the arguments
the model passed to tools, so raising a target to `trace` is the operator's opt-in to logging request content, while
`debug` stays safe to enable in production.

## Controller and xDS

The initial implementation adds no controller translation, CRD, protobuf, or xDS resource. `llm.toolRuntime` is a local
static configuration field. It is normalized into a compiled `ToolRegistry` during local configuration loading and then
attached to the LLM request policy used by configured models, including their failover paths.

A future Kubernetes-facing API would need explicit policy attachment, secret references, backend references, controller
validation, and xDS representation. That work is out of scope for this design.

## Policy Attachment

Tool Runtime configuration is part of the local LLM policy rather than a client-selected policy. AgentGateway merges
route and backend LLM policies using the existing ordering. Responses request defaults and request policies are applied
before managed-tool activation, so a policy may supply the `tools` field and activate the runtime.

The runtime itself does not introduce new attachment or precedence rules. It captures the canonical post-policy request,
then re-renders each subsequent round through the already selected provider and LLM request policy. Existing final
response policies and guardrails run only after the last model round.

## Compatibility and Migration

The feature is opt-in. Without `llm.toolRuntime`, or when a request declares no configured managed tool, AgentGateway
uses its previous single-upstream-call behavior. Client-owned function tools remain passthrough in that inactive mode.

When any managed tool activates the runtime, every function declaration in that request must be registered. Mixing a
managed tool with an unmanaged function is rejected rather than partially executing an ambiguous tool set. Active mode
also rejects `background: true` and non-null `conversation` state.

Clients receive the final model response rather than intermediate calls, and the final usage object is the saturating
sum of usage reported by all internal model rounds. Operators adopting the feature must provision the HTTP functions
and E2B-compatible service, configure credentials and network controls, and account for the additional model rounds and
tool latency.

For `stream: true`, AgentGateway changes every internal provider request to non-streaming, completes the same bounded
model/tool loop, applies final response policies, and serializes the final canonical Response as a valid Responses SSE
sequence (`response.created`, item/content events and deltas, then `response.completed`). The terminal event carries
aggregate usage, and any upstream `tools` echo is restored to the client's original declarations. This preserves the
streaming wire contract and prevents intermediate calls, reserved function names, and tool outputs from reaching the
client, but it does not reduce time to first token because the loop is buffered.

`stream_options` is removed only from the internal non-streaming request. AgentGateway validates and retains
`include_obfuscation` for client emission: delta events contain random obfuscation by default or when explicitly true,
and omit it when the client explicitly sets false. Only message content produces output-text/refusal deltas; reasoning
and other non-message items remain item lifecycle events and are retained in their completed form.

The design does not require changes to existing provider configuration. Provider conversion must support the Responses
function-calling shapes used in each round; all rounds remain pinned to the initially selected provider and backend.

Clients opt in to AgentGateway-managed Programmatic Tool Calling with `programmatic_tool_calling` plus at least one
tool whose `allowed_callers` includes `programmatic`. The final client response retains the original tool declarations
but does not expose the reserved synthetic function, generated Python, replay transcript, or intermediate tool
results. This is request-shape compatibility for models without native PTC, not native OpenAI JavaScript runtime or
intermediate-item compatibility.

## Risks and Tradeoffs

- **Gateway orchestration increases request duration and resource use.** Bounded rounds, calls, concurrency, output,
  backend timeouts, and one absolute deadline prevent unbounded loops.
- **Tool execution can have side effects.** Tool calls and managed responses are not automatically retried. Operators
  should make ordinary functions idempotent by `call_id` when practical.
- **Built-ins are compatibility mappings, not native OpenAI execution.** This makes backends portable and
  operator-controlled but does not reproduce native OpenAI call-item wire types or hosted-tool behavior.
- **Python PTC deliberately differs from OpenAI's JavaScript runtime.** It maximizes reuse of the existing E2B
  `runCode` path and works with upstream models that can generate Python function arguments, but code examples and
  intermediate runtime semantics are AgentGateway-specific.
- **Sequential replay increases Sandbox starts.** One new Sandbox per unresolved call plus one completion pass is
  operationally simpler and preserves existing cleanup guarantees, but it has higher latency than a callback channel or
  a persistent Program Sandbox.
- **The Program Sandbox retains E2B's default network access.** AgentGateway does not inject backend credentials, but
  generated Python can transmit user data or replayed tool results. Operators must account for that exposure and must
  not infer no-egress or ZDR-equivalent behavior.
- **MCP program results use the generic MCP result envelope.** Input schemas come from discovery, while programs may
  need to parse text content when a server does not advertise structured output. The Weather example demonstrates that
  explicit parsing path.
- **For direct Code Interpreter, one Sandbox per call would provide stronger isolation but higher startup cost.**
  Reusing one Sandbox per model Response minimizes lifecycle overhead. Separate contexts isolate interpreter variables,
  while the documented shared filesystem/process boundary remains. Program replay deliberately uses separate Sandboxes
  because host-side tool results are available only after each `runCode` execution ends.
- **One Sandbox across the whole client request would improve reuse further but retain state across reasoning rounds.**
  The selected boundary is one model Response so each subsequent reasoning round receives a fresh Sandbox.
- **Direct E2B integration couples the backend to a protocol.** It removes the operational FC Sandbox adapter and keeps
  lifecycle control in AgentGateway, at the cost of maintaining compatible control-plane and NDJSON parsing logic.
- **Buffering is required.** The runtime needs complete intermediate Responses output for validation, ordering, usage,
  and continuation. A streaming client receives SSE only after the final round has completed, so this mode provides
  protocol compatibility rather than incremental time-to-first-token.
- **Fail-closed active mode limits composition with unmanaged functions.** This avoids accidental local execution or an
  incomplete loop but means hybrid client-side/server-side tool handling requires a future explicit contract.
- **Web Search quality and availability depend on Tavily and FC.** The gateway treats the function as a bounded backend;
  provider quotas, regional deployment, and result relevance remain operational concerns.

Alternatives considered for the original direct Tool Runtime were native passthrough to provider-hosted tools, a
separate external agent orchestrator, a generic code-executing HTTP function, and one Sandbox per direct code call.
They were not selected because they respectively reduce portability/operator control, move the Responses loop outside
AgentGateway, weaken the purpose-built Sandbox boundary, or add avoidable Sandbox lifecycle overhead.

For the Programmatic Tool Calling extension, alternatives were a Sandbox-to-Gateway callback API and a local embedded
Python runtime. The callback route would require a reachable request-scoped endpoint and credential protocol; the local
runtime would add a new code-execution security boundary inside AgentGateway. Sequential E2B record/replay was selected
because it reuses the existing backend and keeps all tool authorization in the gateway.

## Test Plan

- Validate configuration defaults, required tools, duplicate names/built-ins, backend pairing, URL/domain/timeout rules,
  environment/file secret handling, redaction, normalization snapshots, and failover registry reuse.
- Unit-test built-in mapping, trusted-option separation, reserved names, unsupported fields, schema validation,
  malformed model calls, size limits, result normalization, and output truncation.
- Unit-test remote MCP declaration validation, authorization redaction, automatic execution for `auto`/`never`,
  rejection of approval-resume policies, allowed-tool filtering,
  schema import, forced tool choice translation, bearer authentication, session reuse, and tools/call execution through
  the shared `PolicyClient`/`McpHttpClient` path.
- Unit-test `allowed_callers` on built-ins and MCP declarations; verify programmatic-only tools are
  hidden from direct calls, dual-mode tools retain both routes, and invalid direct/programmatic caller combinations
  fail closed.
- Unit-test Python synthetic-function generation after MCP discovery, including bounded descriptions, public MCP names,
  imported input schemas, generic MCP result documentation, and removal of client credentials.
- Unit-test the program wrapper and state machine for one call, multiple sequential calls, loops, conditions, JSON
  scalar/object/list results, deterministic replay, nonce-bound protocol parsing, Python syntax/runtime errors, missing
  or repeated `program_output`, undeclared tools, argument schema mismatch, call limits, output limits, and deadline
  expiration.
- Unit-test HTTP Tool request envelopes, bearer authentication, deadline propagation, body limits, application
  errors, and sanitized transport/protocol failures.
- Test the E2B wire protocol with local mock control and data planes: authentication headers, create response parsing,
  fixed template and Python context, bounded NDJSON parsing, per-call context removal, stable sequential execution, one
  Sandbox per batch, and exactly one termination retry.
- Test zero-call, single-round, multi-round, round-limit, call-limit, and absolute model-header, model-body, and
  tool-time deadline behavior through the proxy.
- Test bounded parallelism and prove that Web Search overlaps an E2B batch while final outputs retain model order.
- Test usage aggregation, provider/backend pinning, full intermediate-output preservation, final-only response policies,
  and disabling general retries whenever the managed runtime handles a response.
- Test `stream: true` through the proxy: internal rounds request JSON, intermediate tool events remain private, emitted
  SSE events satisfy the strict Responses event types, output deltas contain only the final answer, and
  `response.completed` contains aggregate usage.
- Run the Python Web Search suite against a fake Tavily transport for ingress authentication, request validation, domain
  enforcement, provider failures, normalization, and output bounds.
- Build the release binary and run `examples/llm-tool-runtime/functional_test.py` through public `/v1/responses` for Web
  Search, same-Response multi-code Sandbox reuse, dual-tool overlap, active-mode rejection, and unmanaged passthrough.
- Keep live FC Web Search and E2B smoke tests opt-in behind `AGENTGATEWAY_LIVE_TOOLS=1`; never require cloud credentials
  for default unit or functional tests.
- Run the opt-in live WeatherAPI MCP case through the release binary, assert its final Beijing-weather marker, and
  verify a successful `remote_mcp` metric delta.
- Add a hermetic Programmatic MCP Weather case that discovers `get_forecast`, makes the generated Python call
  `tools.call("weather.get_forecast", ...)`, replays a three-day provider result, and verifies three daily extrema plus
  period extrema.
- Add the opt-in `programmatic-mcp-weather` live case using the existing FC WeatherAPI Remote MCP server. Require
  `PROGRAMMATIC_MCP_WEATHER_3DAY_OK`, three forecast-day high/low entries, a successful `remote_mcp` metric delta, and
  successful E2B Sandbox operations.
- Unit-test Tool Search declaration validation: a duplicate or optioned `tool_search`, `defer_loading` without
  `tool_search`, `tool_search` with nothing deferred after discovery, `defer_loading` combined with programmatic
  `allowed_callers`, and a `tool_choice` forcing a deferred function.
- Unit-test that a deferred Remote MCP server contributes no function declarations, that exactly one
  `_agentgateway_tool_search` declaration is produced, and that its index names the server and its public tool names but
  never the reserved internal names.
- Unit-test that a search appends matching declarations after the existing ones, that a repeated search still reports the
  definitions without declaring a name twice, and that a zero-scoring candidate stays deferred.
- Unit-test that a `namespace` declaration is never forwarded, that its members are injected under their original names,
  and that the namespace name and description are composed onto each member description.
- Unit-test the deferred-catalog, search-index, and search-call bounds, including that exceeding the call limit yields a
  `tool_search_limit` application error rather than a request failure, and that search calls count against
  `max_tool_calls`.
- Test the full loop through the runner and through the proxy: the first round declares only the search function, the
  second round contains the searched declaration and replays the search output, the request terminates normally, and no
  `_agentgateway_tool_search` item or the deferred catalog reaches the client while the client's original `tools` array
  round-trips verbatim.
- Add a `tool-search-deferred` release-binary functional case that drives a deferred managed function through public
  `/v1/responses` against a local HTTP tool mock, asserting three model rounds, the round-one index without a
  declaration, the round-two appended declaration, one tool-backend request, aggregate usage, and a client body free of
  reserved names.

## Open Questions

- What Kubernetes API and policy attachment should expose managed tools and secret references?
- Should a future active mode support an explicit split between gateway-managed and client-managed functions?
- Should a future streaming mode expose incremental final-round tokens, and how could it do so without leaking a late
  managed Tool Call that requires another hidden round?
- Should future Code Interpreter versions support request-scoped persistent state or downloadable artifacts, and what
  additional isolation and storage controls would that require?
- Should additional hosted-tool compatibility mappings use the same reserved-function mechanism or provider-native
  execution when a selected provider supports them?
- Should a later Programmatic Tool Calling version add `tools.call_many` or a persistent E2B Sandbox after live latency
  data establishes that sequential record/replay is a bottleneck?
- Should AgentGateway eventually offer an operator-controlled no-egress Program Sandbox mode in addition to the initial
  default E2B network behavior?
