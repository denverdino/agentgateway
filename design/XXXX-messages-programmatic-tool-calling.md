# EP-XXXX: Gateway-Executed Programmatic Tool Calling for Anthropic Messages

- Issue: [#XXXX](https://github.com/agentgateway/agentgateway/issues/XXXX)
- Related: [EP-3180](3180-responses-tool-runtime.md)
- Status: proposed
- Date: 8/28/2026

> **Note:** This design reflects the proposal as of the date above. The current implementation may differ as the design
> is implemented, reviewed, or revised.

## Summary

EP-3180 added a managed Tool Runtime to AgentGateway's OpenAI-compatible `/v1/responses` endpoint, including
AgentGateway-executed Programmatic Tool Calling: the model writes one Python program, AgentGateway runs it in an E2B
Sandbox, and program-owned tool calls are resolved through operator-configured backends. That runtime is reachable only
from `process_responses_request`, so a client speaking the Anthropic Messages API cannot use it.

This design brings Programmatic Tool Calling to the inbound Anthropic Messages surface. A client POSTs an ordinary
Anthropic request to `/v1/messages` declaring a `code_execution` tool and marking managed tools with
`allowed_callers`, exactly as it would against Claude. AgentGateway asks the selected model for a Python program,
executes it through the existing E2B record/replay protocol, resolves program-owned calls to managed functions and
managed Web Search through the same authorized backends, and returns only the final Anthropic message.

Every client built-in is rewritten to an Anthropic **custom tool** in the model-facing request. AgentGateway never asks
the upstream for a native `code_execution` tool and never relies on a provider-native Programmatic Tool Calling
implementation. That is what makes the feature work against every model the gateway fronts, including models that do
not implement Programmatic Tool Calling at all, rather than only against Claude. As a direct consequence, no
`server_tool_use`, `code_execution_tool_result`, `container`, or `caller` block ever crosses the client boundary, and
no `anthropic-beta` header participates in the feature.

Reaching the Messages path requires generalizing the runtime over its inbound wire format. The format-bound half of the
runtime is extracted behind a `ManagedConversation` trait with two implementations; everything below that seam — the
registry, backends, E2B lifecycle, program wrapper, record/replay state machine, budget, and telemetry — is reused
without modification. Changes outside `crates/agentgateway/src/llm/tool_runtime/` are additive and confined to two
files, `llm/mod.rs` and `proxy/httpproxy.rs`; no type-crate or `conversion/` change is needed.

## Background

The EP-3180 runtime is already format-agnostic below a narrow seam. `registry.rs`, `backend.rs`, `http_backend.rs`,
`e2b.rs`, `program.rs`, `program_wrapper.py`, `schema.rs`, `transport.rs`, `telemetry.rs`, `RuntimeBudget`, and
`execute_batch` never name a wire type. The record/replay protocol in particular is completely independent of the
inbound format: it exchanges a public tool name, canonical JSON arguments, and a JSON result.

The coupling to OpenAI Responses is confined to ten places:

| Concern | Location |
| --- | --- |
| Canonical request | `tool_runtime/mod.rs:337` (`canonical_request: responses::Request`) |
| Declared-tool set | `tool_runtime/mod.rs:793` (`rest_field("tools")` filtered to `type == "function"`) |
| Collect model calls | `tool_runtime/mod.rs:787` (scans `response.output` for `OutputItem::FunctionCall`) |
| Append round history | `tool_runtime/mod.rs:390` (`append_raw_input_values`, then `tool_choice` reset) |
| Tool output item | `tool_runtime/runner.rs:370` (`function_call_output`) |
| Usage aggregation | `tool_runtime/mod.rs:1593` (`ToolRuntimeSummary.usage: responses::Usage`) |
| Finalize | `tool_runtime/mod.rs:1611` (`finalize_managed_response`) |
| Serialize and SSE | `tool_runtime/mod.rs:1639`, `:1646` |
| Upstream re-render | `llm/mod.rs:1956` (`rerender_responses_request`) |
| Error envelope | `ToolRuntimeError::into_openai_response` |

The inbound Anthropic surface itself already exists and is first class. `RouteType::Messages` and
`InputFormat::Messages` (`crates/llm/src/lib.rs:156`, `:181`) are distinct from the upstream `ChatFormat`, the default
route table maps `/v1/messages` plus Vertex `:rawPredict` and `:streamRawPredict` to `RouteType::Messages`
(`llm/model_router.rs:62`), `process_messages_request` (`llm/mod.rs:1727`) parses `types::messages::Request`, and
`CHAT_TRANSLATIONS` (`llm/mod.rs:320`) already renders a Messages request upstream as `AnthropicMessages`,
`OpenAICompletions`, `OpenAIResponses`, or `BedrockConverse`. Nothing on that path touches `tool_runtime`.

Anthropic server tools currently survive as opaque passthrough. `ServerTool` flattens unknown fields into `extra`
(`crates/llm/src/types/messages.rs:1234`), and the typed `ContentBlock` carries a `#[serde(other)]` unit variant
`Unknown` (`:753`) that lets unmodelled response blocks parse without failing. So a client declaring
`code_execution_20260120` against a Claude upstream works today with no gateway semantics whatsoever. That
passthrough is not this feature: it requires the upstream to implement Programmatic Tool Calling, it gives the operator
no control over what the program may call, and it executes tools outside AgentGateway's authorization, policy, and
telemetry path.

Note that the typed `Unknown` variant is a *unit* variant: it tolerates an unknown block but does not retain its
contents. Only the loose request-side `ContentPart::Unknown(serde_json::Value)` (`:57`) preserves a block verbatim.
That asymmetry is the reason this design appends raw upstream content rather than re-serializing from the typed form,
and it is why the fidelity invariant is pinned by a test.

Two upstream Anthropic behaviors are relevant and were verified against the published documentation. Programmatic Tool
Calling requires no beta header and is not its own tool type: it is the code-execution tool (version
`code_execution_20260120` or later) plus `allowed_callers` on other tools. And `allowed_callers` is documented as *not*
a security boundary — "Do not rely on `allowed_callers` as a security boundary" — because Anthropic validates it for
presentation and `tool_choice` rather than enforcing it.

## Goals

- Accept Anthropic-shaped Programmatic Tool Calling declarations on the inbound Messages surface and execute the
  generated program inside AgentGateway.
- Make the feature independent of the selected upstream, so it works against Claude, OpenAI-compatible, and Bedrock
  Converse providers alike, including models with no native Programmatic Tool Calling support.
- Let a generated program invoke request-authorized managed functions and managed Web Search through the existing
  `tools.call(name, arguments)` interface.
- Keep tool destinations, credentials, backend selection, and hard limits under operator control, and treat
  `allowed_callers` as a real authorization boundary.
- Reuse the EP-3180 registry, backends, E2B lifecycle, program wrapper, record/replay state machine, budget, limits, and
  telemetry without modification, and require no configuration change from an operator who already enabled the runtime.
- Preserve Anthropic conversation fidelity across internal rounds, including `thinking` blocks with their `signature`
  and any `cache_control` placement.
- Accept `stream: true` and emit a valid Anthropic event sequence carrying aggregate usage.
- Confine invasive change to `tool_runtime/`; keep every edit outside that package additive.
- Preserve the existing Messages request path exactly when the runtime is not active.

## Non-Goals

- Remote MCP on the Messages route. Anthropic's `mcp_servers` and `mcp_toolset` declarations are rejected while the
  runtime is active. Anthropic itself documents MCP tools as not programmatically callable.
- Tool Search and `defer_loading` on the Messages route.
- Client-owned tools with pause and resume. AgentGateway does not emit a paused `tool_use` carrying `caller`, does not
  return a `container`, and holds no container state across client requests. EP-3180's non-goal of persistent Sandboxes
  across requests is retained unchanged.
- Passthrough to a provider-native Programmatic Tool Calling runtime, or equivalence with Anthropic's container
  runtime. Programs are Python in E2B.
- Exposing `server_tool_use`, `code_execution_tool_result`, `container`, or `caller` blocks to clients. Program calls
  and their results remain internal to the managed loop, exactly as `program` and `program_output` do on Responses.
- Programmatic invocation of Code Interpreter as a nested tool. On the Messages route the code-execution declaration is
  the program runtime itself; a program that needs to evaluate code writes that code directly.
- Native Anthropic result-block compatibility. Because AgentGateway emits neither
  `code_execution_tool_result`/`code_execution_result` nor
  `bash_code_execution_tool_result`/`bash_code_execution_result`, the published inconsistency between those two
  families does not need to be resolved for this design.
- Token counting for active-mode requests. `RouteType::AnthropicTokenCount` parses as `InputFormat::CountTokens`, which
  is a distinct variant from `InputFormat::Messages`, so it never activates the runtime.
- A new Kubernetes API, controller translation, or xDS resource.

## API

### Client request

Clients use the ordinary Anthropic Messages shape. Programmatic Tool Calling is opted into with a PTC-capable
code-execution declaration plus at least one tool whose `allowed_callers` names a code-execution version:

```json
{
  "model": "smart",
  "max_tokens": 1024,
  "messages": [
    { "role": "user", "content": "Find the three most recent orders over $500 and total them." }
  ],
  "tools": [
    { "type": "code_execution_20260120", "name": "code_execution" },
    {
      "name": "query_orders",
      "description": "Query the orders database. Returns rows as JSON objects.",
      "input_schema": {
        "type": "object",
        "properties": { "since": { "type": "string" } },
        "required": ["since"]
      },
      "allowed_callers": ["code_execution_20260120"]
    }
  ],
  "stream": false
}
```

`query_orders` must also exist in the operator registry. The generated program calls it as
`tools.call("query_orders", {"since": "..."})`, and managed Web Search as `tools.call("web_search", {"query": "..."})`.

### Declaration mapping

Every managed declaration is rewritten to an Anthropic custom tool before the first model round. The model therefore
emits only ordinary `tool_use` blocks.

| Client declaration | Gateway action |
| --- | --- |
| `{"type": "code_execution_20260120"\|"code_execution_20260521", "name": "code_execution"}` | Requires an operator `codeInterpreter` E2B tool. Rewritten to `_agentgateway_code_interpreter` for direct calls. When at least one other tool is programmatically callable, `_agentgateway_programmatic_tool_calling` is additionally declared as the program runtime. |
| `{"type": "code_execution_20250522"\|"code_execution_20250825", "name": "code_execution"}` | Direct code interpreter only, matching Anthropic's own version floor for Programmatic Tool Calling. Rewritten to `_agentgateway_code_interpreter`. |
| `{"type": "web_search_2025*"\|"web_search_2026*", "name": "web_search", ...}` | Requires an operator `webSearch` HTTP tool. `allowed_domains`, `blocked_domains`, `user_location`, and `max_uses` are validated and retained as trusted gateway-side options. Rewritten to `_agentgateway_web_search` with one required `query` string. |
| Custom tool whose `name` is in the operator registry | Managed function. Direct route keeps the client-visible name and `input_schema`; programmatic route adds it to the program catalog. |
| Anything else — `mcp_toolset`, `mcp_servers`, `computer_toolset_*`, `browser_toolset_*`, unrecognized server tools | Rejected while the runtime is active. |

`allowed_callers` is stripped from the model-facing request; it is a gateway authorization flag, not model-visible
input. A programmatic-only tool is absent from the model-facing `tools` array entirely.

`code_execution_20260521` normalizes to `code_execution_20260120` on read, matching Anthropic's documented behavior
that response blocks always tag the caller as `code_execution_20260120` regardless of the declared version. Both
values are accepted on input. The normalized values map onto the existing `AllowedCallers` type
(`tool_runtime/mod.rs:159`), so caller parsing is shared with the Responses mapper.

Two Anthropic behaviors are mirrored rather than simplified, so that a client written against Claude does not silently
behave differently through the gateway:

- On `web_search_20260209` and later, `allowed_callers` **defaults** to `["code_execution_20260120"]`, not
  `["direct"]`.
- A request that leaves that default in place without declaring a code-execution tool is rejected with the documented
  message instructing the caller to set `allowed_callers: ["direct"]`.

### Rejections

The following are rejected as `invalid_request_error` in the Anthropic error envelope, before any backend is contacted:

- A code-execution declaration with no operator `codeInterpreter` E2B tool configured. The E2B backend supplies the
  Python runtime, so Programmatic Tool Calling cannot be served without it.
- `allowed_callers` naming a code-execution version when no code-execution tool is declared.
- A custom tool name that is not in the operator registry. Active mode is fail-closed, matching EP-3180 rather than
  partially executing an ambiguous tool set.
- A client tool name in the reserved `_agentgateway_` namespace.
- `mcp_servers`, `mcp_toolset`, or a computer or browser toolset.
- A `container` request field. This design holds no container state.
- `tool_choice` forcing a tool whose `allowed_callers` omits `direct`, matching Anthropic.
- `tool_choice.disable_parallel_tool_use: true` together with Programmatic Tool Calling, matching Anthropic.
- A recursive `$ref` cycle in the `input_schema` of a programmatically callable tool. Anthropic rejects this with
  `Circular $ref detected`. AgentGateway's only current `$ref` control is the `RejectExternalReferences` retriever
  (`tool_runtime/schema.rs:31`), which blocks *external* references; internal cycles are left to the `jsonschema`
  crate's own compile behavior. Whether an explicit cycle check is needed, or whether `ArgumentSchema::compile`
  (`schema.rs:11`) already fails on a cycle, must be established by test before implementing this rejection.

Unlike Anthropic, AgentGateway enforces `allowed_callers` as a hard boundary. A programmatic-only tool is genuinely
withheld from the model's direct declarations, and a direct model call naming it is rejected.

### Configuration

No configuration change. `llm.toolRuntime` is reused exactly as documented in EP-3180 — the same `tools` list, the same
`webSearch`, `codeInterpreter`, and ordinary function backends, the same secret handling, and the same limits. The
compiled `ToolRegistry` already hangs off the LLM request policy (`llm/policy/mod.rs:163`), and
`process_messages_request` reaches that policy through the shared `prepare_chat_request` path, so an operator who
configured the runtime for Responses gets Messages support with no edits.

Activation is keyed on `InputFormat::Messages` rather than on the request path, so Vertex `:rawPredict` and
`:streamRawPredict` routes behave identically to `/v1/messages`.

## Runtime Design

### The `ManagedConversation` seam

The format-bound half of the runtime moves behind one trait with two implementations. Folding usage handling into the
trait keeps `runner::run` from naming any wire type at all.

```rust
#[async_trait]
pub(crate) trait ManagedConversation: Send {
    /// Buffered and translated upstream round for this wire format.
    type Round: Send;
    /// Client-facing managed result for this wire format.
    type Final: Send;

    fn collect_model_calls(&self, round: &Self::Round) -> Result<CollectedToolCalls, ToolRuntimeError>;
    fn accumulate_usage(&mut self, round: &Self::Round);
    fn append_round_history(&mut self, round: Self::Round, outputs: Vec<Value>);
    fn tool_output_item(
        &self,
        call_id: &Strng,
        result: ToolExecutionResult,
        max_output_bytes: usize,
    ) -> Result<Value, ToolRuntimeError>;
    fn finalize(self, round: Self::Round, budget: &RuntimeBudget) -> Self::Final;
    fn state(&self) -> &ManagedToolState;
    fn state_mut(&mut self) -> &mut ManagedToolState;
    fn format(&self) -> ManagedFormat;
    async fn execute_tool_search(
        &mut self,
        _arguments: Value,
        _max_output_bytes: usize,
    ) -> Result<ToolExecutionResult, ToolRuntimeError> {
        Err(ToolRuntimeError::internal())
    }
```

A trait with per-format implementations is preferred over an enum with per-function `match` arms because each format
then reads top to bottom in its own file, and because the enum would size one struct to its largest variant while the
two runtimes are already boxed separately at the call site (`httpproxy.rs:2469`).

`ManagedToolState` is extracted from `PreparedToolRuntime` (`tool_runtime/mod.rs:337`) and holds only the genuinely
format-agnostic fields and methods, including the registry, programmatic catalog, parallel flag, deadline, and
program-call resolution. `pending_remote_mcp`, `tool_search`, `execute_tool_search`, `initialize_remote_mcp`,
`install_remote_mcp_tools`, and the schema refreshers remain on `PreparedToolRuntime`: each reads or mutates the
Responses declaration array through `canonical_request.rest_field("tools")`. The trait provides a default
`execute_tool_search` that returns `Err(ToolRuntimeError::internal())`, so Messages cannot accidentally enter the
Responses-only path. `PreparedToolRuntime` keeps that shared state plus its `responses::Request` and echo/stream fields;
the new `PreparedMessagesRuntime` keeps shared state plus its `messages::Request`. This avoids embedding an unused
`responses::Request` inside the Messages runtime without pretending request-bound operations are format-agnostic.

`ModelRound` (`tool_runtime/runner.rs:26`) becomes `ModelRound<R>`; the loop body in `runner::run` is otherwise
unchanged.

### Components

New files, all inside `tool_runtime/`:

- `messages/mapper.rs` validates Anthropic tool declarations, rewrites built-ins, retains trusted Web Search options,
  parses `allowed_callers`, and decides whether the runtime is active.
- `messages/conversation.rs` implements `ManagedConversation` for `PreparedMessagesRuntime`.
- `messages/stream.rs` encodes a completed managed Messages response as an Anthropic event sequence, mirroring the
  placement of `encode_managed_streaming_response` (`mod.rs:1646`) for Responses.

The Responses mapper stays flat at `tool_runtime/mapper.rs` rather than moving to a matching `responses/` subdirectory.
The asymmetry is deliberate: `remote_mcp/` already establishes the subdirectory precedent for a cohesive feature, and
grouping only the new code keeps the diff and the rebase conflict surface small. A later symmetry pass is a pure
rename.

Field access on the canonical request needs no new type-crate API. `rest_field` and `replace_rest_field` are already a
private extension trait, `ResponsesRequestExt` (`tool_runtime/mod.rs:62-81`), over the flattened `rest` value; the
Messages equivalent is a `MessagesRequestExt` beside it, since `messages::Request` carries the same
`#[serde(flatten)] pub rest` (`crates/llm/src/types/messages.rs:29`). Appending a round needs no helper either, because
`pub messages: Vec<RequestMessage>` (`:15`) is directly pushable. So unlike the Responses path — which reaches into
`append_raw_input_values` (`crates/llm/src/types/responses.rs:476`) — this design requires **no change to
`crates/llm/src/types/messages.rs` at all**.

Additive changes outside the package:

| File | Change |
| --- | --- |
| `crates/agentgateway/src/llm/mod.rs` | `rerender_messages_request`, `buffer_messages_round_response`, `translate_prebuffered_messages_round`, `BufferedMessagesRound`, one `ResponseProcessingInput` variant, and the `prepare` plus `streaming = false` calls inside the existing `process_messages_request` (`:1727`) that `process_responses_request` already makes at `:1893` and `:1898`. |
| `crates/agentgateway/src/proxy/httpproxy.rs` | `PinnedMessagesRoundTrip` and `run_managed_messages_runtime`, mirroring `:2351-2517`, plus one branch in the existing `RouteType::Messages` arm (`:2889`). |

Two files outside `tool_runtime/`, both additive. No `conversion/` change is required either: the `Messages` request
translations already exist, so `rerender_messages_request` is its Responses twin over a different `ChatRequest`
variant. `messages_translation_for_route` is a widened allowlist of `responses_translation_for_route`, not a
structurally different match: `provider_format` maps both `ChatFormat::BedrockConverse + InputFormat::Messages` and
native Anthropic to `ProviderFormat::Messages`, so the route type alone cannot distinguish them and the existing
provider-keyed special case must cover both.

### Request flow

```text
client POST /v1/messages
  -> parse request and apply LLM request defaults/policies
  -> validate tool declarations and rewrite managed built-ins to custom tools
  -> inactive: use the existing single upstream call path
  -> active:
       create one absolute RuntimeBudget
       pin the initially selected provider, backend target, and request template
       repeat:
         render and authenticate the next model request
         call the pinned model backend within the remaining deadline
         buffer and translate the response into a canonical Messages response
         aggregate this round's token usage
         authorize and validate all tool_use blocks
         no tool_use blocks: finalize and return
         direct calls:
           reserve call budget
           group and execute backends with bounded concurrency
           sort results back into content order
           append the assistant message verbatim
           append one user message of tool_result blocks
         program call:
           validate and extract generated Python
           repeat E2B runCode with the bounded replay transcript
           execute each pending authorized tool through its existing backend
           stop at program_output or a bounded error
           append the assistant message verbatim
           append one user message with the synthetic tool_result
  -> replace final-round usage with aggregate usage
  -> run existing final response policies and serialization once
  -> return only the final message
```

### Round history

After a round that produced calls, two messages are appended to the canonical request:

```json
{ "role": "assistant", "content": [ "<raw content blocks, verbatim>" ] }
{ "role": "user", "content": [ { "type": "tool_result", "tool_use_id": "toolu_...", "content": "<json>" } ] }
```

The assistant message carries content through `messages::Content`, whose unknown block variant preserves fields through
`#[serde(flatten)]`; `thinking` signatures and `cache_control` therefore survive round history without a separate
`raw_output` vector or `merge_final_output` helper. Final non-streaming serialization is simply
`serde_json::to_vec(&response)`. Anthropic's requirement that a `tool_result` message immediately follow its `tool_use`
message, and that a `thinking` block lead the assistant turn, both follow from appending in this order.

`tool_result` blocks are emitted in the content order of their `tool_use` blocks, preserving EP-3180's guarantee that
results are sorted back into model order. `tool_use.id` plays the `call_id` role, so the existing
`SANDBOX_MAX_CALL_ID_BYTES` bound of 256 stays meaningful.

Two shape differences from Responses:

- Anthropic `tool_use.input` is a JSON object, not the string `arguments` Responses uses. `collect_model_calls` reads
  it directly and enforces `maxArgumentsBytes` against the serialized form rather than parsing a string.
- A `ToolApplicationError` becomes a `tool_result` with `is_error: true` carrying the structured error JSON as text,
  which is the idiomatic Anthropic mapping, rather than a success block carrying `ok: false`.

After outputs are appended, `tool_choice` is reset to `{"type": "auto"}` for the same reason as on Responses: a forced
choice must not trap the model in an unbounded call sequence. A round whose assistant content contains no `tool_use`
block is terminal. A program call mixed with direct calls in one assistant message is rejected, matching the existing
rule that keeps authorization and replay association unambiguous.

### Parallelism and Sandbox reuse

Anthropic's control is `tool_choice.disable_parallel_tool_use`, so the runtime's `parallel` flag is that field
inverted, defaulting to parallel. Because a Programmatic Tool Calling request rejects
`disable_parallel_tool_use: true`, such a request is always parallel-eligible; direct managed batches honor the
client's setting. `maxParallelToolCalls`, E2B batching, one Sandbox per model response with a fresh Python context per
call, and one Sandbox per program replay pass all behave exactly as EP-3180 specifies, because `backend.rs`, `e2b.rs`,
and `program.rs` are reused unmodified.

### Streaming

Internal rounds are always non-streaming: `stream: true` is removed from the model-facing request, the loop completes,
final response policies run once, and only then is the response encoded. Anthropic has no `stream_options` and no
`include_obfuscation` analogue, so both Responses-specific concerns drop away.

The emitted sequence matches the existing golden snapshots
(`crates/llm/src/tests/response/responses/stream.responses-messages-streaming.snap`,
`.../anthropic/stream_thinking.messages-messages-streaming.snap`):

```text
message_start        -> message with content: [], stop_reason: null
per content block:
  content_block_start  -> {index, content_block: <empty-shell block>}
  content_block_delta  -> text_delta | thinking_delta then signature_delta
  content_block_stop   -> {index}
message_delta        -> {delta: {stop_reason, stop_sequence}, usage}
message_stop
```

A `text` block emits its whole text as one `text_delta`. A `thinking` block emits `thinking_delta` then
`signature_delta`. Any block type without a delta representation, including `redacted_thinking`, is emitted complete
inside `content_block_start` followed immediately by `content_block_stop` — the same rule EP-3180 applies to non-message
Responses items. There is no `data: [DONE]` sentinel; Anthropic streams end at `message_stop`.

One deliberate deviation from the neighbouring translator snapshot: `stream.responses-messages-streaming.snap`
zero-fills `usage` in `message_start` because a live translation does not yet know the totals. Here the loop has
already completed, so aggregate `input_tokens` and non-zero cache fields are emitted in `message_start` and aggregate
`output_tokens` in `message_delta`, which is both what Anthropic does and what SDK clients read.

Non-streaming returns the final Messages response as JSON with aggregate usage, the last round's `stop_reason`, no
`container`, and no `tools` echo. In both modes the client sees no `_agentgateway_*` name, no reserved-function
`tool_use`, no generated Python, and no replay transcript.

### Error handling

The three EP-3180 error classes are unchanged; only the envelope differs, selected through the existing
`ChatErrorFormat::Anthropic` mapping (`llm/mod.rs:1161`):

- Request and limit errors return a sanitized `invalid_request_error`.
- Application errors are returned to the model as `tool_result` blocks with `is_error: true`, consuming round and call
  budget so the model or program can repair or explain the result. A generated-program syntax or runtime failure, or a
  missing `program_output`, becomes one structured synthetic error; a nested tool application error becomes a replay
  value.
- Authentication, transport, protocol, configuration, and cleanup failures terminate the request as `api_error`.
  Expiration of the one absolute deadline returns the deadline status EP-3180 already uses.

Raw backend bodies, endpoint URLs, credentials, Sandbox IDs, source code, query text, stdout, and stderr are never
placed in client-visible infrastructure errors. As today, a managed response is marked handled so the general retry
path cannot replay a request that may already have executed side-effecting tools.

### Telemetry

Existing metric families, closed-label enums, and content-free logging are reused. The request-level
`tool_runtime_requests` family adds one bounded `format` label with values `responses` and `messages`, so an operator
can distinguish the two inbound surfaces. Per-round metrics remain unchanged; format cardinality is not multiplied
across model rounds. Generated Python remains available only on the `trace` target
`agentgateway::llm::tool_runtime::runner`.

## Controller and xDS

None. `llm.toolRuntime` remains a local static configuration field compiled into a `ToolRegistry` at load time and
attached to the LLM request policy. A future Kubernetes-facing API is out of scope, as in EP-3180.

## Policy Attachment

Unchanged. The runtime configuration is part of the local LLM policy, not a client-selected policy. Route and backend
LLM policies merge using the existing ordering, request defaults and request policies are applied before managed-tool
activation, and existing final response policies and guardrails run once after the last model round.

## Compatibility and Migration

Strictly opt-in. A Messages request declaring no managed tool takes today's path unchanged, including a request that
declares Anthropic server tools the gateway forwards opaquely. No operator configuration change is required.

When the runtime does activate, active mode is fail-closed exactly as on Responses: every custom tool in that request
must be registered, and an unregistered one is rejected rather than partially executed. Operators who front Claude and
today rely on native Programmatic Tool Calling passthrough keep that behavior as long as they do not configure a
managed registry entry matching the declared tool names.

Clients receive the final message rather than intermediate calls, and final usage is the saturating sum across internal
model rounds of `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`.

## Risks and Tradeoffs

- **This is not Claude's native runtime.** Programs are Python in E2B, so stdout semantics, the roughly four-minute
  pending-call timeout, and the ninety-second per-cell limit documented for Anthropic do not apply; AgentGateway's own
  bounds do. The mirror image is the payoff: because the gateway never requests native Programmatic Tool Calling, the
  feature works on models that lack it, including Claude Haiku 4.5, which is documented as accepting the newer tool
  `type` strings while silently degrading to non-PTC behavior.
- **`allowed_callers` becomes a real authorization boundary**, which is stricter than Anthropic, who explicitly warn
  against relying on it. Stricter is the correct default for a gateway, but a program that Anthropic would allow to
  call a `["direct"]`-only tool is rejected here.
- **Extended thinking is the sharpest correctness edge.** Anthropic requires `thinking` blocks and their signatures to
  be returned unmodified. Verbatim raw-content append satisfies this, but the invariant fails silently if a future
  change re-serializes from the typed form, so it is pinned by a test rather than by a comment.
- **The `ManagedToolState` extraction edits shipped code** inside `tool_runtime/`. The existing runtime suite is the
  safety net, and zero snapshot churn is the acceptance criterion for the extraction being behavior-preserving.
- **Buffering is still required**, so a streaming Messages client gets protocol compatibility rather than improved time
  to first token.
- **Rewriting built-ins to custom tools costs native fidelity.** Clients never see `server_tool_use`,
  `code_execution_tool_result`, `container`, or `caller`, so a client that inspects those blocks must be adapted. The
  benefit is upstream independence and the elimination of any dependency on unresolved details of Anthropic's result
  block families.
- **Programmatic Code Interpreter is deliberately absent.** On this route the code-execution declaration *is* the
  program runtime, so nesting it would mean a Sandbox inside a Sandbox for no capability gain.

Alternatives considered:

- **Translate at the edge**: convert the inbound Messages request into a `responses::Request`, run the existing loop
  unchanged, and translate back. Rejected. Each round re-renders the canonical request upstream through
  `responses_translation_for_route`, and a Claude upstream would then require `Responses → AnthropicMessages`, which is
  the explicit gap at `llm/mod.rs:347` and is refused for `caller` and `namespace` at `conversion/responses.rs:797`.
  Round-tripping `thinking` signatures through Responses items also risks silent loss. Fronting Claude — the primary
  motivation — would have been the worst-supported case.
- **A parallel Messages runtime** sharing only leaf modules. Rejected because it duplicates the loop, budget
  accounting, usage aggregation, limit accounting, and error taxonomy, which is precisely where divergence would be
  silent and correctness bugs subtle.

## Test Plan

- Unit-test declaration mapping for each code-execution version, `allowed_callers` normalization of
  `code_execution_20260521`, the `web_search_20260209`+ programmatic default, and the documented rejection when that
  default stands without a code-execution tool.
- Unit-test that a programmatic-only tool is absent from the model-facing `tools` array and that a direct model call
  naming it is rejected.
- Unit-test every rejection: reserved `_agentgateway_` name, unregistered custom tool, `mcp_toolset`, `mcp_servers`,
  computer and browser toolsets, `container` field, `tool_choice` forcing a non-`direct` tool,
  `disable_parallel_tool_use: true` with Programmatic Tool Calling, and a recursive `$ref` cycle.
- Unit-test `collect_model_calls` over `tool_use` blocks: object `input`, `id` as `call_id`, the serialized-arguments
  bound, empty and duplicate ids, undeclared names, and a program call mixed with direct calls.
- Unit-test `append_round_history`: message ordering, `tool_result` order matching `tool_use` order, `is_error: true`
  for application errors, and `tool_choice` reset to `{"type": "auto"}`.
- Unit-test that `thinking` with its `signature` and any `cache_control` survive an appended round byte-for-byte.
- Verify the existing runtime suite and all snapshots pass with zero churn after the `ManagedToolState` extraction.
- Unit-test the program state machine on the Messages path: one call, several sequential calls, scalar, object, and
  list results, a Python syntax error surfacing as a synthetic `tool_result` with `is_error`, a missing
  `program_output`, an undeclared tool, argument schema mismatch, and call, output, and deadline exhaustion.
- Add golden snapshots for the managed Messages event stream: a text-only final round, a final round carrying
  `thinking` and `signature`, an unmodelled block emitted whole in `content_block_start`, aggregate usage split across
  `message_start` and `message_delta`, and no `[DONE]` sentinel.
- Unit-test usage aggregation across rounds, including cache fields and saturating addition.
- Test through the proxy: zero-call, single-round, multi-round, round-limit, call-limit, model-header, model-body, and
  tool-time deadlines, provider and backend pinning across rounds, retries disabled once the runtime handles a
  response, and `stream: true` end to end.
- Test one identical Messages request against an Anthropic-format upstream and an OpenAI-completions-format upstream,
  both completing the loop. This is the direct evidence that rewriting built-ins to custom tools bought upstream
  independence.
- Unit-test `MessagesRequestExt` round-tripping of the flattened `rest` value, including a request whose `rest` is
  absent, so `tools` and `tool_choice` mutation is exercised without touching `crates/llm`.
- Add a `messages-programmatic` case to `examples/llm-tool-runtime/functional_test.py` driving the release binary
  through public `/v1/messages` against the local HTTP tool mock, asserting round count, exactly one tool-backend
  request, aggregate usage, and a client body free of reserved names.
- Keep live E2B coverage opt-in behind `AGENTGATEWAY_LIVE_TOOLS=1`; never require cloud credentials for default unit or
  functional tests.

## Open Questions

- Should a follow-up bring remote MCP to the Messages route through `mcp_servers` and `mcp_toolset`, given that
  Anthropic forbids `allowed_callers` on `mcp_toolset` and documents MCP tools as not programmatically callable? Doing
  so would be an explicit AgentGateway extension beyond their contract.
- Should Tool Search reach the Messages route by reusing `mcp_toolset`'s `default_config.defer_loading`, rather than
  introducing a second deferral vocabulary?
- Should AgentGateway ever offer true Anthropic pause and resume with client-owned tools, and what container registry,
  expiry, and replay-transcript persistence would that require?
- When the upstream is Claude and natively supports Programmatic Tool Calling, should an operator be able to choose
  native passthrough over gateway execution, and how would managed-tool authorization be expressed in that mode?
