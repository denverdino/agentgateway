# Responses tool runtime example

This example gives AgentGateway a static, operator-owned registry for the
Responses API `web_search` and `code_interpreter` built-ins. The model sees the
built-ins requested by the client; AgentGateway rewrites them to reserved
function calls, invokes the registered backends, and makes the
next model round with canonical `function_call_output` items. Only the final
model Response is returned to the client.

The checked-in [`config.yaml`](config.yaml) is runnable after its required
environment variables are set. It contains no deployment URL or credential.
AgentGateway's config loader expands `$VARIABLE` references and fails startup
when a referenced variable is missing.

## Deployment order

Deploy in this exact order so no component is configured against a guessed
endpoint or credential.

1. **Tavily API key.** Create a Tavily API key and keep it in Function Compute
   server-side environment variables or secrets.
2. **Web Search Function.** Deploy [`web-search-fc`](web-search-fc) as a Python
   HTTP Function with handler `handler.handler`. Configure the real HTTPS
   `TAVILY_API_KEY` and `INGRESS_BEARER_TOKEN` in the Function. This artifact
   has no third-party runtime dependency. Record the authenticated Function
   URL and an operator-owned ingress bearer token only after deployment
   succeeds.
3. **E2B-compatible Sandbox.** Create an API key and record the E2B-compatible
   control-plane URL and Sandbox domain. AgentGateway uses these values
   directly; no Function Compute Sandbox adapter is deployed.
4. **Weather Remote MCP Function.** Deploy
   [`weather-mcp-fc`](weather-mcp-fc) as an FC Custom Runtime Web Function.
   Its deployment utility records `FC_WEATHER_MCP_URL` and
   `FC_WEATHER_MCP_TOKEN` for the live Remote MCP case.
5. **Static AgentGateway YAML.** Supply all variables below, validate the
   checked-in YAML, and start AgentGateway. Do not substitute request-provided
   endpoints or credentials into the registry.
6. **Smoke request.** Wait for readiness, then send the JavaScript or curl
   request below. Confirm two model rounds when the model selects tools, final
   output only, and the expected bounded telemetry.

The gateway process requires these values:

| Variable | Required value |
|---|---|
| `AGENTGATEWAY_API_KEY` | Client-facing gateway API key accepted by the strict ingress policy |
| `AGENTGATEWAY_UPSTREAM_MODEL` | Exact upstream Responses-capable model name |
| `AGENTGATEWAY_UPSTREAM_API_KEY` | Upstream provider API key |
| `AGENTGATEWAY_UPSTREAM_BASE_URL` | Real upstream API base URL, including `/v1` when the provider requires it |
| `FC_WEB_SEARCH_URL` | Deployed Web Search Function HTTPS URL |
| `FC_WEB_SEARCH_TOKEN` | Web Search Function ingress bearer token |
| `E2B_API_KEY` | E2B-compatible Sandbox API key; sent only as `X-API-Key` to the control plane |
| `E2B_API_URL` | E2B-compatible control-plane origin, for example `https://api.cn-hangzhou.e2b.fc.aliyuncs.com` |
| `E2B_DOMAIN` | Sandbox data-plane domain, for example `cn-hangzhou.e2b.fc.aliyuncs.com` |

Validate and run the release binary from the repository root:

```sh
: "${AGENTGATEWAY_API_KEY:?set the client-facing gateway API key}"
: "${AGENTGATEWAY_UPSTREAM_MODEL:?set the upstream model}"
: "${AGENTGATEWAY_UPSTREAM_API_KEY:?set the upstream API key}"
: "${AGENTGATEWAY_UPSTREAM_BASE_URL:?set the upstream API base URL}"
: "${FC_WEB_SEARCH_URL:?set the deployed Web Search Function URL}"
: "${FC_WEB_SEARCH_TOKEN:?set the Web Search Function bearer token}"
: "${E2B_API_KEY:?set the E2B API key}"
: "${E2B_API_URL:?set the E2B API URL}"
: "${E2B_DOMAIN:?set the E2B Sandbox domain}"

./target/release/agentgateway --validate-only \
  --file examples/llm-tool-runtime/config.yaml
./target/release/agentgateway \
  --file examples/llm-tool-runtime/config.yaml
```

Readiness is served at `http://127.0.0.1:19001/healthz/ready`; the Responses API
is served at `http://127.0.0.1:4000/v1/responses`.

The checked-in `llm.policies.apiKey` policy is `strict` and accepts
`AGENTGATEWAY_API_KEY` only from `Authorization: Bearer ...`. A missing or
incorrect client key is rejected before the upstream model. The OpenAI SDK's
`apiKey` option below supplies this enforced gateway credential; it is distinct
from `AGENTGATEWAY_UPSTREAM_API_KEY`, which AgentGateway uses only for the
configured provider.

## Client smoke requests

This uses the current OpenAI JavaScript Responses API shape. Set
`AGENTGATEWAY_BASE_URL` to the gateway API base ending in `/v1`, and keep the
client-facing gateway key in `AGENTGATEWAY_API_KEY`.

```js
import OpenAI from "openai";

const { AGENTGATEWAY_BASE_URL, AGENTGATEWAY_API_KEY } = process.env;
if (!AGENTGATEWAY_BASE_URL || !AGENTGATEWAY_API_KEY) {
  throw new Error("set AGENTGATEWAY_BASE_URL and AGENTGATEWAY_API_KEY");
}

const client = new OpenAI({
  baseURL: AGENTGATEWAY_BASE_URL,
  apiKey: AGENTGATEWAY_API_KEY,
});

const response = await client.responses.create({
  model: "smart",
  input: "Search for today's market data and calculate the average.",
  tools: [
    { type: "web_search" },
    { type: "code_interpreter", container: { type: "auto" } },
  ],
  stream: false,
});

console.log(response.output_text);
```

The curl equivalent is:

```sh
: "${AGENTGATEWAY_BASE_URL:?set the gateway API base ending in /v1}"
: "${AGENTGATEWAY_API_KEY:?set the gateway API key}"

curl --fail-with-body --max-time 125 \
  -H "Authorization: Bearer ${AGENTGATEWAY_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data '{"model":"smart","input":"Search for today’s market data and calculate the average.","tools":[{"type":"web_search"},{"type":"code_interpreter","container":{"type":"auto"}}],"stream":false}' \
  "${AGENTGATEWAY_BASE_URL}/responses"
```

Remote MCP uses the same Responses endpoint. The server must expose MCP
Streamable HTTP over HTTPS. AgentGateway discovers the allowed tools before
the first model round, maps them to request-local functions, and reuses one MCP
session for every call to that server in this Response:

```sh
: "${AGENTGATEWAY_BASE_URL:?set the gateway API base ending in /v1}"
: "${AGENTGATEWAY_API_KEY:?set the gateway API key}"
: "${REMOTE_MCP_URL:?set the HTTPS MCP Streamable HTTP endpoint}"
: "${REMOTE_MCP_TOKEN:?set the MCP OAuth access token}"

curl --fail-with-body --max-time 125 \
  -H "Authorization: Bearer ${AGENTGATEWAY_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data "$(jq -n \
    --arg url "${REMOTE_MCP_URL}" \
    --arg token "${REMOTE_MCP_TOKEN}" \
    '{
      model: "smart",
      input: "Use the calendar server to list my next events.",
      tools: [{
        type: "mcp",
        server_label: "calendar",
        server_description: "Calendar tools",
        server_url: $url,
        authorization: $token,
        allowed_tools: ["list_events"],
        require_approval: "never"
      }],
      parallel_tool_calls: true,
      stream: false
    }')" \
  "${AGENTGATEWAY_BASE_URL}/responses"
```

The current implementation intentionally rejects `connector_id`, approval
flows that require a client round trip, private/non-HTTPS production URLs,
legacy HTTP/SSE, and server-initiated sampling or elicitation. Both
`require_approval: "auto"` and `"never"` execute inside AgentGateway without an
approval interruption; `"always"` and per-tool approval policies are rejected.
The MCP authorization value is never forwarded to the model or echoed in the
final Response.

AgentGateway-managed Programmatic Tool Calling is an opt-in Python extension
for upstream models that do not provide OpenAI's native JavaScript/V8 runtime.
Declare `programmatic_tool_calling` and set `allowed_callers` to
`["programmatic"]` or `["direct", "programmatic"]` on Web Search, Code
Interpreter, or Remote MCP declarations. The generated Python calls one
authorized tool at a time with `tools.call(name, arguments)` and completes with
`program_output(value)`. Remote MCP names use
`{server_label}.{tool_name}`, for example:

```python
forecast = tools.call("weather.get_forecast", {"q": "Shanghai", "days": 3})
program_output(forecast["structuredContent"])
```

When a Remote MCP tool advertises `outputSchema`, AgentGateway includes it as
`output_schema` in the programmatic tool catalog. The schema describes the
`structuredContent` field of the successful object returned by `tools.call`.

Each unresolved call is executed through AgentGateway's existing authorized
backend and the Python program is replayed with a verified transcript. Replay
divergence terminates the request instead of retrying after possible tool side
effects. Large nested results may be truncated further to fit the aggregate
replay budget. If that budget cannot hold another bounded result, the model
receives a `program_replay_limit` tool error and no additional nested tool is
executed. Python runs through the configured E2B Code Interpreter backend, even when the
program only calls Web Search or MCP. The Sandbox keeps E2B's default network
behavior; AgentGateway does not inject backend URLs, tokens, or E2B credentials
into the generated program.

Managed execution is foreground-only. Active managed requests support
`stream: true`, but buffer the internal model/tool loop and emit only the final
Response as OpenAI-compatible SSE events. This preserves the streaming wire
contract without exposing intermediate Tool Calls or rewritten reserved tool
definitions; it does not provide an
incremental time-to-first-token benefit. Active requests reject background
mode, conversation state, a mix of managed and unmanaged tools, and client declarations using the reserved
`_agentgateway_` prefix with a sanitized HTTP 400 response. A managed backend
infrastructure or cleanup failure is HTTP 502. The one absolute runtime
deadline covers model headers, model body reads, and tool work; expiration is
HTTP 504. Client-owned function tools remain a one-model-call passthrough path.
Managed SSE delta events use obfuscation by default and honor
`stream_options.include_obfuscation: false`; that option is not forwarded to
the internal non-streaming model rounds.

## Sandbox reuse and isolation boundary

When one model Response emits multiple Python calls, AgentGateway creates
**one E2B Sandbox for that Response**, executes calls sequentially, and creates a fresh Python context for
each call. Call IDs and results retain model order, while Python interpreter
variables are isolated between contexts. A later Response creates a different
Sandbox.

The gateway enforces the E2B boundary before any control-plane call: one
same-Response Sandbox batch contains at most eight executions, each code value
is at most 32 KiB, and each non-empty `call_id` is at most 256 UTF-8 bytes.
These specialized checks apply in addition to the runtime-wide argument and
tool-call limits. The checked-in E2B backend timeout is 120 seconds, matching
`totalTimeout`; time already spent in model rounds reduces the time available
for create, context, execute, and cleanup calls. An `e2b` backend timeout may
be at most 24 hours.

Fresh Python contexts do not mean fresh operating systems. Calls sharing the
Sandbox can communicate through filesystem or process state. Do not use this
mode for workloads that require OS isolation between code calls. The gateway
does not expose command, filesystem, package-install, artifact, template,
container, endpoint, credential, or timeout selection through its request.

Deploy the Sandbox without business VPC attachment, OSS/NAS mounts, business
roles, business secrets, or unrestricted public egress. Use a dedicated API
key, fixed template and region, finite Function/Sandbox memory, and finite
lifetime. Sandbox creation happens once per validated batch. Cleanup is a
bounded E2B `DELETE`, with exactly one bounded retry only after a failed first attempt;
a definitive cleanup failure discards completed results and becomes HTTP 502.

The direct backend caps NDJSON ingestion and normalized output at
`maxOutputBytes`. Its wrapper bounds ordinary Python stdout/stderr, but raw
file-descriptor writes, native modules, rich results, and shared OS process
state remain outside that wrapper. Platform
memory and lifetime limits remain the hard fail-stop controls for those
bypasses.

Runtime metrics and spans contain only configured tool names and closed
backend/outcome/operation values. They do not label query, code, call ID,
stdout/stderr, result URL, endpoint, credential, Sandbox ID, template, or
request ID.

## Hermetic release-binary functional tests

The standard-library harness starts the real release executable, three local
loopback mocks (model, Web Search, and E2B control/data planes), and a temporary static
config. Every assertion travels through public `/v1/responses`; the gateway is
never simulated in-process. Temporary files and children are cleaned up on
success, failure, and interruption, and diagnostics never dump credentials,
submitted code, or tool output. Held reservations make the gateway/readiness
ports distinct before launch; bounded retries cover the remaining release-and-
bind race. A missing and incorrect client key must return 401 without reaching
the model, then an authenticated identity probe confirms the selected child and
temporary config. Every mock enforces its fixed path, bearer auth, exact JSON
content type, and selected upstream model. The overlap case uses a bounded
two-backend barrier, so both backends must observe each other before completing.

```sh
cargo build --release -p agentgateway-app
python3 examples/llm-tool-runtime/functional_test.py \
  --binary ./target/release/agentgateway
```

List or select cases individually:

```sh
python3 examples/llm-tool-runtime/functional_test.py --list-cases
python3 examples/llm-tool-runtime/functional_test.py \
  --binary ./target/release/agentgateway \
  --case dual-tool-overlap
```

See [`FUNCTIONAL_TEST_CASES.md`](FUNCTIONAL_TEST_CASES.md) for the exact
assertions and the additional Rust, Python, and opt-in live coverage. These are
reproducible hermetic local/release checks and require no cloud credential;
this example does not add or claim repository CI workflow wiring.

## Real-backend functional test application

[`live_functional_test.py`](live_functional_test.py) starts an isolated
AgentGateway release process and tests its public Responses API against the
real LLM model, deployed FC Web Search Function, and E2B-compatible
Sandbox. Unlike `functional_test.py`, this application does not mock any
upstream or tool backend.

The generated temporary config follows the regular system-config shape:
`gateways.default` owns a random loopback port, `llm.gateways` attaches the LLM
routes, and the `bailian` provider is referenced by the configured model. It
does not open the system database, enable the UI, modify
`~/.config/agentgateway/config.yaml`, or bind port 4000. The child process and
temporary config are removed on success, failure, and interruption.

Known values are loaded from the process environment first and the repository
root `.env` second. The required names are:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI-compatible model credential used by the configured provider |
| `FC_WEB_SEARCH_URL` | Deployed FC Web Search Function URL |
| `FC_WEB_SEARCH_TOKEN` | FC Function ingress bearer token |
| `E2B_API_KEY` | E2B-compatible Sandbox API key |
| `E2B_API_URL` | E2B-compatible control-plane origin |
| `E2B_DOMAIN` | E2B Sandbox data-plane domain |
| `FC_WEATHER_MCP_URL` | Deployed FC WeatherAPI Remote MCP `/mcp` endpoint |
| `FC_WEATHER_MCP_TOKEN` | Remote MCP application ingress bearer token |

`AGENTGATEWAY_LIVE_MODEL` optionally overrides the default
`qwen3.6-flash`, and `AGENTGATEWAY_LIVE_UPSTREAM_BASE_URL` optionally overrides
the default LLM compatible-mode `/v1` endpoint. The harness only loads
these allowlisted names and never prints credentials or backend response
bodies on failure.

Build AgentGateway and run all live cases:

```sh
cargo build --release -p agentgateway-app
python3 examples/llm-tool-runtime/live_functional_test.py \
  --binary ./target/release/agentgateway
```

Run or inspect cases individually:

```sh
python3 examples/llm-tool-runtime/live_functional_test.py --list-cases
python3 examples/llm-tool-runtime/live_functional_test.py \
  --binary ./target/release/agentgateway \
  --case code-interpreter \
  --show-output
```

The cases are `web-search`, `code-interpreter`, `combined`,
`programmatic-server-tools`, `programmatic-mcp-weather`,
`streaming-tool-runtime`, `remote-mcp-weather`, and
`tool-search-weather-mcp`. The non-streaming cases
check the HTTP response and final marker, then read the local AgentGateway
metrics endpoint to prove the expected `http`, `e2b`, or `remote_mcp` backend
recorded a successful tool call. `remote-mcp-weather` discovers the allowlisted
`get_forecast` and `get_current_weather` tools from the authenticated FC Web
Function configured by `FC_WEATHER_MCP_URL`, queries Beijing weather, and
requires the final marker `WEATHER_MCP_BEIJING_OK`.
`tool-search-weather-mcp` declares the same server with `defer_loading: true`
alongside `{"type": "tool_search"}` and sends no `tool_choice`, so its
declarations are withheld from the first model round. The live model has to call
tool search, receive the injected declaration, and only then query Shanghai
weather; the case requires an observed temperature plus the marker
`TOOL_SEARCH_WEATHER_MCP_OK`, and asserts the declarations echo back with
`defer_loading` intact and no reserved `_agentgateway` name. A successful
`remote_mcp` metric is therefore only reachable through a search that injected
the tool.
`programmatic-mcp-weather` exposes only `weather.get_forecast` to Python,
requests exactly three forecast days (the WeatherAPI Free-plan window), and
requires the day with the highest daily maximum and the day with the lowest
daily minimum (choosing the earliest date for ties), plus
`PROGRAMMATIC_MCP_WEATHER_3DAY_OK` in the final answer. `--show-output` prints
each case's final answer, and runs AgentGateway with `LOG_FORMAT=json` plus
`RUST_LOG=info,agentgateway::llm::tool_runtime::runner=trace` so the generated
Python is read back from the child log and printed before the answer. That TRACE
event is the only place the program source is exposed — the debug level records
sizes only. The source inlines the arguments the model passed to tools, so use
this opt-in output only with non-sensitive test prompts.
The `combined` request sets `parallel_tool_calls: true`; AgentGateway runs
independent Tool Backend operations concurrently up to `maxParallelToolCalls`
and restores model call order before the next round. Setting the field to
`false` executes operations serially, while omitting it defaults to `true`.
`streaming-tool-runtime` sends `stream: true` with Web Search and verifies the
SSE lifecycle, final output marker, and successful Web Search backend metric.
Run only the live WeatherAPI MCP case with:

```sh
uv run --with alibabacloud_fc20230330 \
  python examples/llm-tool-runtime/weather-mcp-fc/deploy.py
python3 examples/llm-tool-runtime/live_functional_test.py \
  --binary ./target/release/agentgateway \
  --case remote-mcp-weather \
  --show-output

python3 examples/llm-tool-runtime/live_functional_test.py \
  --binary ./target/release/agentgateway \
  --case programmatic-mcp-weather \
  --show-output

python3 examples/llm-tool-runtime/live_functional_test.py \
  --binary ./target/release/agentgateway \
  --case tool-search-weather-mcp \
  --show-output
```

Run the
hermetic harness unit tests without cloud access using:

```sh
python3 examples/llm-tool-runtime/test_live_functional_test.py
```

### OpenAI Python SDK streaming case

[`openai_sdk_test.py`](openai_sdk_test.py) connects to an already-running
AgentGateway through the official OpenAI Python SDK. It requests Web Search and
Code Interpreter together with `parallel_tool_calls=True` and `stream=True`,
then verifies the typed event lifecycle, the terminal completed response, and
that concatenated `response.output_text.delta` values equal the final
`response.output_text`.

With AgentGateway listening on its normal port 4000, run:

```sh
uv run --with openai \
  python examples/llm-tool-runtime/openai_sdk_test.py
```

For a gateway using strict client authentication or a different listener/model:

```sh
AGENTGATEWAY_API_KEY="$CLIENT_API_KEY" \
AGENTGATEWAY_BASE_URL="http://127.0.0.1:4000/v1" \
AGENTGATEWAY_LIVE_MODEL="qwen3.6-flash" \
uv run --with openai \
  python examples/llm-tool-runtime/openai_sdk_test.py
```

The script intentionally uses `AGENTGATEWAY_API_KEY` for the client-facing
credential. `OPENAI_API_KEY` remains the upstream provider credential consumed
by AgentGateway and is not sent by this SDK test.

Run its offline contract tests without the OpenAI package or cloud access:

```sh
python3 examples/llm-tool-runtime/test_openai_sdk_test.py
```

## Opt-in live backend tests

The ignored Rust tests call the deployed Web Search Function and the direct E2B
backend. They use a 30-second absolute limit for each call; AgentGateway owns
the create/context/kill lifecycle and cleanup. The tests read only the known keys below from the process environment and,
when absent there, a repository-root `.env`; they never enumerate, print, or
expand values.

```sh
export AGENTGATEWAY_LIVE_TOOLS=1
: "${FC_WEB_SEARCH_URL:?set the deployed Web Search Function URL}"
: "${FC_WEB_SEARCH_TOKEN:?set the Web Search Function bearer token}"
: "${E2B_API_KEY:?set the E2B API key}"
: "${E2B_API_URL:?set the E2B API URL}"
: "${E2B_DOMAIN:?set the E2B Sandbox domain}"

cargo test -p agentgateway llm::tests::task12_live_tools \
  --lib -- --ignored --nocapture
```

If the gate or an endpoint variable is missing, each test prints `SKIPPED`
with missing variable **names only** and makes no network call. These smokes do
not deploy, publish, or modify any Alibaba Cloud resource.

## Component tests

The Web Search Function suite is hermetic by default; direct E2B protocol
coverage lives in the Rust Tool Runtime tests:

```sh
python3 examples/llm-tool-runtime/web-search-fc/test_handler.py
cargo test -p agentgateway llm::tool_runtime --lib
```

The WeatherAPI Remote MCP component has separate Node and deployment tests:

```sh
cd examples/llm-tool-runtime/weather-mcp-fc
npm test
python3 test_deploy.py
```
