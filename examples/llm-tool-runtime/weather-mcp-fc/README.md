# WeatherAPI Remote MCP Web Function

This directory packages a Remote MCP server for Alibaba Cloud Function
Compute. It is based on the tool contract and fixed WeatherAPI REST origin in
[`weatherapicom/weatherapi-mcp`](https://github.com/weatherapicom/weatherapi-mcp),
but replaces stdio with stateless MCP Streamable HTTP.

The initial server exposes only `get_current_weather` and `get_forecast`. It
listens on `0.0.0.0:${CAPort:-9000}` and serves authenticated `POST /mcp` plus
unauthenticated `GET /healthz`. Every MCP request gets a fresh stateless SDK
transport, so Function Compute does not need session affinity.
Both tools advertise an MCP `outputSchema` and return matching
`structuredContent`, while retaining JSON text content for clients that do not
consume structured MCP results. Forecast responses project the provider payload
to each date's minimum and maximum temperature, avoiding replaying bulky hourly
data through Programmatic Tool Calling.

## Authentication boundary

| Function environment variable | Purpose |
|---|---|
| `WEATHERAPI_KEY` | Sent only to the fixed `https://api.weatherapi.com/v1` origin |
| `INGRESS_BEARER_TOKEN` | Required as `Authorization: Bearer ...` on `/mcp` |

The FC Trigger uses platform `authType: anonymous` because Remote MCP clients
do not sign Alibaba Cloud control-plane requests. The application still
validates `INGRESS_BEARER_TOKEN` before parsing MCP JSON.

## Deploy from `.env`

Required repository-root `.env` variables:

```text
ALIBABA_CLOUD_ACCESS_KEY_ID=...
ALIBABA_CLOUD_ACCESS_KEY_SECRET=...
ALIBABA_CLOUD_REGION=cn-hangzhou
WEATHERAPI_KEY=...
```

Optional names are `ALIBABA_CLOUD_SECURITY_TOKEN`,
`ALIBABA_CLOUD_FC_ENDPOINT`, `FC_WEATHER_MCP_FUNCTION_NAME`,
`FC_WEATHER_MCP_TRIGGER_NAME`, and `FC_WEATHER_MCP_TOKEN`.

```sh
uv run --with alibabacloud_fc20230330 \
  python examples/llm-tool-runtime/weather-mcp-fc/deploy.py
```

The deployment utility installs locked production dependencies, builds a
deterministic ZIP, and creates or updates one FC Custom Runtime Web Function
and HTTP Trigger. It then writes the ignored local values
`FC_WEATHER_MCP_URL` and `FC_WEATHER_MCP_TOKEN` without printing the token.
The SDK uses its official regional `fcv3` endpoint unless an explicit endpoint
override is configured.

## Direct MCP smoke requests

```sh
set -a
. ./.env
set +a

curl --fail-with-body --max-time 30 \
  -H "Authorization: Bearer ${FC_WEATHER_MCP_TOKEN}" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"1.0"}}}' \
  "${FC_WEATHER_MCP_URL}"

curl --fail-with-body --max-time 30 \
  -H "Authorization: Bearer ${FC_WEATHER_MCP_TOKEN}" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  "${FC_WEATHER_MCP_URL}"

curl --fail-with-body --max-time 30 \
  -H "Authorization: Bearer ${FC_WEATHER_MCP_TOKEN}" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  --data '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_current_weather","arguments":{"q":"Beijing","aqi":"no"}}}' \
  "${FC_WEATHER_MCP_URL}"
```

## Tests and limits

```sh
cd examples/llm-tool-runtime/weather-mcp-fc
npm ci
npm test
python3 test_deploy.py
```

The server caps MCP bodies at 64 KiB, provider responses at 512 KiB,
credentials at 4 KiB, location queries at 256 characters, forecast days at
1–14, and WeatherAPI calls at 10 seconds. Redirects are rejected and provider
errors are sanitized. The upstream project declares the MIT license.
