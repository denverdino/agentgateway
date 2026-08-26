# Tavily Web Search Function

This Alibaba Cloud Function Compute HTTP function implements AgentGateway's
typed Web Search backend using Tavily. It has no third-party
runtime dependencies.

## Deploy and configure

Deploy this directory as a Python Function Compute HTTP function with the
handler entry point `handler.handler`.  Configure the following server-side
environment variables in Function Compute; do not put the API key in a client
request, AgentGateway configuration, source code, or logs.

```text
TAVILY_API_KEY=tvly-YOUR_TAVILY_API_KEY
INGRESS_BEARER_TOKEN=YOUR_AGENTGATEWAY_TO_FUNCTION_TOKEN
```

The handler always posts to the fixed `https://api.tavily.com/search` endpoint
and sends `TAVILY_API_KEY` only in the server-side
`Authorization: Bearer ...` header. The function accepts no caller-controlled
endpoint, API key, or timeout. It disables Tavily answers, raw content, and
images so the model receives only the gateway's bounded normalized results.

Deploy the built-in Python runtime with an FC 3.0 anonymous HTTP Trigger that
allows only `POST`. The handler unwraps the FC 3.0 HTTP event envelope and
requires `Authorization: Bearer <INGRESS_BEARER_TOKEN>` before it parses the
request or calls Tavily. Anonymous platform ingress is therefore not
anonymous application access. Configure the same secret as
`FC_WEB_SEARCH_TOKEN` on AgentGateway's `http` backend. Direct
event invocation remains available for trusted Function Compute control-plane
tests and does not use the HTTP bearer boundary.

## Gateway request and response

AgentGateway sends a flattened, trusted JSON object:

```json
{
  "query": "current AgentGateway announcements",
  "allowed_domains": ["help.aliyun.com"],
  "search_context_size": "medium",
  "user_location": {"type": "approximate", "country": "CN"}
}
```

Only those four fields are accepted. `query` is required. `allowed_domains`
contains DNS names only; matching is exact or at a dot boundary, so
`example.com` permits `news.example.com` but never `badexample.com`.
`search_context_size` maps `low`, `medium`, and `high` to Tavily
`fast`/3, `basic`/5, and `advanced`/10 respectively. `allowed_domains` is sent
as Tavily `include_domains` and independently enforced again on normalized
results. Location is validated but not forwarded in this release.

On success the handler returns the typed Web Search body below. AgentGateway's
AgentGateway's HTTP Tool backend validates it and adds `ok: true` before returning model-visible
tool output.

```json
{
  "results": [
    {
      "title": "Result title",
      "url": "https://help.aliyun.com/example",
      "snippet": "Result excerpt",
      "published_at": null
    }
  ]
}
```

Validation and upstream service problems remain model-visible application
errors with HTTP 200 semantics:

```json
{
  "ok": false,
  "error": {
    "type": "provider_failure",
    "message": "The web search provider request failed.",
    "retryable": true
  },
  "stdout": "",
  "stderr": ""
}
```

The error response intentionally excludes provider response bodies, URLs, and
credentials.

## Safety limits

`handler.py` names and enforces these UTF-8/network caps:

- `MAX_EVENT_BYTES` (16 KiB) and `MAX_QUERY_BYTES` (4 KiB)
- `MAX_PROVIDER_RESPONSE_BYTES` (512 KiB), read once with a one-byte overflow
  check before JSON parsing
- `MAX_ALLOWED_DOMAINS` (50) and `MAX_PROVIDER_RESULT_ITEMS` (100)
- `MAX_TITLE_BYTES` (1 KiB), `MAX_SNIPPET_BYTES` (4 KiB), and
  `MAX_RESULT_URL_BYTES` (4 KiB)
- `MAX_SERIALIZED_OUTPUT_BYTES` (32 KiB), enforced by UTF-8-aware field
  truncation and stable tail-result removal without emitting malformed JSON

The provider and every retained result URL must be HTTPS. Redirects are
rejected, provider response bodies are bounded before decoding, non-2xx and
invalid JSON results are sanitized, and normalized HTTPS URLs are de-duplicated
in provider order.

## Local smoke test

The tests mock `urllib` and do not call Tavily or Alibaba Cloud. From the repository
root, run:

```sh
python3 -m pytest examples/llm-tool-runtime/web-search-fc/test_handler.py -q
```

If pytest is unavailable in the local interpreter, the suite is also directly
executable with the standard library:

```sh
python3 examples/llm-tool-runtime/web-search-fc/test_handler.py
```

For a live smoke check, set a real `TAVILY_API_KEY`, then invoke the deployed
HTTP function through its authenticated Function Compute ingress. Do not put
either API key in a request body, source file, or log.
